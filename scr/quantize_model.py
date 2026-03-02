import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import os


def quantize_model_dynamic(model_path, output_path):
    """
    Apply dynamic INT8 quantization to the model.
    Reduces model size ~4x, inference ~2-3x faster, minimal accuracy loss.
    """
    print(f"Loading model from {model_path}...")
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    
    # Original model size
    model.eval()
    original_size = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Original model parameters: {original_size:.2f}M")
    
    # Apply dynamic quantization (INT8 on weights only)
    print("Applying dynamic INT8 quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={torch.nn.Linear},  # Quantize linear layers
        dtype=torch.qint8
    )
    
    # Save quantized model
    os.makedirs(output_path, exist_ok=True)
    quantized_model.save_pretrained(output_path)
    
    # Also save tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"Quantized model saved to {output_path}")
    
    # Check file sizes
    import glob
    orig_safetensors = glob.glob(f"{model_path}/*.safetensors")
    quant_safetensors = glob.glob(f"{output_path}/*.safetensors")
    
    if orig_safetensors and quant_safetensors:
        orig_size = os.path.getsize(orig_safetensors[0]) / 1e6
        quant_size = os.path.getsize(quant_safetensors[0]) / 1e6
        print(f"Model file size: {orig_size:.2f}MB → {quant_size:.2f}MB ({100*quant_size/orig_size:.1f}%)")


def test_quantized_inference(model_path, tokenizer_path, text="This is a test comment."):
    """Test inference speed with quantized model."""
    import time
    
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_path)
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Warm up
    with torch.no_grad():
        _ = model(**inputs)
    
    # Time inference
    n_runs = 100
    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(**inputs)
    elapsed = (time.time() - start) / n_runs * 1000
    
    print(f"Average inference time: {elapsed:.2f}ms per sample")


if __name__ == "__main__":
    quantize_model_dynamic("models/final_model", "models/final_model_quantized")
    print("\nTesting quantized model inference speed...")
    test_quantized_inference("models/final_model_quantized", "models/final_model_quantized")
