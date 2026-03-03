import pandas as pd
import numpy as np
import re


class TextAugmenter:
    """Simple text augmentation for imbalanced toxic comment data."""

    @staticmethod
    def synonym_replacement(text, n=2, seed=42):
        """Replace random words with synonyms (simple version)."""
        np.random.seed(seed)
        synonyms = {
            "bad": "terrible, awful, poor",
            "good": "great, excellent, fine",
            "hate": "despise, abhor, dislike",
            "love": "adore, cherish, like",
            "stupid": "dumb, idiotic, moronic",
            "smart": "clever, intelligent, bright",
        }
        words = text.split()
        for _ in range(n):
            if len(words) > 0:
                idx = np.random.randint(len(words))
                word = words[idx].lower().strip(".,!?;:")
                if word in synonyms:
                    words[idx] = np.random.choice(synonyms[word].split(", "))
        return " ".join(words)

    @staticmethod
    def random_swap(text, n=2, seed=42):
        """Randomly swap two words in the sentence."""
        np.random.seed(seed)
        words = text.split()
        for _ in range(min(n, len(words) // 2)):
            idx1, idx2 = np.random.choice(len(words), 2, replace=False)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return " ".join(words)

    @staticmethod
    def random_deletion(text, p=0.1, seed=42):
        """Randomly delete words with probability p."""
        np.random.seed(seed)
        if len(text.split()) == 1:
            return text
        words = [w for w in text.split() if np.random.rand() > p]
        return " ".join(words) if words else text

    @staticmethod
    def back_translation(text):
        """Placeholder for back-translation (requires a translation API)."""
        return text  # In production, use Google Translate or mBART


def augment_toxic_examples(csv_path, output_path, target_label_ratios=None):
    """
    Augment text for underrepresented toxic labels.
    For each label with low positive rate, oversample by augmenting existing examples.

    Args:
        csv_path: path to input train.csv
        output_path: path to save augmented training data
        target_label_ratios: dict of {label: target_positive_ratio}
    """
    df = pd.read_csv(csv_path)
    LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    augmenter = TextAugmenter()

    if target_label_ratios is None:
        # Aim for 10% positive rate per label if it's currently lower
        target_label_ratios = {label: 0.10 for label in LABELS}

    augmented_rows = []
    for label in LABELS:
        pos_count = (df[label] == 1).sum()
        pos_rate = pos_count / len(df)
        target_positive = int(len(df) * target_label_ratios[label])

        if pos_count < target_positive:
            deficit = target_positive - pos_count
            positive_examples = df[df[label] == 1]

            for _ in range(deficit):
                # Sample a random positive example and augment it
                sample = positive_examples.sample(1).iloc[0].copy()
                aug_text = augmenter.synonym_replacement(sample["comment_text"], n=1)
                augmented_rows.append(sample.copy())
                augmented_rows[-1]["comment_text"] = aug_text

            print(f"{label}: {pos_count} → {target_positive} (added {deficit} augmented samples)")

    augmented_df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    augmented_df.to_csv(output_path, index=False)
    print(f"\nAugmented dataset saved to {output_path}")
    print(f"Original size: {len(df)}, Augmented size: {len(augmented_df)}")


if __name__ == "__main__":
    augment_toxic_examples("data/train.csv", "data/train_augmented.csv")
