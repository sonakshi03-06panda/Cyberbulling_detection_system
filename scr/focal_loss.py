import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    """Focal Loss for multi-label classification with imbalanced data.
    Reduces weight for easy negatives, focuses on hard positives/negatives."""

    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        """
        Args:
            alpha: weighting factor for positive class (0-1), default 0.25
            gamma: focusing parameter (0-5), higher = more focus on hard examples, default 2.0
            pos_weight: per-class weight for positive examples
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, labels):
        """Compute focal loss.
        Args:
            logits: (batch_size, num_classes) raw model outputs
            labels: (batch_size, num_classes) binary labels (0 or 1)
        """
        # Compute BCE with logits
        bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none", weight=self.pos_weight)

        # Compute probabilities
        p = torch.sigmoid(logits)

        # Compute focal term: (1-p_t)^gamma
        p_t = p * labels + (1 - p) * (1 - labels)
        focal_term = (1 - p_t) ** self.gamma

        # Combine alpha weighting
        alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
        focal_loss = alpha_t * focal_term * bce

        return focal_loss.mean()
