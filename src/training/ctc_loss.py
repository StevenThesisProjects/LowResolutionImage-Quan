"""Hard-Character Confusion-Aware CTC Loss with Prototype Angular Margin Regularization (Stage 1b / Stage 2)."""
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# Comprehensive list of visually ambiguous character pairs in low-resolution license plates:
# 1. Cross-type pairs (Mercosur/Brazilian 5th-character letter/digit ambiguity)
# 2. Intra-digit topology lookalikes (e.g. 6<->8, 8<->0, 2<->7, 1<->3)
# 3. Intra-letter topology lookalikes (e.g. B<->D, H<->M, V<->Y, O<->Q)
HARD_CONFUSION_PAIRS: List[Tuple[str, str]] = [
    # Classic cross-category lookalikes
    ("8", "B"),
    ("0", "D"),
    ("0", "O"),
    ("5", "S"),
    ("2", "Z"),
    ("1", "I"),
    ("6", "G"),
    # High-frequency intra-digit confusions on Scenario-B
    ("6", "8"),
    ("8", "0"),
    ("8", "9"),
    ("2", "7"),
    ("1", "3"),
    ("1", "7"),
    ("5", "6"),
    ("5", "9"),
    ("4", "6"),
    ("8", "3"),
    # High-frequency intra-letter confusions on Scenario-B
    ("B", "D"),
    ("B", "A"),
    ("B", "R"),
    ("H", "M"),
    ("H", "N"),
    ("N", "M"),
    ("V", "Y"),
    ("W", "V"),
    ("W", "M"),
    ("X", "V"),
    ("X", "K"),
    ("Y", "T"),
    ("O", "D"),
    ("O", "Q"),
    ("C", "G"),
    ("P", "R"),
]


class HardCharacterCTCLoss(nn.Module):
    """
    Discriminative CTC Objective for Low-Resolution License Plate Recognition:
    1. Per-sample Hard-Character Focal CTC Loss:
       Upweights sequences containing visually ambiguous characters and sequences
       with higher alignment uncertainty.
    2. Prototype Angular Margin Regularization (L_margin):
       Penalizes cosine similarity > margin `m` between classifier weight vectors
       (w_a, w_b) for all confusing pairs (a, b) in HARD_CONFUSION_PAIRS, forcing
       orthogonal/well-separated decision boundaries in embedding space.
    """

    def __init__(
        self,
        char2idx: Dict[str, int],
        blank: int = 0,
        alpha_hard: float = 0.4,
        gamma_focal: float = 0.5,
        margin: float = 0.10,
        lambda_margin: float = 2.0,
    ):
        super().__init__()
        self.char2idx = char2idx
        self.blank = blank
        self.alpha_hard = alpha_hard
        self.gamma_focal = gamma_focal
        self.margin = margin
        self.lambda_margin = lambda_margin

        self.ctc_none = nn.CTCLoss(blank=blank, reduction="none", zero_infinity=True)

        # Register indices of confusing pairs and set of hard characters
        pair_indices = []
        hard_char_indices = set()
        for ca, cb in HARD_CONFUSION_PAIRS:
            if ca in char2idx and cb in char2idx:
                ia, ib = char2idx[ca], char2idx[cb]
                pair_indices.append((ia, ib))
                hard_char_indices.add(ia)
                hard_char_indices.add(ib)

        self.pair_indices = pair_indices
        self.hard_char_indices = hard_char_indices

    def compute_margin_loss(self, head_weight: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Compute angular margin loss on classifier prototype weights W: [Num_Classes, D].
        L_margin = mean_{(a,b)} (max(0, cos(w_a, w_b) - m))^2
        """
        if head_weight is None or len(self.pair_indices) == 0:
            return torch.tensor(0.0)

        w_norm = F.normalize(head_weight.float(), p=2, dim=1)
        idx_a = [p[0] for p in self.pair_indices]
        idx_b = [p[1] for p in self.pair_indices]

        wa = w_norm[idx_a]  # [P, D]
        wb = w_norm[idx_b]  # [P, D]
        cos_sim = (wa * wb).sum(dim=1)  # [P]

        excess = F.relu(cos_sim - self.margin)
        return (excess ** 2).mean()

    def forward(
        self,
        preds_permuted: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        head_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            preds_permuted: Log-probabilities of shape [T, B, C]
            targets: 1D concatenated targets or 2D padded targets
            input_lengths: [B]
            target_lengths: [B]
            head_weight: Classifier linear head weight [C, D] for prototype margin regularization
        """
        device = preds_permuted.device
        preds_fp32 = torch.clamp(preds_permuted.float(), min=-50.0)

        # Compute per-sample CTC loss (handle MPS fallback cleanly)
        if device.type == "mps":
            sample_losses = self.ctc_none(
                preds_fp32.cpu(),
                targets.cpu(),
                input_lengths.cpu(),
                target_lengths.cpu(),
            ).to(device)
        else:
            sample_losses = self.ctc_none(
                preds_fp32,
                targets,
                input_lengths,
                target_lengths,
            )

        # Normalize per-sample loss by target length so it is on the same scale as reduction='mean'
        target_lens_dev = target_lengths.to(device).float().clamp(min=1.0)
        per_char_losses = sample_losses / target_lens_dev  # [B]

        # Compute per-sample hard-character ratio from concatenated 1D targets
        B = target_lengths.size(0)
        hard_weights = torch.ones(B, dtype=torch.float32, device=device)
        if targets.dim() == 1 and len(self.hard_char_indices) > 0:
            targets_list = targets.detach().cpu().tolist()
            lengths_list = target_lengths.detach().cpu().tolist()
            offset = 0
            w_list = []
            for length in lengths_list:
                seq = targets_list[offset : offset + length]
                offset += length
                if length > 0:
                    n_hard = sum(1 for c in seq if c in self.hard_char_indices)
                    w_list.append(1.0 + self.alpha_hard * (n_hard / float(length)))
                else:
                    w_list.append(1.0)
            hard_weights = torch.tensor(w_list, dtype=torch.float32, device=device)

        # Focal modulation for hard sequences (bounded to prevent outlier explosion)
        focal_factor = 1.0 + self.gamma_focal * (1.0 - torch.exp(-per_char_losses.detach().clamp(min=0.0, max=10.0)))
        total_weights = hard_weights * focal_factor
        # Normalize weights so average weight is 1.0 (preserving effective learning rate)
        total_weights = total_weights / total_weights.mean().clamp(min=1e-6)

        loss_ctc = (per_char_losses * total_weights).mean()

        if head_weight is not None and self.lambda_margin > 0:
            loss_margin = self.compute_margin_loss(head_weight).to(device)
            return loss_ctc + self.lambda_margin * loss_margin

        return loss_ctc
