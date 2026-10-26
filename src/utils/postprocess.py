"""Post-processing utilities for OCR decoding (Pure Greedy CTC by default, with optional uniform domain syntax constraint)."""
from itertools import groupby
from typing import Dict, List, Tuple

import numpy as np
import torch

# Deterministic Mercosur position-4 valid letter set (digits 0-9 converted to A-J on Mercosur plates)
VALID_MERCOSUR_POS4_LETTERS = frozenset("ABCDEFGHIJ")


def extract_7slot_ctc_tensor(
    preds: torch.Tensor,
    anchor_preds: torch.Tensor = None,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Projects a [B, T, C] CTC log-probability tensor (for any sequence length T, e.g. T=16 or T=33)
    into a phase-invariant, non-blank normalized 7-slot tensor [B, 7, C] and a validity mask [B].
    """
    lp_np = (preds.detach().float().cpu().numpy() / max(1e-3, temperature)).astype(np.float32)
    g_np = np.argmax(lp_np, axis=2)
    B, T_n, C = lp_np.shape

    anchor_segs_list = None
    T_anc = T_n
    if anchor_preds is not None:
        anc_np = anchor_preds.detach().float().cpu().numpy()
        anc_g = np.argmax(anc_np, axis=2)
        T_anc = anc_np.shape[1]
        anchor_segs_list = []
        for b in range(B):
            s_b = []
            t_i = 0
            for c_idx, grp in groupby(anc_g[b]):
                gsz = len(list(grp))
                if c_idx != 0:
                    s_b.append((t_i, t_i + gsz))
                t_i += gsz
            anchor_segs_list.append(s_b)

    out = np.zeros((B, 7, C), dtype=np.float32)
    valid = np.zeros(B, dtype=bool)

    for b in range(B):
        segs = []
        t_i = 0
        for c_idx, grp in groupby(g_np[b]):
            gsz = len(list(grp))
            if c_idx != 0:
                segs.append((t_i, t_i + gsz))
            t_i += gsz

        if len(segs) == 7:
            valid[b] = True
            for k, (ts, te) in enumerate(segs):
                out[b, k, :] = np.max(lp_np[b, ts:te, :], axis=0)
        elif anchor_segs_list is not None and len(anchor_segs_list[b]) == 7:
            valid[b] = True
            for k, (ts, te) in enumerate(anchor_segs_list[b]):
                ts_n = max(0, int(round(ts * T_n / float(T_anc))))
                te_n = min(T_n, max(ts_n + 1, int(round(te * T_n / float(T_anc)))))
                out[b, k, :] = np.max(lp_np[b, ts_n:te_n, :], axis=0)
        else:
            # Uniformly partition the sequence into 7 intervals as a safe fallback instead of zero-filling
            edges = np.linspace(0, T_n, 8, dtype=int)
            for k in range(7):
                ts_n = int(edges[k])
                te_n = max(ts_n + 1, int(edges[k + 1]))
                out[b, k, :] = np.max(lp_np[b, ts_n:te_n, :], axis=0)

    out_t = torch.from_numpy(out)
    char_lp = torch.nn.functional.log_softmax(out_t[:, :, 1:], dim=-1)
    full_lp = torch.full((B, 7, C), -30.0, dtype=torch.float32)
    full_lp[:, :, 1:] = char_lp
    return full_lp, torch.from_numpy(valid)


def slots_to_canonical_ctc(slot_lp: torch.Tensor) -> torch.Tensor:
    """Interleaves a [B, 7, C] slot tensor with CTC blanks to form a canonical [B, 13, C] CTC tensor."""
    B, _, C = slot_lp.shape
    blank_step = torch.full((B, 1, C), -30.0, dtype=slot_lp.dtype, device=slot_lp.device)
    blank_step[:, 0, 0] = 0.0
    steps = []
    for k in range(7):
        steps.append(slot_lp[:, k : k + 1, :])
        if k < 6:
            steps.append(blank_step)
    return torch.cat(steps, dim=1)


def decode_with_confidence(
    preds: torch.Tensor,
    idx2char: Dict[int, str],
    use_format_prior: bool = False,
) -> List[Tuple[str, float]]:
    """CTC decode predictions with confidence scores.

    By default (`use_format_prior=False`), performs Pure Greedy CTC Decoding identical to the
    original baseline (no positional syntax constraints, no geographic/frequency priors).
    If `use_format_prior=True` is explicitly requested (e.g. for an Ablation Study on Domain Syntax),
    restricts each of the 7 character slots to its syntactically valid character class (Letters A-Z
    at Pos 0,1,2; Digits 0-9 at Pos 3,5,6; Digits 0-9 or Mercosur A-J at Pos 4) under a strictly
    UNIFORM prior over valid classes.
    """
    log_probs_np = preds.detach().float().cpu().numpy()
    greedy_indices_np = np.argmax(log_probs_np, axis=2)
    max_log_probs_np = np.max(log_probs_np, axis=2)

    letter_indices = [idx for idx, c in idx2char.items() if c.isalpha()]
    digit_indices = [idx for idx, c in idx2char.items() if c.isdigit()]
    pos4_indices = digit_indices + [
        idx for idx in letter_indices if idx2char[idx] in VALID_MERCOSUR_POS4_LETTERS
    ]

    batch_size = log_probs_np.shape[0]
    results: List[Tuple[str, float]] = []

    for batch_idx in range(batch_size):
        path = greedy_indices_np[batch_idx]
        sample_log_probs = log_probs_np[batch_idx]  # [T, C]
        sample_max_lp = max_log_probs_np[batch_idx]  # [T]

        # Collect non-blank CTC segments: (char_idx, start_t, end_t, max_log_prob)
        segments: List[Tuple[int, int, int, float]] = []
        time_idx = 0
        for char_idx, group in groupby(path):
            group_size = len(list(group))
            if char_idx != 0:
                seg_lp = float(np.max(sample_max_lp[time_idx:time_idx + group_size]))
                segments.append((int(char_idx), time_idx, time_idx + group_size, seg_lp))
            time_idx += group_size

        pred_chars: List[str] = []
        confidences: List[float] = []

        apply_grammar = use_format_prior and (len(segments) == 7) and bool(letter_indices) and bool(digit_indices)

        for pos_idx, (char_idx, t_start, t_end, seg_lp) in enumerate(segments):
            raw_char = idx2char.get(char_idx, '')
            chosen_char = raw_char
            chosen_lp = seg_lp

            if apply_grammar:
                seg_max_lp = np.max(sample_log_probs[t_start:t_end, :], axis=0)

                # Positions 0, 1, 2: Uniform selection among Letters (A-Z) if greedy picked a Digit
                if pos_idx in (0, 1, 2) and not raw_char.isalpha():
                    best_letter_idx = max(letter_indices, key=lambda idx: float(seg_max_lp[idx]))
                    chosen_char = idx2char.get(best_letter_idx, raw_char)
                    chosen_lp = float(seg_max_lp[best_letter_idx])

                # Positions 3, 5, 6: Uniform selection among Digits (0-9) if greedy picked a Letter
                elif pos_idx in (3, 5, 6) and not raw_char.isdigit():
                    best_digit_idx = max(digit_indices, key=lambda idx: float(seg_max_lp[idx]))
                    chosen_char = idx2char.get(best_digit_idx, raw_char)
                    chosen_lp = float(seg_max_lp[best_digit_idx])

                # Position 4: Uniform selection among Mercosur valid set (0-9, A-J) if greedy picked K-Z
                elif pos_idx == 4 and raw_char.isalpha() and raw_char not in VALID_MERCOSUR_POS4_LETTERS:
                    best_p4_idx = max(pos4_indices, key=lambda idx: float(seg_max_lp[idx]))
                    chosen_char = idx2char.get(best_p4_idx, raw_char)
                    chosen_lp = float(seg_max_lp[best_p4_idx])

            pred_chars.append(chosen_char)
            confidences.append(float(np.clip(np.exp(chosen_lp), 0.0, 1.0)))

        pred_str = "".join(pred_chars)
        confidence = float(np.mean(confidences)) if confidences else 0.0
        results.append((pred_str, confidence))

    return results


class CTCDecoder:
    """Convenience CTC decoder wrapper around decode_with_confidence."""
    def __init__(self, idx2char: Dict[int, str], blank_idx: int = 0):
        self.idx2char = idx2char
        self.blank_idx = blank_idx

    def decode_batch(self, preds: torch.Tensor, use_format_prior: bool = False) -> List[str]:
        results = decode_with_confidence(preds, self.idx2char, use_format_prior=use_format_prior)
        return [text for text, _ in results]
