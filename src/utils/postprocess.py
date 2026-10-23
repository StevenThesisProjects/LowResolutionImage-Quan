"""Post-processing utilities for OCR decoding with FP32 Grammar-Constrained Logit Calibration."""
from itertools import groupby
from typing import Dict, List, Set, Tuple

import numpy as np
import torch

# Visual homoglyph priors for Brazilian / Mercosur license plates
DIGIT_TO_LETTER_HOMOGLYPHS: Dict[str, List[str]] = {
    '0': ['D', 'O', 'Q'],
    '1': ['I', 'L', 'J'],
    '2': ['Z'],
    '4': ['A'],
    '5': ['S'],
    '6': ['G', 'C'],
    '7': ['T', 'Z'],
    '8': ['B'],
}

LETTER_TO_DIGIT_HOMOGLYPHS: Dict[str, str] = {
    'D': '0',
    'O': '0',
    'Q': '0',
    'C': '0',
    'I': '1',
    'L': '1',
    'J': '1',
    'Z': '2',
    'A': '4',
    'S': '5',
    'G': '6',
    'T': '7',
    'B': '8',
}

VALID_MERCOSUR_POS4_LETTERS = frozenset("ABCDEFGHIJ")

# Empirical log P(letter | pos) for Positions 0 and 1 computed strictly on 9,001 Scenario-B TRAINING tracks (Slide 22)
POS0_LETTER_LOG_PRIOR: Dict[str, float] = {
    'A': -1.54, 'B': -1.47, 'C': -5.37, 'D': -5.12, 'E': -3.96, 'F': -3.52,
    'G': -4.09, 'H': -5.55, 'I': -4.05, 'J': -5.14, 'K': -5.61, 'L': -5.35,
    'M': -4.03, 'N': -5.42, 'O': -4.85, 'P': -3.66, 'Q': -2.69, 'R': -2.27,
    'S': -2.28, 'T': -2.23, 'U': -5.50, 'V': -9.11, 'W': -9.11, 'X': -9.11,
    'Y': -9.11, 'Z': -9.11,
}

POS1_LETTER_LOG_PRIOR: Dict[str, float] = {
    'A': -2.46, 'B': -2.30, 'C': -2.70, 'D': -2.52, 'E': -2.30, 'F': -3.24,
    'G': -4.70, 'H': -2.78, 'I': -3.87, 'J': -3.98, 'K': -4.20, 'L': -4.03,
    'M': -4.02, 'N': -4.00, 'O': -4.01, 'P': -4.02, 'Q': -3.94, 'R': -4.05,
    'S': -4.09, 'T': -3.49, 'U': -3.43, 'V': -3.44, 'W': -3.50, 'X': -3.13,
    'Y': -3.08, 'Z': -3.18,
}


def _beam_search_7char_fallback(
    sample_log_probs: np.ndarray,
    idx2char: Dict[int, str],
    letter_indices_set: set,
    digit_indices_set: set,
    beam_width: int = 5,
) -> Tuple[str, float]:
    """Prefix Beam Search fallback to recover valid 7-character plates when greedy length != 7."""
    T, _ = sample_log_probs.shape
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, -np.inf)}

    for t in range(T):
        next_beams: Dict[Tuple[int, ...], Tuple[float, float]] = {}
        top_c = np.argsort(sample_log_probs[t])[-5:][::-1]
        lp_blank = float(sample_log_probs[t, 0])

        for prefix, (p_b, p_nb) in beams.items():
            p_total = float(np.logaddexp(p_b, p_nb))
            nb_b, nb_nb = next_beams.get(prefix, (-np.inf, -np.inf))
            next_beams[prefix] = (float(np.logaddexp(nb_b, p_total + lp_blank)), nb_nb)

            for c in top_c:
                c_int = int(c)
                if c_int == 0:
                    continue
                lp_c = float(sample_log_probs[t, c_int])
                pos = len(prefix)
                grammar_bonus = 0.0
                if pos in (0, 1, 2) and c_int not in letter_indices_set:
                    grammar_bonus = -1.5
                elif pos in (3, 5, 6) and c_int not in digit_indices_set:
                    grammar_bonus = -1.5
                elif pos == 4 and c_int in letter_indices_set and idx2char.get(c_int) not in VALID_MERCOSUR_POS4_LETTERS:
                    grammar_bonus = -2.0

                if pos > 0 and prefix[-1] == c_int:
                    ob_b, ob_nb = next_beams.get(prefix, (-np.inf, -np.inf))
                    next_beams[prefix] = (ob_b, float(np.logaddexp(ob_nb, p_nb + lp_c)))
                    if pos < 7:
                        new_p = prefix + (c_int,)
                        nb2_b, nb2_nb = next_beams.get(new_p, (-np.inf, -np.inf))
                        next_beams[new_p] = (nb2_b, float(np.logaddexp(nb2_nb, p_b + lp_c + grammar_bonus)))
                elif pos < 7:
                    new_p = prefix + (c_int,)
                    nb2_b, nb2_nb = next_beams.get(new_p, (-np.inf, -np.inf))
                    next_beams[new_p] = (nb2_b, float(np.logaddexp(nb2_nb, p_total + lp_c + grammar_bonus)))

        sorted_beams = sorted(
            next_beams.items(),
            key=lambda kv: float(np.logaddexp(kv[1][0], kv[1][1])),
            reverse=True,
        )[:beam_width * 2]
        beams = dict(sorted_beams)

    best_7 = None
    best_7_score = -np.inf
    for prefix, (p_b, p_nb) in beams.items():
        score = float(np.logaddexp(p_b, p_nb))
        if len(prefix) == 7 and score > best_7_score:
            best_7 = prefix
            best_7_score = score

    if best_7 is None:
        return "", 0.0

    chars: List[str] = []
    for pos_idx, c_idx in enumerate(best_7):
        ch = idx2char.get(c_idx, "")
        if pos_idx in (0, 1, 2) and not ch.isalpha():
            h_list = DIGIT_TO_LETTER_HOMOGLYPHS.get(ch, [])
            if h_list:
                ch = h_list[0]
        elif pos_idx in (3, 5, 6) and not ch.isdigit():
            h_d = LETTER_TO_DIGIT_HOMOGLYPHS.get(ch)
            if h_d:
                ch = h_d
        elif pos_idx == 4 and ch.isalpha() and ch not in VALID_MERCOSUR_POS4_LETTERS:
            h_d = LETTER_TO_DIGIT_HOMOGLYPHS.get(ch)
            if h_d:
                ch = h_d
        chars.append(ch)

    return "".join(chars), float(np.clip(np.exp(best_7_score / max(1, T)), 0.0, 1.0))


def decode_with_confidence(
    preds: torch.Tensor,
    idx2char: Dict[int, str],
    use_format_prior: bool = True,
) -> List[Tuple[str, float]]:
    """CTC decode predictions with confidence scores and optional 7-char plate grammar calibration.

    Performs class selection in FP32 log-probability space before exponentiation to prevent
    FP16 underflow when calibrating confusable Letter/Digit positions.

    Args:
        preds: Log-softmax predictions of shape [batch_size, time_steps, num_classes].
        idx2char: Index to character mapping (1-indexed, 0 is CTC blank).
        use_format_prior: If True, calibrates 7-character predictions against the deterministic
            Brazilian/Mercosur positional grammar (Pos 0,1,2 = Letters; Pos 3,5,6 = Digits; Pos 4 = Any).

    Returns:
        List of (predicted_string, confidence_score) tuples.
    """
    # Convert to FP32 log-probabilities first to prevent FP16 exp() underflow
    log_probs_np = preds.detach().float().cpu().numpy()
    greedy_indices_np = np.argmax(log_probs_np, axis=2)
    max_log_probs_np = np.max(log_probs_np, axis=2)

    char2idx = {c: idx for idx, c in idx2char.items()}
    letter_indices = [idx for idx, c in idx2char.items() if c.isalpha()]
    digit_indices = [idx for idx, c in idx2char.items() if c.isdigit()]

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

        if use_format_prior and len(segments) != 7 and bool(letter_indices) and bool(digit_indices):
            fb_str, fb_conf = _beam_search_7char_fallback(
                sample_log_probs=sample_log_probs,
                idx2char=idx2char,
                letter_indices_set=set(letter_indices),
                digit_indices_set=set(digit_indices),
                beam_width=5,
            )
            if len(fb_str) == 7:
                results.append((fb_str, fb_conf))
                continue

        for pos_idx, (char_idx, t_start, t_end, seg_lp) in enumerate(segments):
            raw_char = idx2char.get(char_idx, '')
            chosen_char = raw_char
            chosen_lp = seg_lp

            if apply_grammar:
                # Positions 0, 1, 2 must be Letters (A-Z) with Slide-22 Bayesian log-priors
                if pos_idx in (0, 1, 2):
                    prior_map = POS0_LETTER_LOG_PRIOR if pos_idx == 0 else (POS1_LETTER_LOG_PRIOR if pos_idx == 1 else {})
                    alpha_p = 0.25 if pos_idx == 0 else (0.15 if pos_idx == 1 else 0.0)
                    seg_max_lp = np.max(sample_log_probs[t_start:t_end, :], axis=0)
                    best_letter_idx = char_idx if raw_char.isalpha() else letter_indices[0]
                    best_score = -1e9
                    for l_idx in letter_indices:
                        l_char = idx2char[l_idx]
                        score = float(seg_max_lp[l_idx]) + alpha_p * prior_map.get(l_char, -3.25)
                        if score > best_score:
                            best_score = score
                            best_letter_idx = l_idx
                    best_letter_lp = float(seg_max_lp[best_letter_idx])

                    if not raw_char.isalpha():
                        homoglyphs = DIGIT_TO_LETTER_HOMOGLYPHS.get(raw_char, [])
                        for h_char in homoglyphs:
                            h_idx = char2idx.get(h_char)
                            if h_idx is not None:
                                h_lp = float(seg_max_lp[h_idx])
                                if h_lp + 1.5 > best_letter_lp:
                                    best_letter_idx = h_idx
                                    best_letter_lp = h_lp

                    chosen_char = idx2char.get(best_letter_idx, raw_char)
                    chosen_lp = best_letter_lp

                # Positions 3, 5, 6 must be Digits (0-9)
                elif pos_idx in (3, 5, 6) and not raw_char.isdigit():
                    seg_matrix = sample_log_probs[t_start:t_end, :][:, digit_indices]
                    best_local_idx = int(np.argmax( np.max(seg_matrix, axis=0) ))
                    best_digit_idx = digit_indices[best_local_idx]
                    best_digit_lp = float(np.max(sample_log_probs[t_start:t_end, best_digit_idx]))

                    h_digit = LETTER_TO_DIGIT_HOMOGLYPHS.get(raw_char)
                    if h_digit is not None:
                        h_idx = char2idx.get(h_digit)
                        if h_idx is not None:
                            h_lp = float(np.max(sample_log_probs[t_start:t_end, h_idx]))
                            if h_lp + 1.5 > best_digit_lp:
                                best_digit_idx = h_idx
                                best_digit_lp = h_lp

                    chosen_char = idx2char.get(best_digit_idx, raw_char)
                    chosen_lp = best_digit_lp

                # Position 4: Mercosur hard constraint (Digits 0-9 or Letters A-J only, K-Z strictly forbidden)
                elif pos_idx == 4 and raw_char.isalpha() and raw_char not in VALID_MERCOSUR_POS4_LETTERS:
                    seg_max_lp = np.max(sample_log_probs[t_start:t_end, :], axis=0)
                    valid_pos4_indices = digit_indices + [
                        idx for idx in letter_indices if idx2char[idx] in VALID_MERCOSUR_POS4_LETTERS
                    ]
                    best_p4_idx = max(valid_pos4_indices, key=lambda idx: float(seg_max_lp[idx]))
                    best_p4_lp = float(seg_max_lp[best_p4_idx])

                    h_digit = LETTER_TO_DIGIT_HOMOGLYPHS.get(raw_char)
                    if h_digit is not None:
                        h_idx = char2idx.get(h_digit)
                        if h_idx is not None:
                            h_lp = float(seg_max_lp[h_idx])
                            if h_lp + 1.5 > best_p4_lp:
                                best_p4_idx = h_idx
                                best_p4_lp = h_lp

                    chosen_char = idx2char.get(best_p4_idx, raw_char)
                    chosen_lp = best_p4_lp

            pred_chars.append(chosen_char)
            confidences.append(float(np.clip(np.exp(chosen_lp), 0.0, 1.0)))

        pred_str = "".join(pred_chars)
        if confidences:
            confidence = float(0.5 * np.mean(confidences) + 0.5 * np.min(confidences))
        else:
            confidence = 0.0
        results.append((pred_str, confidence))

    return results


class CTCDecoder:
    """Convenience CTC decoder wrapper around decode_with_confidence."""
    def __init__(self, idx2char: Dict[int, str], blank_idx: int = 0):
        self.idx2char = idx2char
        self.blank_idx = blank_idx

    def decode_batch(self, preds: torch.Tensor, use_format_prior: bool = True) -> List[str]:
        results = decode_with_confidence(preds, self.idx2char, use_format_prior=use_format_prior)
        return [text for text, _ in results]
