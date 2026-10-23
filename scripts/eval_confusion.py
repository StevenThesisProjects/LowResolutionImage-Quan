import numpy as np
from collections import defaultdict
import json
import argparse
import os

def get_alignment(ref, hyp):
    """
    Computes Levenshtein alignment between reference and hypothesis strings.
    Returns a list of operations: 'C' (Correct), 'S' (Substitution), 'I' (Insertion), 'D' (Deletion),
    and the corresponding characters.
    """
    n = len(ref)
    m = len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,      # Deletion
                           dp[i][j-1] + 1,      # Insertion
                           dp[i-1][j-1] + cost) # Substitution / Correct
                           
    # Backtrack
    i, j = n, m
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if ref[i-1] == hyp[j-1] else 1):
            if ref[i-1] == hyp[j-1]:
                ops.append(('C', ref[i-1], hyp[j-1]))
            else:
                ops.append(('S', ref[i-1], hyp[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(('D', ref[i-1], ''))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ops.append(('I', '', hyp[j-1]))
            j -= 1
    
    ops.reverse()
    return ops

def evaluate_confusion(predictions, ground_truths):
    """
    Evaluates predictions against ground truths, outputs CER and Confusion stats.
    """
    total_edits = 0
    total_chars = 0
    exact_matches = 0
    
    confusions = defaultdict(int)
    
    for ref, hyp in zip(ground_truths, predictions):
        if ref == hyp:
            exact_matches += 1
            
        ops = get_alignment(ref, hyp)
        for op, r, h in ops:
            if op == 'S':
                confusions[(r, h)] += 1
                total_edits += 1
            elif op == 'I':
                total_edits += 1
            elif op == 'D':
                total_edits += 1
        
        total_chars += len(ref)
        
    cer = total_edits / total_chars if total_chars > 0 else 0
    acc = exact_matches / len(ground_truths) if len(ground_truths) > 0 else 0
    
    return acc, cer, confusions

def main():
    # For testing, we use dummy data
    ground_truths = ["29A12345", "30B8888", "51D55555", "79S22222"]
    predictions   = ["29A1234S", "3088888", "51055555", "79522222"]
    
    acc, cer, confusions = evaluate_confusion(predictions, ground_truths)
    
    print("=== Task 4.2: Character Confusion Analysis ===")
    print(f"Exact Match Accuracy: {acc*100:.2f}%")
    print(f"Character Error Rate (CER): {cer*100:.2f}%")
    
    print("\n--- Top Confusions (Substitution) ---")
    sorted_conf = sorted(confusions.items(), key=lambda x: x[1], reverse=True)
    
    target_pairs = [('8', 'B'), ('B', '8'), ('0', 'D'), ('D', '0'), ('5', 'S'), ('S', '5'), ('2', '7'), ('7', '2')]
    
    for (r, h), count in sorted_conf:
        marker = "🔥" if (r, h) in target_pairs else "  "
        print(f"{marker} {r} -> {h} : {count} times")

if __name__ == "__main__":
    main()
