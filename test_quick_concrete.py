"""
Quick test of the concrete cases with the new prompt.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bloom_prompt import rewrite_to_target_level

# Test the three concrete cases
test_cases = [
    ("Explain what virtual memory is.", "Understand"),
    ("Explain what virtual memory is.", "Apply"),
    ("Explain what virtual memory is.", "Analyze"),
]

for original, target in test_cases:
    print(f"\nOriginal: {original}")
    print(f"Target: {target}")
    
    rewrite, needs_review, error = rewrite_to_target_level(original, target)
    
    if rewrite:
        print(f"Rewrite: {rewrite}")
        print(f"Needs Review: {needs_review}")
        if error:
            print(f"Review Reason: {error}")
    else:
        print(f"Failed: {error}")
