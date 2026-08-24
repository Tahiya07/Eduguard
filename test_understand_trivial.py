"""
Test the specific Understand trivial transformation case.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bloom_prompt import _is_trivial_transformation

# Test the specific case from requirements
original = "Explain what virtual memory is."
trivial_rewrite = "How would you explain what virtual memory is?"
meaningful_rewrite = "How would you explain the purpose and function of virtual memory in managing system resources?"

print("Testing Understand trivial transformation rejection")
print(f"Original: {original}")
print()

print("Test 1: Trivial rewrite")
print(f"Rewrite: {trivial_rewrite}")
is_trivial, reason = _is_trivial_transformation(original, trivial_rewrite, "Understand")
print(f"Is Trivial: {is_trivial}")
print(f"Reason: {reason}")
assert is_trivial, "Should reject trivial 'how would you explain' wrapping"
print("[PASS]")
print()

print("Test 2: Meaningful rewrite")
print(f"Rewrite: {meaningful_rewrite}")
is_trivial, reason = _is_trivial_transformation(original, meaningful_rewrite, "Understand")
print(f"Is Trivial: {is_trivial}")
print(f"Reason: {reason}")
assert not is_trivial, "Should accept meaningful Understand transformation"
print("[PASS]")
print()

print("All Understand tests passed!")
