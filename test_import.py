"""
Test basic import of bloom_prompt module.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from bloom_prompt import TARGET_TRANSFORMATION_POLICY, build_targeted_rewrite_prompt, _validate_output_format, _semantic_cognitive_check_deterministic
    print("[SUCCESS] All imports successful")
    print()
    print("Transformation policies loaded:")
    for level in TARGET_TRANSFORMATION_POLICY.keys():
        print(f"  - {level}")
    print()
    print("Functions available:")
    print("  - build_targeted_rewrite_prompt")
    print("  - _validate_output_format")
    print("  - _semantic_cognitive_check_deterministic")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    import traceback
    traceback.print_exc()
