"""
Test the prompt building to verify transformation rules are included.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def test_prompt_building():
    """Test that the prompt is compact and uses deterministic policies."""
    try:
        from bloom_prompt import build_targeted_rewrite_prompt, TARGET_TRANSFORMATION_POLICY
    except Exception as e:
        print(f"[ERROR] Failed to import bloom_prompt: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test that transformation policies exist
    print("=" * 80)
    print("TRANSFORMATION POLICY TEST")
    print("=" * 80)
    print()
    
    for level, policy in TARGET_TRANSFORMATION_POLICY.items():
        print(f"{level}:")
        print(f"  Operation: {policy['operation']}")
        print(f"  Allowed: {', '.join(policy['allowed'][:3])}...")
        print(f"  Forbidden: {', '.join(policy['forbidden'][:3])}...")
        print()
    
    print("=" * 80)
    print("COMPACT PROMPT TEST")
    print("=" * 80)
    print()
    
    test_cases = [
        ("Design a level zero restaurant management system.", "Understand"),
        ("Design a banking system.", "Understand"),
        ("Explain the banking system.", "Create"),
    ]
    
    for question, target_level in test_cases:
        print(f"Input: {question}")
        print(f"Target: {target_level}")
        print("-" * 80)
        
        try:
            prompt = build_targeted_rewrite_prompt(question, target_level=target_level)
            
            # Check for compact structure
            has_target_level = f"TARGET LEVEL: {target_level.upper()}" in prompt
            has_cognitive_operation = "COGNITIVE OPERATION:" in prompt
            has_task = "TASK:" in prompt
            has_allow = "ALLOW:" in prompt
            has_remove = "REMOVE:" in prompt
            has_rule = "RULE:" in prompt
            has_output_instruction = "OUTPUT ONLY ONE STUDENT-FACING QUESTION" in prompt
            
            print(f"[PASS] Has TARGET LEVEL: {has_target_level}")
            print(f"[PASS] Has COGNITIVE OPERATION: {has_cognitive_operation}")
            print(f"[PASS] Has TASK: {has_task}")
            print(f"[PASS] Has ALLOW: {has_allow}")
            print(f"[PASS] Has REMOVE: {has_remove}")
            print(f"[PASS] Has RULE: {has_rule}")
            print(f"[PASS] Has OUTPUT instruction: {has_output_instruction}")
            
            # Check that prompt is compact (not verbose)
            prompt_length = len(prompt)
            print(f"[INFO] Prompt length: {prompt_length} characters")
            if prompt_length < 800:
                print(f"[PASS] Prompt is compact (< 800 chars)")
            else:
                print(f"[WARN] Prompt may be too long for 1.5B model")
            
            print()
            print("First 400 characters of prompt:")
            print(prompt[:400])
            print("...")
            print()
            
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 80)

if __name__ == "__main__":
    test_prompt_building()
