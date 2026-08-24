def _is_trivial_transformation(original: str, rewrite: str, target_level: str) -> tuple[bool, str]:
    """
    Check if the rewrite is a trivial transformation that doesn't meaningfully change cognitive demand.
    
    Rejects:
    - Identical questions after normalization
    - Surface-level changes (punctuation, capitalization, filler words)
    - "Explain" → "How would you explain" wrapping without elaboration
    - Cognitive-demand-preserving transformations
    
    Relies on semantic and task validators for cognitive validation to avoid false positives.
    Only catches obvious trivial wrapping cases here.
    
    Args:
        original: The original question
        rewrite: The generated rewrite
        target_level: The target Bloom level
    
    Returns:
        tuple: (is_trivial, reason)
    """
    import re
    
    # Normalize both questions for comparison
    def normalize(q: str) -> str:
        q = q.lower().strip()
        q = re.sub(r'[^\w\s]', '', q)  # Remove punctuation
        q = re.sub(r'\s+', ' ', q)  # Normalize whitespace
        return q
    
    norm_original = normalize(original)
    norm_rewrite = normalize(rewrite)
    
    # Check for identity
    if norm_original == norm_rewrite:
        return True, "Rewrite is identical to original after normalization"
    
    # Check for trivial "How would you explain" wrapping for Understand
    if target_level == "Understand":
        # Pattern: "explain X" → "how would you explain X"
        if norm_original.startswith("explain ") and norm_rewrite.startswith("how would you explain "):
            original_content = norm_original.replace("explain ", "", 1)
            rewrite_content = norm_rewrite.replace("how would you explain ", "", 1)
            if original_content == rewrite_content:
                return True, "Trivial transformation: 'explain' to 'how would you explain' without elaboration"
        
        # Pattern: "what is X" → "how would you explain what X is"
        if norm_original.startswith("what is ") and norm_rewrite.startswith("how would you explain what"):
            return True, "Trivial transformation: wrapping 'what is' with 'how would you explain'"
        
        # Check for elaboration beyond simple wrapping
        if norm_rewrite.startswith("how would you explain "):
            rewrite_content = norm_rewrite.replace("how would you explain ", "", 1)
            original_content = norm_original.replace("explain ", "", 1).replace("what is ", "", 1)
            
            # If the content is essentially the same, it's trivial
            if rewrite_content == original_content or rewrite_content.startswith(original_content):
                return True, "Understand transformation lacks meaningful elaboration"
    
    # Check for trivial verb wrapping for other levels
    trivial_wrappings = {
        "Apply": [("apply ", "how would you apply ")],
        "Analyze": [("analyze ", "how would you analyze ")],
        "Evaluate": [("evaluate ", "how would you evaluate ")],
        "Create": [("design ", "how would you design ")],
    }
    
    if target_level in trivial_wrappings:
        for original_prefix, rewrite_prefix in trivial_wrappings[target_level]:
            if norm_original.startswith(original_prefix) and norm_rewrite.startswith(rewrite_prefix):
                original_content = norm_original.replace(original_prefix, "", 1)
                rewrite_content = norm_rewrite.replace(rewrite_prefix, "", 1)
                if original_content == rewrite_content:
                    return True, f"Trivial transformation: '{original_prefix}' to '{rewrite_prefix}' without elaboration"
    
    # For other cognitive validation, rely on semantic and task validators
    # to avoid false positives on valid transformations that don't use expected keywords
    
    return False, ""
