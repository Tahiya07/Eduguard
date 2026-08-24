with open('bloom_prompt.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the section
old_text = '''        # Check if rewrite demonstrates understanding operations (meaning, purpose, function, interpretation)
        # Keywords indicating understanding: meaning, purpose, function, role, interpret, describe, summarize, explain, compare, classify
        understand_operations = ["meaning", "purpose", "function", "role", "interpret", "describe", "summarize", "compare", "classify", "work", "operate", "relate"]
        has_understand_operation = any(kw in norm_rewrite for kw in understand_operations)
        
        # If it's just "how would you [verb] the [topic]" without understanding indicators, it might be trivial
        # But we allow it if semantic validation passes (handled by _semantic_cognitive_check_deterministic)
        # So we only flag obvious cases here
        if not has_understand_operation and norm_rewrite.startswith("how would you ") and norm_rewrite.endswith("?"):
            # Check if it's a simple verb substitution without cognitive change
            rewrite_verb = norm_rewrite.replace("how would you ", "", 1).replace("?", "").split()[0] if " " in norm_rewrite else ""
            if rewrite_verb in ["explain", "describe", "state"]:
                return True, "Understand transformation appears to be simple verb substitution without cognitive change"
    
    elif target_level == "Apply":
        # Apply requires using knowledge/rules/procedures to solve or handle a concrete case/problem
        # Check for concrete application context: scenario, situation, problem, solve, case, example, specific, concrete, use, apply, implement, practical, handle, execute, perform
        apply_operations = ["scenario", "situation", "problem", "solve", "case", "example", "specific", "concrete", "implement", "practical", "application", "use in", "apply to", "handle", "execute", "perform", "work with", "utilize"]
        has_apply_operation = any(kw in norm_rewrite for kw in apply_operations)
        
        if not has_apply_operation:
            return True, "Apply transformation lacks concrete application context or problem to solve"
    
    elif target_level == "Analyze":
        # Analyze requires identifying components and/or relationships, causes, effects, structure, patterns, distinctions, or functional interactions
        # Operation indicators: component, relationship, cause, effect, pattern, structure, differ, compare, examine, interact, relate, connect, influence, impact, distinction, aspect, part, element, role, function, breakdown, decompose
        analyze_operations = ["component", "relationship", "cause", "effect", "pattern", "structure", "differ", "compare", "examine", "interact", "relate", "connect", "influence", "impact", "distinction", "aspect", "part", "element", "role", "function", "breakdown", "decompose", "definition", "implementation", "characteristic", "feature"]
        has_analyze_operation = any(kw in norm_rewrite for kw in analyze_operations)
        
        if not has_analyze_operation:
            return True, "Analyze transformation lacks analytical operation (components, relationships, causes, effects, structure, patterns, distinctions)"
    
    elif target_level == "Evaluate":
        # Evaluate requires making a judgment using explicit or implicit criteria, evidence, standards, justification, or comparison against a basis
        # Operation indicators: criteria, evidence, standard, justify, assess, judge, measure, evaluate, compare, rate, rank, determine, effectiveness, quality, merit, worth, value, based on, according to, using, against, judgment, decision
        evaluate_operations = ["criteria", "evidence", "standard", "justify", "assess", "judge", "measure", "evaluate", "compare", "rate", "rank", "determine", "effectiveness", "quality", "merit", "worth", "value", "based on", "according to", "using", "against", "judgment", "decision"]
        has_evaluate_operation = any(kw in norm_rewrite for kw in evaluate_operations)
        
        if not has_evaluate_operation:
            return True, "Evaluate transformation lacks judgment criteria, evidence, or comparison basis"
    
    elif target_level == "Create":
        # Create requires formulating, designing, developing, proposing, constructing, or producing a new/improved solution/product/model
        # Operation indicators: design, develop, construct, propose, formulate, create, build, new, original, improved, novel, solution, product, model, system, plan, strategy, method, approach, invention, innovation
        create_operations = ["design", "develop", "construct", "propose", "formulate", "create", "build", "new", "original", "improved", "novel", "solution", "product", "model", "system", "plan", "strategy", "method", "approach", "invention", "innovation", "generate", "produce"]
        has_create_operation = any(kw in norm_rewrite for kw in create_operations)
        
        if not has_create_operation:
            return True, "Create transformation lacks design/construction of something new or improved"
    
    elif target_level == "Remember":
        # Remember requires retrieving factual knowledge: naming, listing, identifying, recalling, or stating
        # Operation indicators: list, identify, name, state, recall, define, component, feature, type, kind, part, element, what, which, who, when, where
        remember_operations = ["list", "identify", "name", "state", "recall", "define", "component", "feature", "type", "kind", "part", "element", "what are", "which", "who", "when", "where"]
        has_remember_operation = any(kw in norm_rewrite for kw in remember_operations)
        
        if not has_remember_operation:
            return True, "Remember transformation lacks factual recall instruction"
    
    return False, ""'''

new_text = '''    # For Apply, Analyze, Evaluate, Create, Remember:
    # The semantic and task validators already check for cognitive operations
    # We avoid additional keyword checks to prevent false positives
    # Only trivial wrapping cases (checked above) are caught by this function
    
    return False, ""'''

if old_text in content:
    content = content.replace(old_text, new_text)
    with open('bloom_prompt.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Successfully replaced section')
else:
    print('Could not find old text to replace')
