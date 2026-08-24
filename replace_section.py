with open('bloom_prompt.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the section to replace
start_idx = None
end_idx = None
for i, line in enumerate(lines):
    if '# Check if rewrite demonstrates understanding operations' in line:
        start_idx = i
    if start_idx is not None and i > start_idx and 'return False, ""' in line:
        end_idx = i + 1
        break

if start_idx and end_idx:
    # Keep lines before start_idx, skip to end_idx, then add new simplified version
    new_lines = lines[:start_idx] + [
        '    # For Apply, Analyze, Evaluate, Create, Remember:\n',
        '    # The semantic and task validators already check for cognitive operations\n',
        '    # We avoid additional keyword checks to prevent false positives\n',
        '    # Only trivial wrapping cases (checked above) are caught by this function\n',
        '\n',
        '    return False, ""\n',
    ]
    
    with open('bloom_prompt.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print(f'Replaced lines {start_idx} to {end_idx} with simplified version')
else:
    print('Could not find section to replace')
