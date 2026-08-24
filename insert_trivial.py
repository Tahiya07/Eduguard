with open('trivial_function.py', 'r', encoding='utf-8') as f:
    new_function = f.read()

with open('bloom_prompt.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Insert after build_corrective_prompt (before _clean_rewrite)
insert_marker = 'def _clean_rewrite(text: str) -> str:'
if insert_marker in content:
    content = content.replace(insert_marker, new_function + '\n\n' + insert_marker)
    with open('bloom_prompt.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Successfully inserted _is_trivial_transformation')
else:
    print('Could not find insertion point')
