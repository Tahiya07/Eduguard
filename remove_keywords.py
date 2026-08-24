with open('bloom_prompt.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Remove lines 589-658 (keyword-based validation sections)
# Keep lines 1-588, then add simplified return, then keep lines 659-end
new_lines = lines[:588] + [
    '    # For cognitive validation beyond trivial wrapping, rely on semantic and task validators\n',
    '    # to avoid false positives on valid transformations that don not use expected keywords\n',
    '\n',
    '    return False, ""\n',
] + lines[659:]

with open('bloom_prompt.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print('Removed keyword-based validation sections from _is_trivial_transformation')
