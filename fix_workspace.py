#!/usr/bin/env python3
"""Fix the workspace.tsx file by converting literal \n to actual newlines."""

import re

file_path = r'c:\Users\tahiy\PycharmProjects\Eduguard\frontend\app\workspace.tsx'

# Read the file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace literal backslash-n with actual newlines
content = content.replace('\\n', '\n')

# Now format the file properly
lines = content.split('\n')
formatted_lines = []
indent = 0

for line in lines:
    line = line.rstrip()
    
    if not line:
        if formatted_lines and formatted_lines[-1]:
            formatted_lines.append('')
        continue
    
    # Count braces to determine indentation
    open_count = line.count('{')
    close_count = line.count('}')
    
    # Reduce indent for closing braces at start of line
    if line.strip().startswith('}'):
        indent = max(0, indent - 1)
    
    # Add the line with proper indentation
    formatted_lines.append('  ' * indent + line)
    
    # Adjust indent for next line
    indent += open_count - close_count
    indent = max(0, indent)

# Join and clean up
result = '\n'.join(formatted_lines)

# Remove excessive blank lines
result = re.sub(r'\n\n\n+', '\n\n', result)

# Write back
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(result.strip() + '\n')

print(f"File formatted successfully!")
print(f"Total lines: {len(result.split(chr(10)))}")
