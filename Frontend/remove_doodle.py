with open('src/components/Layout/Doodles.jsx', 'r') as f:
    content = f.read()

# Remove wineScribbleDoodle from ALL_DOODLES array
content = content.replace(',\n    wineScribbleDoodle', '')

# Remove the actual component definition
start = content.find('export const wineScribbleDoodle')
if start != -1:
    end = content.find('export const ALL_DOODLES')
    if end != -1:
        content = content[:start] + content[end:]

with open('src/components/Layout/Doodles.jsx', 'w') as f:
    f.write(content)
