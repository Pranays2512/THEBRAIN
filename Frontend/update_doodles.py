with open('ScribbleOutput.jsx') as f:
    doodle_code = f.read()

with open('src/components/Layout/Doodles.jsx', 'r') as f:
    original = f.read()

# insert doodle_code before ALL_DOODLES
insert_pos = original.find('export const ALL_DOODLES')
new_content = original[:insert_pos] + doodle_code + "\n" + original[insert_pos:]

# add wineScribbleDoodle to ALL_DOODLES array
new_content = new_content.replace('archDoodle\n];', 'archDoodle,\n    wineScribbleDoodle\n];')

with open('src/components/Layout/Doodles.jsx', 'w') as f:
    f.write(new_content)
