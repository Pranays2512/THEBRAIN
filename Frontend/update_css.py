with open('src/components/Layout/Background.css', 'r') as f:
    css = f.read()

# remove old stagger rules
start_idx = css.find('/* Stagger drawing one by one')
if start_idx != -1:
    css = css[:start_idx]

stagger_rules = "/* Stagger drawing one by one for organic feel (up to 120 elements) */\n"
for i in range(1, 121):
    delay = (i-1) * 0.1 # 0.1s per line
    stagger_rules += f".doodle-group g > *:nth-child({i}) {{ animation-delay: {delay:.1f}s; }}\n"

with open('src/components/Layout/Background.css', 'w') as f:
    f.write(css + stagger_rules)
