import random
import math

def generate_scribble_path(x_min, x_max, y_min, y_max, num_points, vertical=False, chaotic=10):
    points = []
    # Start at a random edge
    if vertical:
        x = random.uniform(x_min, x_max)
        y = y_min
    else:
        x = x_min
        y = random.uniform(y_min, y_max)
        
    points.append(f"M {x:.1f} {y:.1f}")
    
    for _ in range(num_points):
        if vertical:
            y = y_max if y < (y_min + y_max)/2 else y_min
            y += random.uniform(-chaotic, chaotic)
            x += random.uniform(-chaotic*2, chaotic*2)
        else:
            x = x_max if x < (x_min + x_max)/2 else x_min
            x += random.uniform(-chaotic, chaotic)
            y += random.uniform(-chaotic*2, chaotic*2)
            
        x = max(x_min, min(x_max, x))
        y = max(y_min, min(y_max, y))
        
        # Sometimes use a curve instead of a line
        if random.random() < 0.3:
            cx = x + random.uniform(-20, 20)
            cy = y + random.uniform(-20, 20)
            points.append(f"Q {cx:.1f} {cy:.1f}, {x:.1f} {y:.1f}")
        else:
            points.append(f"L {x:.1f} {y:.1f}")
            
    return " ".join(points)

def build_svg():
    paths = []
    
    # --- WINE BOTTLE ---
    # Bottle base (dense horizontal shading on left, sparse on right)
    for _ in range(15): paths.append(generate_scribble_path(100, 220, 300, 600, 15, vertical=False, chaotic=15))
    # Bottle neck
    for _ in range(8): paths.append(generate_scribble_path(135, 185, 100, 300, 10, vertical=False, chaotic=8))
    # Bottle cork area
    for _ in range(5): paths.append(generate_scribble_path(130, 190, 70, 100, 8, vertical=False, chaotic=5))
    
    # Vertical cross-hatching for bottle
    for _ in range(10): paths.append(generate_scribble_path(100, 130, 300, 600, 12, vertical=True, chaotic=10)) # dark left edge
    for _ in range(10): paths.append(generate_scribble_path(190, 220, 300, 600, 12, vertical=True, chaotic=10)) # dark right edge
    for _ in range(5): paths.append(generate_scribble_path(135, 145, 100, 300, 8, vertical=True, chaotic=5)) # neck left
    for _ in range(5): paths.append(generate_scribble_path(175, 185, 100, 300, 8, vertical=True, chaotic=5)) # neck right

    # --- WINE GLASS ---
    # Bowl
    for _ in range(10): paths.append(generate_scribble_path(240, 360, 350, 480, 15, vertical=False, chaotic=12))
    for _ in range(8): paths.append(generate_scribble_path(240, 260, 350, 480, 10, vertical=True, chaotic=10))
    for _ in range(8): paths.append(generate_scribble_path(340, 360, 350, 480, 10, vertical=True, chaotic=10))
    
    # Stem
    for _ in range(5): paths.append(generate_scribble_path(290, 310, 480, 580, 6, vertical=True, chaotic=4))
    
    # Base
    for _ in range(6): paths.append(generate_scribble_path(250, 350, 570, 590, 10, vertical=False, chaotic=5))
    
    # Liquid inside glass (dense scribbles)
    for _ in range(8): paths.append(generate_scribble_path(245, 355, 420, 475, 20, vertical=False, chaotic=15))

    
    # Build React Component String
    react_code = "export const wineScribbleDoodle = (\n  <g transform=\"translate(250, 50) scale(1.1)\">\n"
    for i, p in enumerate(paths):
        react_code += f"    <path className=\"sketch-line\" d=\"{p}\" />\n"
    react_code += "  </g>\n);\n"
    
    return react_code

if __name__ == "__main__":
    with open("ScribbleOutput.jsx", "w") as f:
        f.write(build_svg())
    print("Generated ScribbleOutput.jsx")
