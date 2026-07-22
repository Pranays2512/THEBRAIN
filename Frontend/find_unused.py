import os
import re

def get_all_files(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(('.js', '.jsx', '.css')):
                files.append(os.path.join(root, filename))
    return files

def extract_imports(filepath):
    imports = []
    with open(filepath, 'r') as f:
        content = f.read()
        
    # Match import '...' or import ... from '...'
    # This is a simple regex, it will find paths
    matches = re.findall(r'import\s+(?:.*?\s+from\s+)?[\'"](.*?)[\'"]', content)
    for match in matches:
        if match.startswith('.'): # Relative import
            # resolve path
            dir_path = os.path.dirname(filepath)
            resolved = os.path.normpath(os.path.join(dir_path, match))
            imports.append(resolved)
    return imports

def build_graph(entry_points, all_files):
    visited = set()
    queue = list(entry_points)
    
    # Helper to resolve an import path to an actual file
    def resolve_path(path):
        possible_extensions = ['', '.js', '.jsx', '.css', '/index.js', '/index.jsx']
        for ext in possible_extensions:
            if (path + ext) in all_files:
                return path + ext
        return None

    while queue:
        curr = queue.pop(0)
        if curr in visited:
            continue
            
        visited.add(curr)
        
        if curr.endswith(('.js', '.jsx')):
            imports = extract_imports(curr)
            for imp in imports:
                resolved = resolve_path(imp)
                if resolved and resolved not in visited:
                    queue.append(resolved)
                    
    return visited

def main():
    src_dir = os.path.abspath('src')
    all_files = set(get_all_files(src_dir))
    
    # Entry points
    entry_points = {
        os.path.join(src_dir, 'main.jsx'),
    }
    
    used_files = build_graph(entry_points, all_files)
    unused_files = all_files - used_files
    
    print("UNUSED FILES:")
    for f in sorted(unused_files):
        print(f)

if __name__ == '__main__':
    main()
