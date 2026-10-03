import ast
from pathlib import Path

p = Path("src/models/surrogate_models.py")
tree = ast.parse(p.read_text(encoding="utf-8-sig"))

for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "fit":
        if isinstance(getattr(node, "parent", None), ast.ClassDef):
            pass

lines = p.read_text(encoding="utf-8-sig").splitlines()
for start, end in [(1, 180)]:
    for i in range(start - 1, min(end, len(lines))):
        print(f"{i+1}: {lines[i]}")
