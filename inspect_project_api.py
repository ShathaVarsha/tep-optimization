import ast
from pathlib import Path

for filename in [
    "src/data/tep_loader.py",
    "src/models/surrogate_models.py",
]:
    path = Path(filename)
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))

    print(f"\n{'=' * 65}\n{filename}\n{'=' * 65}")

    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            print(f"\n{type(node).__name__}: {node.name}")
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        args = [a.arg for a in item.args.args]
                        print(f"  method: {item.name}({', '.join(args)})")
