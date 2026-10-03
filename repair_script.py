from pathlib import Path
import ast

p = Path("compare_xmv11_optimization_response_bounded.py")
s = p.read_text(encoding="utf-8-sig")
lines = s.splitlines()

# Normalize indentation to spaces throughout the script.
fixed = "\n".join(line.expandtabs(4) for line in lines) + "\n"
p.write_text(fixed, encoding="utf-8")

try:
    ast.parse(fixed)
    print("Python syntax check passed.")
except SyntaxError as e:
    print(f"Syntax error remains at line {e.lineno}: {e.msg}")
    print("Nearby lines:")
    for i in range(max(0, (e.lineno or 1) - 4), min(len(lines), (e.lineno or 1) + 3)):
        print(f"{i+1}: {lines[i]!r}")
