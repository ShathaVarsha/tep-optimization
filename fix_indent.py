from pathlib import Path
import ast

p = Path("compare_xmv11_optimization_response_bounded.py")
s = p.read_text(encoding="utf-8-sig")

s = s.replace(
    "        A = np.vstack(A_rows)\n    b = np.asarray(b_rows, dtype=float)",
    "    A = np.vstack(A_rows)\n    b = np.asarray(b_rows, dtype=float)"
)

p.write_text(s, encoding="utf-8")
ast.parse(s)
print("Syntax check passed.")
