from pathlib import Path

p = Path("compare_xmv11_optimization_response_bounded.py")
s = p.read_text(encoding="utf-8")

marker = "A = np.vstack(A_rows)"
addition = '''if kind == "cost_minimization":
            target_production = float(df["XMEAS_17"].mean())
            A_rows.append(-prod_c)
            b_rows.append(prod_b - target_production)

        '''

if "target_production = float(df[\"XMEAS_17\"].mean())" in s:
    print("Production constraint already exists; no change made.")
elif marker not in s:
    raise SystemExit("Could not find the expected constraint-building line. File unchanged.")
else:
    s = s.replace(marker, addition + marker, 1)
    p.write_text(s, encoding="utf-8")
    print("Added minimum-production constraint for cost minimization.")
