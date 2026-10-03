from pathlib import Path

p = Path("src/process/variable_dictionary.py")
lines = p.read_text(encoding="utf-8").splitlines()

for tag in ("XMV_11", "XMEAS_17"):
    print(f"\n{'=' * 60}\n{tag}\n{'=' * 60}")
    matches = [i for i, line in enumerate(lines) if f'"{tag}"' in line]

    for i in matches:
        start = max(0, i - 1)
        end = min(len(lines), i + 12)
        print("\n".join(f"{j+1}: {lines[j]}" for j in range(start, end)))
