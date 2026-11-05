"""Assemble src/attn_<v>.c = student code + the handout driver, unchanged."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRV = "/* ------------------------------------------------------------------------- */\n/* Driver: do not change below this line."

naive = (ROOT / "src" / "attn_naive.c").read_text()
driver = naive[naive.index(DRV):]

for v in ("loops", "tiled", "fused", "best"):
    body = (ROOT / "build" / f"b_{v}.c").read_text().rstrip() + "\n\n"
    out = body + driver
    (ROOT / "src" / f"attn_{v}.c").write_text(out)
    print(f"  attn_{v}.c  {out.count(chr(10)):4d} lines")
