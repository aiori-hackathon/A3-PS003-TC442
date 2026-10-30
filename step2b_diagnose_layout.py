"""
STEP 2b - Diagnose the real column layout and time order of the dataset.

    python step2b_diagnose_layout.py

Run this when step2 reports a layout or time-order problem. It only READS the
data and prints facts; it changes nothing. Paste the whole output to your
helper/mentor so config.py can be corrected.
"""
import re
import sys
import zipfile

import numpy as np

from src import config, data

np.set_printoptions(precision=3, suppress=True, linewidth=120)


def docx_text(path, limit=4000):
    """Plain text of a .docx without extra libraries."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf8", "ignore")
    xml = re.sub(r"</w:p>", "\n", xml)
    return re.sub(r"<[^>]+>", "", xml)[:limit]


def one_hot_share(flat, a, b):
    block = flat[:, a:b]
    return bool(np.isin(block, [0, 1]).all()), float(np.mean(block.sum(1) == 1))


def match(a, b):
    return float(np.isclose(a, b, rtol=1e-5, atol=1e-7).mean())


def main() -> int:
    try:
        raw = data.load_raw()
    except Exception as e:
        print(f"ERROR: {e}")
        return 1
    X_tr, Y_tr, X_te, Y_te = raw["X_tr"], raw["Y_tr"], raw["X_te"], raw["Y_te"]
    X = np.concatenate([X_tr, X_te])          # (2101, 36, F) in file order
    Y = np.concatenate([Y_tr, Y_te])
    T, S, F = X.shape
    flat = X.reshape(-1, F)

    print("=== A. Dataset documentation found in data/raw ===")
    docs = list(config.DATA_RAW.glob("*.docx"))
    if docs:
        for d in docs:
            print(f"--- {d.name} ---")
            print(docx_text(d))
    else:
        print("No .docx found (it may not have been extracted).")

    print("\n=== B. Every column from 10 onwards ===")
    print(" col |  min   max  | unique | share==1 | binary")
    for j in range(10, F):
        c = flat[:, j]
        u = np.unique(np.round(c, 6))
        print(f" {j:3d} | {c.min():5.2f} {c.max():5.2f} | {len(u):6d} | {np.mean(c == 1):8.3f} | "
              f"{bool(np.isin(c, [0, 1]).all())}   values: {u[:6]}{' ...' if len(u) > 6 else ''}")

    print("\n=== C. Candidate one-hot blocks (>= 95% of rows have exactly one 1) ===")
    found = []
    for L in (24, 7, 5, 4, 3, 2):
        for a in range(10, F - L + 1):
            binary, share = one_hot_share(flat, a, a + L)
            if binary and share >= 0.95:
                found.append((a, a + L, share))
    if found:
        for a, b, share in found:
            print(f"  columns {a}-{b - 1} (length {b - a}): {share:.1%} rows exactly one 1")
    else:
        print("  none found")
    # how many 1s per row in each column group, for the first rows
    print("  positions of non-zero values (cols >= 10) in the first 6 rows of sensor 0:")
    for t in range(6):
        nz = np.nonzero(X[t, 0, 10:])[0] + 10
        print(f"    t={t}: " + ", ".join(f"{j}={X[t, 0, j]:g}" for j in nz))

    print("\n=== D. Time-order tests (train + test in file order) ===")
    lag = X[:, :, 0:10]
    for newest in (0, 9):
        res = [f"t+{d}: {match(lag[d:, :, newest], Y[:-d]):.1%}" for d in (1, 2, 3, 4)]
        print(f"  newest lag = col {newest}: target[t] == newest lag at " + ", ".join(res))
    print(f"  lags shift by one step between rows (oldest->newest order): "
          f"{match(lag[1:, :, 0:9], lag[:-1, :, 1:10]):.1%}")
    print(f"  lags shift by one step between rows (newest->oldest order): "
          f"{match(lag[1:, :, 1:10], lag[:-1, :, 0:9]):.1%}")
    print("  target[t] equals lag column j of the SAME row t:     " +
          " ".join(f"{j}:{match(lag[:, :, j], Y):.0%}" for j in range(10)))
    print("  target[t] equals lag column j of the NEXT row t+1:   " +
          " ".join(f"{j}:{match(lag[1:, :, j], Y[:-1]):.0%}" for j in range(10)))

    lag_values = set(np.round(lag.ravel(), 6))
    in_lags = np.mean([v in lag_values for v in np.round(Y.ravel()[:20000], 6)])
    print(f"  share of target values that appear anywhere among lag values: {in_lags:.1%}")
    print(f"  lag range {lag.min():.3f}-{lag.max():.3f}, mean {lag.mean():.3f} | "
          f"target range {Y.min():.3f}-{Y.max():.3f}, mean {Y.mean():.3f}")

    print("\n=== E. Can a hidden time order be rebuilt? (successor search) ===")
    for name, src, dst in [("oldest->newest", slice(1, 10), slice(0, 9)),
                           ("newest->oldest", slice(0, 9), slice(1, 10))]:
        index = {}
        for b in range(T):
            index.setdefault(np.round(lag[b, :, dst], 6).tobytes(), []).append(b)
        succ = [index.get(np.round(lag[a, :, src], 6).tobytes(), []) for a in range(T)]
        found_any = np.mean([len(s) > 0 for s in succ])
        unique = np.mean([len(s) == 1 for s in succ])
        next_row = np.mean([s == [a + 1] for a, s in enumerate(succ)])
        print(f"  {name}: successor found for {found_any:.1%} of rows "
              f"(unique {unique:.1%}); successor is simply the next row for {next_row:.1%}")

    print("\nDone. Paste everything above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
