"""
CUR / ICR benchmark test for RQ-Kmeans: standard vs. L3-Balanced (type1 & type2).

Metric definitions (aligned with OneSearch Table 2)
─────────────────────────────────────────────────────
CUR_L1        : unique L1 codes used / k1
CUR_L1xL2     : unique (L1, L2) path pairs used / (k1 * k2)
CUR_Total     : unique full SIDs / total_sid_space
                  total_sid_space = k1 * k2 * k3  (all modes, fair comparison)
ICR           : items whose SID is unique (count=1) / N  (one-to-one identifiability)

balanced_type:
  0 = standard global K-means on all layers
  1 = per-(L1,L2) parent balanced K-means on last layer  →  K1+K2+K1*K2*K3 centroids
  2 = per-L2 parent balanced K-means on last layer       →  K1+K2+K2*K3   centroids

Run
───
    cd rq-opq
    python test_cur_icr.py               # run all benchmarks
    python test_cur_icr.py --verbose     # print unique-SID counts too
    python test_cur_icr.py --n 5000     # faster run with fewer items
"""

import sys, os, argparse, tempfile, textwrap
from collections import Counter
from typing import Dict, List
import numpy as np
from rq_dynamic import quantitative_codebook

# Configurations: each k_list tested with balanced_type 0 / 1 / 2
CONFIGS = [
    # (display label,                    k_list,          balanced_type)
    ("512-512-512    std",               [512,  512,  512], 0),
    ("512-512-512    +l3bal_L1xL2",      [512,  512,  512], 1),
    ("512-512-512    +l3bal_L2",         [512,  512,  512], 2),
    ("1024-512-256   std",               [1024, 512,  256], 0),
    ("1024-512-256   +l3bal_L1xL2",      [1024, 512,  256], 1),
    ("1024-512-256   +l3bal_L2",         [1024, 512,  256], 2),
    ("2048-512-128   std",               [2048, 512,  128], 0),
    ("2048-512-128   +l3bal_L1xL2",      [2048, 512,  128], 1),
    ("2048-512-128   +l3bal_L2",         [2048, 512,  128], 2),
]

N_DEFAULT = 50000
D_DEFAULT = 64


def compute_metrics(IdList: List[np.ndarray], k_list: List[int], balanced_type: int) -> Dict:
    """
    Compute CUR_L1, CUR_L1xL2, CUR_Total, ICR, unique_sids, unique_items.

    CUR_Total denominator = k1*k2*k3 for all balanced_type values (fair comparison).
    ICR numerator = items whose full SID appears exactly once (one-to-one).
    """
    N  = len(IdList[0])
    k1 = k_list[0]
    k2 = k_list[1]
    k3 = k_list[2]

    # CUR_L1
    cur_l1 = len(np.unique(IdList[0])) / k1

    # CUR_L1xL2: unique (l1, l2) prefix pairs
    l1l2_pairs = set(zip(IdList[0].tolist(), IdList[1].tolist()))
    cur_l1l2   = len(l1l2_pairs) / (k1 * k2)

    # Full SID tuples
    sid_tuples  = list(zip(*[IdList[l].tolist() for l in range(len(k_list))]))
    unique_sids = len(set(sid_tuples))

    # CUR_Total (same denominator k1*k2*k3 for all modes)
    total_space = k1 * k2 * k3
    cur_total   = unique_sids / total_space

    # ICR: fraction of items with an exclusively-owned SID
    sid_counts   = Counter(sid_tuples)
    unique_items = sum(1 for cnt in sid_counts.values() if cnt == 1)
    icr = unique_items / N

    return {
        "CUR_L1":       cur_l1,
        "CUR_L1xL2":    cur_l1l2,
        "CUR_Total":    cur_total,
        "ICR":          icr,
        "unique_sids":  unique_sids,
        "unique_items": unique_items,
        "N":            N,
    }


def _fmt(v: float) -> str:
    return f"{v * 100:.2f}%"


def _print_row(label: str, m: Dict, verbose: bool = False):
    row = (f"  {label:<32}"
           f"CUR_L1={_fmt(m['CUR_L1'])}  "
           f"CUR_L1xL2={_fmt(m['CUR_L1xL2'])}  "
           f"CUR_Total={_fmt(m['CUR_Total'])}  "
           f"ICR={_fmt(m['ICR'])}")
    if verbose:
        row += (f"  [unique_sids={m['unique_sids']:,}"
                f"  unique_items={m['unique_items']:,}"
                f" / N={m['N']:,}]")
    print(row)


def _make_gaussian_mixture(N, d, n_centers=512, spread=1.0, seed=0):
    """Data generators - Balanced: each centre equally likely."""
    rng     = np.random.default_rng(seed)
    centres = (rng.standard_normal((n_centers, d)) * 5).astype(np.float32)
    assign  = rng.integers(0, n_centers, size=N)
    noise   = (rng.standard_normal((N, d)) * spread).astype(np.float32)
    return centres[assign] + noise


def _make_power_law(N, d, n_centers=512, alpha=1.5, seed=0):
    """
    Data generators
    Unbalanced: centre popularity ∝ rank^{-alpha} (Zipf-like).
    Mimics e-commerce catalogues where a few categories dominate traffic.
    """
    rng     = np.random.default_rng(seed)
    centres = (rng.standard_normal((n_centers, d)) * 5).astype(np.float32)
    w       = np.array([(i + 1) ** (-alpha) for i in range(n_centers)])
    w      /= w.sum()
    assign  = rng.choice(n_centers, size=N, p=w)
    noise   = (rng.standard_normal((N, d)) * 0.5).astype(np.float32)
    return centres[assign] + noise

def run_one(M, k_list, balanced_type, tmpdir):
    _, IdList = quantitative_codebook(
        folder_path=tmpdir,
        refer_folder='',
        M=M.copy(),
        k_list=k_list,
        start_L=0,
        L=len(k_list),
        is_norm=0,
        balanced_type=balanced_type,
    )
    return compute_metrics(IdList, k_list, balanced_type)

def run_benchmark(dist: str, N: int, D: int, verbose: bool = False) -> Dict:
    if dist == "balanced":
        M = _make_gaussian_mixture(N, D, n_centers=512, spread=1.0, seed=42)
        title = f"Balanced Gaussian mixture  (N={N:,}, d={D}, centers=512)"
    else:
        M = _make_power_law(N, D, n_centers=512, alpha=1.5, seed=42)
        title = f"Power-law Gaussian mixture (N={N:,}, d={D}, centers=512, α=1.5)"

    print(f"\n{'='*100}")
    print(f"  Distribution: {title}")
    print(f"{'='*100}")
    print(f"  {'Config':<32}CUR_L1        CUR_L1xL2     CUR_Total     ICR")
    print(f"  {'-'*96}")

    results = {}
    for label, k_list, bt in CONFIGS:
        with tempfile.TemporaryDirectory() as tmpdir:
            m = run_one(M, k_list, bt, tmpdir)
        results[label] = m
        _print_row(label, m, verbose)

    return results

def test_param_counts():
    """
    Verify centroid counts for all three balanced_type values:
      type 0 (standard)  : k1 + k2 + k3
      type 1 (L1×L2 bal) : k1 + k2 + k1*k2*k3
      type 2 (L2 bal)    : k1 + k2 + k2*k3
    """
    print("\n[TEST] test_param_counts")
    k_list = [16, 8, 4]   # small so test is fast
    M = _make_gaussian_mixture(2000, 32, n_centers=16, seed=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        RQ0, _ = quantitative_codebook(tmpdir, '', M.copy(), k_list, 0, 3, 0, balanced_type=0)
        RQ1, _ = quantitative_codebook(tmpdir, '', M.copy(), k_list, 0, 3, 0, balanced_type=1)
        RQ2, _ = quantitative_codebook(tmpdir, '', M.copy(), k_list, 0, 3, 0, balanced_type=2)

    k1, k2, k3 = k_list

    cnt0 = sum(c.shape[0] for c in RQ0)
    cnt1 = sum(c.shape[0] for c in RQ1)
    cnt2 = sum(c.shape[0] for c in RQ2)

    exp0 = k1 + k2 + k3
    exp1 = k1 + k2 + k1 * k2 * k3
    exp2 = k1 + k2 + k2 * k3

    assert cnt0 == exp0, f"type0 centroid count {cnt0} != {exp0}"
    assert cnt1 == exp1, f"type1 centroid count {cnt1} != {exp1}"
    assert cnt2 == exp2, f"type2 centroid count {cnt2} != {exp2}"

    print(f"  type0 (std)   centroids: {cnt0:>8,}  = {k1}+{k2}+{k3}")
    print(f"  type1 (L1×L2) centroids: {cnt1:>8,}  = {k1}+{k2}+{k1}×{k2}×{k3}")
    print(f"  type2 (L2)    centroids: {cnt2:>8,}  = {k1}+{k2}+{k2}×{k3}")
    print("[PASS] test_param_counts\n")


def _assert_balanced_beats_std(res, std_lbl, bal_lbl, tag=""):
    """Both unique_sids and ICR of balanced must be >= standard."""
    s, b = res[std_lbl], res[bal_lbl]
    assert b["unique_sids"] >= s["unique_sids"], (
        f"{tag}[{bal_lbl}] unique_sids {b['unique_sids']:,} < [{std_lbl}] {s['unique_sids']:,}"
    )
    assert b["ICR"] >= s["ICR"], (
        f"{tag}[{bal_lbl}] ICR {_fmt(b['ICR'])} < [{std_lbl}] {_fmt(s['ICR'])}"
    )


def test_cur_icr_balanced_gaussian(N: int, D: int, verbose: bool = False):
    """
    On balanced Gaussian data:
    1. CUR_L1 >= 90% for every config.
    2. Both type1 and type2 balanced >= standard in unique_sids and ICR.
    3. CUR_L1xL2 non-increasing as k1 grows (std configs only).
    """
    print("\n[TEST] test_cur_icr_balanced_gaussian")
    res = run_benchmark("balanced", N, D, verbose)

    # 1. CUR_L1 >= 90%
    for label, k_list, _ in CONFIGS:
        v = res[label]["CUR_L1"]
        assert v >= 0.90, f"{label}: CUR_L1={_fmt(v)} < 90%"
    print("  [PASS] CUR_L1 >= 90% for all configs")

    # 2. balanced type1 and type2 must beat std for each k_list group
    # Groups: each 3 consecutive entries share the same k_list
    for i in range(0, len(CONFIGS), 3):
        std_lbl  = CONFIGS[i][0]
        t1_lbl   = CONFIGS[i+1][0]
        t2_lbl   = CONFIGS[i+2][0]
        _assert_balanced_beats_std(res, std_lbl, t1_lbl, "type1 ")
        _assert_balanced_beats_std(res, std_lbl, t2_lbl, "type2 ")
        s, b1, b2 = res[std_lbl], res[t1_lbl], res[t2_lbl]
        print(f"  [PASS] {std_lbl:<24} ICR={_fmt(s['ICR'])} unique={s['unique_sids']:,}")
        print(f"         {t1_lbl:<24} ICR={_fmt(b1['ICR'])} unique={b1['unique_sids']:,}  (type1 >=std ✓)")
        print(f"         {t2_lbl:<24} ICR={_fmt(b2['ICR'])} unique={b2['unique_sids']:,}  (type2 >=std ✓)")

    # 3. CUR_L1xL2 non-increasing as k1 grows (std only)
    std_lbls = [lbl for lbl, _, bt in CONFIGS if bt == 0]
    if len(std_lbls) >= 2:
        prev_lxl2 = res[std_lbls[0]]["CUR_L1xL2"]
        for lbl in std_lbls[1:]:
            cur_lxl2 = res[lbl]["CUR_L1xL2"]
            assert cur_lxl2 <= prev_lxl2, (
                f"CUR_L1xL2 should be non-increasing as k1 grows: "
                f"{lbl} ({_fmt(cur_lxl2)}) > prev ({_fmt(prev_lxl2)})"
            )
            prev_lxl2 = cur_lxl2
        print("  [PASS] CUR_L1xL2 non-increasing as k1 grows")

    print("[PASS] test_cur_icr_balanced_gaussian\n")
    return res


def test_cur_icr_powerlaw(N: int, D: int, verbose: bool = False):
    """
    On power-law (skewed) data:
    1. Both type1 and type2 balanced >= standard in unique_sids and ICR.
    2. If standard < 100% ICR, both balanced must strictly exceed it.
    """
    print("\n[TEST] test_cur_icr_powerlaw")
    res = run_benchmark("powerlaw", N, D, verbose)

    for i in range(0, len(CONFIGS), 3):
        std_lbl = CONFIGS[i][0]
        t1_lbl  = CONFIGS[i+1][0]
        t2_lbl  = CONFIGS[i+2][0]
        s, b1, b2 = res[std_lbl], res[t1_lbl], res[t2_lbl]

        _assert_balanced_beats_std(res, std_lbl, t1_lbl, "type1 ")
        _assert_balanced_beats_std(res, std_lbl, t2_lbl, "type2 ")

        if s["ICR"] < 1.0:
            assert b1["unique_sids"] > s["unique_sids"], (
                f"type1 must strictly improve when std ICR < 100%: "
                f"{b1['unique_sids']:,} vs {s['unique_sids']:,}"
            )
            assert b2["unique_sids"] > s["unique_sids"], (
                f"type2 must strictly improve when std ICR < 100%: "
                f"{b2['unique_sids']:,} vs {s['unique_sids']:,}"
            )

        r1 = b1["unique_sids"] / s["unique_sids"] if s["unique_sids"] else float('inf')
        r2 = b2["unique_sids"] / s["unique_sids"] if s["unique_sids"] else float('inf')
        print(f"  [PASS] {std_lbl:<24} ICR={_fmt(s['ICR'])} unique={s['unique_sids']:,}")
        print(f"         {t1_lbl:<24} ICR={_fmt(b1['ICR'])} unique={b1['unique_sids']:,}  ({r1:.3f}x, type1)")
        print(f"         {t2_lbl:<24} ICR={_fmt(b2['ICR'])} unique={b2['unique_sids']:,}  ({r2:.3f}x, type2)")

    print("[PASS] test_cur_icr_powerlaw\n")
    return res


def test_larger_k1_improves_icr(N: int, D: int, verbose: bool = False):
    """
    Larger k1 → more unique SIDs (higher ICR) on power-law data.
    Also: larger k1 → LOWER CUR_L1xL2.
    Tested for all three balanced_type values.
    """
    print("\n[TEST] test_larger_k1_improves_icr")
    M = _make_power_law(N, D, n_centers=512, alpha=1.5, seed=7)

    for bt in [0, 1, 2]:
        print(f"  -- balanced_type={bt} --")
        entries = [(lbl, k) for lbl, k, b in CONFIGS if b == bt]
        prev_unique = 0
        prev_lxl2   = float('inf')
        prev_lbl    = None
        for lbl, k_list in entries:
            with tempfile.TemporaryDirectory() as tmpdir:
                m = run_one(M, k_list, bt, tmpdir)
            print(f"    {lbl:<30} ICR={_fmt(m['ICR'])} unique={m['unique_sids']:,} "
                  f"CUR_L1xL2={_fmt(m['CUR_L1xL2'])}")
            if prev_lbl is not None:
                assert m["unique_sids"] >= prev_unique, (
                    f"bt={bt}: {lbl} unique_sids {m['unique_sids']:,} < {prev_lbl} {prev_unique:,}"
                )
                assert m["CUR_L1xL2"] <= prev_lxl2, (
                    f"bt={bt}: {lbl} CUR_L1xL2 {_fmt(m['CUR_L1xL2'])} > {prev_lbl} {_fmt(prev_lxl2)}"
                )
            prev_unique = m["unique_sids"]
            prev_lxl2   = m["CUR_L1xL2"]
            prev_lbl    = lbl
        print(f"  [PASS] bt={bt}: unique_sids non-decreasing, CUR_L1xL2 non-increasing as k1 grows")

    print("[PASS] test_larger_k1_improves_icr\n")


def test_all_configs_summary(N: int, D: int, verbose: bool = False):
    """
    Print a full side-by-side comparison table for both distributions.
    No hard assertions — human-readable sanity check.
    """
    print("\n[INFO] Full summary table — both distributions")
    res_bal = run_benchmark("balanced", N, D, verbose)
    res_pow = run_benchmark("powerlaw", N, D, verbose)

    print(f"\n{'─'*150}")
    print(f"  {'Config':<34} {'Gaussian ICR':>14} {'Powerlaw ICR':>14} "
          f"{'Gaussian CUR_Total':>20} {'Powerlaw CUR_Total':>20} "
          f"{'Gaussian uniq_SIDs':>20} {'Powerlaw uniq_SIDs':>20}")
    print(f"  {'─'*150}")
    for label, _, _ in CONFIGS:
        gb = res_bal.get(label, {})
        gp = res_pow.get(label, {})
        print(f"  {label:<34} "
              f"{_fmt(gb.get('ICR', 0)):>14} "
              f"{_fmt(gp.get('ICR', 0)):>14} "
              f"{_fmt(gb.get('CUR_Total', 0)):>20} "
              f"{_fmt(gp.get('CUR_Total', 0)):>20} "
              f"{gb.get('unique_sids', 0):>20,} "
              f"{gp.get('unique_sids', 0):>20,}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="CUR/ICR benchmark: standard vs. L3-Balanced (type1 & type2)")
    ap.add_argument("--verbose", action="store_true",
                    help="Print unique-SID counts alongside percentages")
    ap.add_argument("--n", type=int, default=N_DEFAULT,
                    help=f"Number of items (default {N_DEFAULT})")
    ap.add_argument("--d", type=int, default=D_DEFAULT,
                    help=f"Embedding dimension (default {D_DEFAULT})")
    args = ap.parse_args()

    N = args.n
    D = args.d

    print(textwrap.dedent(f"""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  CUR / ICR Benchmark: Standard vs. L3-Balanced (type1=L1×L2, type2=L2) ║
    ║  N={N:,}  d={D}                                                         ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    Note on CUR_Total:
      CUR_Total = unique_SIDs / (k1*k2*k3).  Same denominator for all types.
      Key metrics: ICR (one-to-one items) and unique_sids.
    """))

    passed = 0
    failed = 0

    tests = [
        ("test_param_counts",           lambda: test_param_counts()),
        ("test_cur_icr_balanced_data",  lambda: test_cur_icr_balanced_gaussian(N, D, args.verbose)),
        ("test_cur_icr_powerlaw",       lambda: test_cur_icr_powerlaw(N, D, args.verbose)),
        ("test_larger_k1_improves_icr", lambda: test_larger_k1_improves_icr(N, D, args.verbose)),
        ("test_all_configs_summary",    lambda: test_all_configs_summary(N, D, args.verbose)),
    ]

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"\n  ✗ [{name}] FAILED: {e}\n")
            import traceback; traceback.print_exc()
            failed += 1
        except Exception as e:
            print(f"\n  ✗ [{name}] ERROR: {e}\n")
            import traceback; traceback.print_exc()
            failed += 1

    bar = "═" * 72
    print(f"\n{bar}")
    status = "ALL PASSED ✓" if failed == 0 else f"{failed} FAILED ✗"
    print(f"  Results: {passed} passed, {failed} failed / {passed+failed} total  —  {status}")
    print(f"{bar}\n")
    if failed:
        sys.exit(1)
