"""``sgc`` command line: build the data, engineer features, run the leakage tests."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from savings_goal.config import FEATURES_PATH, HOUSEHOLDS_PATH, ROOT, THRESHOLD


def _build(args: argparse.Namespace) -> int:
    import polars as pl

    from savings_goal.data.build import reconcile_totals, write
    from savings_goal.data.schema import validate_households

    report = write(args.tsv, args.out, args.threshold)
    df = pl.read_parquet(args.out) if args.out.suffix != ".csv" else pl.read_csv(args.out)
    if args.out.suffix != ".csv":
        validate_households(df)
    print(report)
    rec = reconcile_totals(df, args.threshold)
    print(
        f"COTOTAL vs 11-category sum: within 1% for {rec['within_1pct']:.2%}; "
        f"label agreement {rec['label_agreement']:.4f}"
    )
    print(f"wrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")
    if args.manifest:
        from savings_goal.manifest import write_manifest

        info = write_manifest(args.manifest, args.tsv)
        print(f"wrote {args.manifest}  (input sha256 {info['input']['sha256'][:16]}...)")
    if args.features:
        return _features(argparse.Namespace(households=args.out, out=args.features))
    return 0


def _features(args: argparse.Namespace) -> int:
    import pandas as pd

    from savings_goal.features.engineer import engineer

    feats, spec = engineer(pd.read_parquet(args.households))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(args.out, index=False)
    print(
        f"features: {len(spec.features)} ({len(spec.numeric)} numeric), "
        f"{len(feats):,} households, test fold {feats['Is_Test'].mean():.1%}"
    )
    print(f"participation indicators: {spec.indicators}")
    print(f"wrote {args.out}")
    return 0


def _train(args: argparse.Namespace) -> int:
    import pandas as pd

    from savings_goal.api import train_tiers

    for k, v in train_tiers(pd.read_parquet(args.features), args.artifacts).items():
        print(f"{k}: {v:.4f}")
    print(f"wrote {args.artifacts}/tier1.joblib, tier2.joblib")
    return 0


def _demo(args: argparse.Namespace) -> int:
    """End to end on a synthetic survey: no ICPSR registration, a few minutes at most."""
    import numpy as np
    import pandas as pd

    from savings_goal.config import CATEGORICALS, SEED
    from savings_goal.data.build import write
    from savings_goal.data.synthetic import synthetic_raw, write_tsv
    from savings_goal.evaluation import leakage_sim, outreach
    from savings_goal.evaluation.cv import oof_predict_proba
    from savings_goal.evaluation.metrics import targeting_table
    from savings_goal.features.engineer import engineer
    from savings_goal.models.pipeline import xgb_pipeline

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"SYNTHETIC data ({args.n} households): this checks the code path, not Indian households."
    )
    tsv = write_tsv(synthetic_raw(args.n, seed=SEED), out / "36151-0002-Data.tsv")
    write(tsv, out / "households.parquet")
    feats, spec = engineer(pd.read_parquet(out / "households.parquet"))
    cols = spec.deployable_numeric + CATEGORICALS
    y, g = feats["Goal_Met"], feats["IDPSU"]
    model = xgb_pipeline(spec.deployable_numeric)
    p = oof_predict_proba(model, feats[cols], y, g)
    rng = np.random.default_rng(SEED)
    scores = {
        "Risk model": 1 - p,
        "Income rule": -feats["INCOME"].to_numpy() + rng.random(len(feats)) * 1e-6,
        "Random contact": rng.random(len(feats)),
    }
    tt = targeting_table((1 - y).to_numpy(), scores, budgets=(0.25,))
    print("\nTargeting at a 25% budget (recall / precision / ceiling capture):")
    print(tt.round(3).to_string(index=False))
    relaxed = (feats["Savings_Rate"] >= 0.15).astype(int)
    p_relaxed = oof_predict_proba(model, feats[cols], relaxed, g)
    scores["Reachability model"] = outreach.reach_score(p_relaxed, p)
    sim = outreach.simulate(feats, scores, outreach.uniform(), [0.25], delta=0.05, cost=150.0)
    print("\nOutreach what-if (assumed response; delta 5 pp; Rs 150 per contact):")
    print(sim[["strategy", "flips_per_1000", "cost_per_flip"]].round(1).to_string(index=False))
    print("\nLeakage curve (oracle ROC-AUC by residual sd of log food on log size):")
    print(leakage_sim.auc_grid((0.0, 0.5, 1.0), (0.63,)).round(3).to_string(index=False))
    tt.to_csv(out / "demo_targeting.csv", index=False)
    print(f"\nwrote {out}/ (households.parquet, demo_targeting.csv)")
    return 0


def _reproduce(args: argparse.Namespace) -> int:
    """Rebuild the data from the ICPSR file, run every notebook, regenerate the figures."""
    import subprocess
    import sys

    from savings_goal.config import RESULTS
    from savings_goal.manifest import write_manifest

    code = _build(
        argparse.Namespace(
            tsv=args.tsv,
            out=HOUSEHOLDS_PATH,
            threshold=THRESHOLD,
            features=FEATURES_PATH,
            manifest=RESULTS / "run_manifest.json",
        )
    )
    if code:
        return code
    slow = ("04_", "05_") if args.fast else ()
    for nb in sorted((ROOT / "notebooks").glob("[0-9][0-9]_*.ipynb")):
        if nb.name.startswith(slow):
            print(f"skipping {nb.name} (--fast: the committed results/model_final.json is used)")
            continue
        print(f"running {nb.name}")
        cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute"]
        subprocess.run(
            [*cmd, "--inplace", "--ExecutePreprocessor.timeout=6000", str(nb)], check=True
        )
    subprocess.run(
        [sys.executable, str(ROOT / "project" / "figures" / "make_figures.py")], check=True
    )
    write_manifest(RESULTS / "run_manifest.json", args.tsv)
    return 0


def _serve(args: argparse.Namespace) -> int:  # pragma: no cover - starts a server
    import uvicorn

    from savings_goal.api import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sgc", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="DS0002 TSV -> households.parquet (validated)")
    b.add_argument("--tsv", required=True, type=Path, help="36151-0002-Data.tsv")
    b.add_argument("--out", default=HOUSEHOLDS_PATH, type=Path)
    b.add_argument(
        "--threshold",
        default=THRESHOLD,
        type=float,
        help="savings-rate benchmark defining Goal_Met (default 0.20)",
    )
    b.add_argument(
        "--features",
        type=Path,
        default=FEATURES_PATH,
        help="also write the engineered feature table here",
    )
    b.add_argument("--no-features", dest="features", action="store_const", const=None)
    b.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="also write a run manifest (versions, commit, input SHA-256) here",
    )
    b.set_defaults(func=_build)

    f = sub.add_parser("features", help="households.parquet -> features.parquet")
    f.add_argument("--households", default=HOUSEHOLDS_PATH, type=Path)
    f.add_argument("--out", default=FEATURES_PATH, type=Path)
    f.set_defaults(func=_features)

    t = sub.add_parser("train", help="fit the two scoring tiers -> artifacts/")
    t.add_argument("--features", default=FEATURES_PATH, type=Path)
    t.add_argument("--artifacts", default=ROOT / "artifacts", type=Path)
    t.set_defaults(func=_train)

    d = sub.add_parser("demo", help="run the pipeline on a synthetic survey (no ICPSR data needed)")
    d.add_argument("--out", default=ROOT / "demo_out", type=Path)
    d.add_argument("--n", default=3000, type=int, help="synthetic households")
    d.set_defaults(func=_demo)

    r = sub.add_parser("reproduce", help="ICPSR TSV -> data, all notebooks, figures, manifest")
    r.add_argument("--tsv", required=True, type=Path, help="36151-0002-Data.tsv")
    r.add_argument(
        "--fast",
        action="store_true",
        help="skip the two slow notebooks (04 tuning, 05 SHAP); committed tuning results are used",
    )
    r.set_defaults(func=_reproduce)

    s = sub.add_parser("serve", help="run the FastAPI scoring service (needs --extra api)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", default=8000, type=int)
    s.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
