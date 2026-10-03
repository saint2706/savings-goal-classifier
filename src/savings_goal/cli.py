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
    b.set_defaults(func=_build)

    f = sub.add_parser("features", help="households.parquet -> features.parquet")
    f.add_argument("--households", default=HOUSEHOLDS_PATH, type=Path)
    f.add_argument("--out", default=FEATURES_PATH, type=Path)
    f.set_defaults(func=_features)

    t = sub.add_parser("train", help="fit the two scoring tiers -> artifacts/")
    t.add_argument("--features", default=FEATURES_PATH, type=Path)
    t.add_argument("--artifacts", default=ROOT / "artifacts", type=Path)
    t.set_defaults(func=_train)

    s = sub.add_parser("serve", help="run the FastAPI scoring service (needs --extra api)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", default=8000, type=int)
    s.set_defaults(func=_serve)

    args = parser.parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
