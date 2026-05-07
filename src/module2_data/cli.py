"""
Command-line entrypoints for Module 2 data preparation.

Examples (from project root):

  python -m src.module2_data.cli embed \\
    --model-dir models/module1_dnbert \\
    --metadata-json data/processed/module1/h5n1_metadata.json \\
    --out-csv data/processed/module2/embeddings.csv

  python -m src.module2_data.cli merge \\
    --metadata-json data/processed/module1/h5n1_metadata.json \\
    --embeddings-csv data/processed/module2/embeddings.csv \\
    --out-classifier data/processed/module2/classifier_train.csv \\
    --out-regressor data/processed/module2/regressor_train.csv \\
    --phenotype-csv data/processed/module2/phenotypes.csv

  # Demo regressor only (not for publication):
  python -m src.module2_data.cli merge ... --fallback-seir
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.module2_data.dnabert_embedding import embed_records_to_dataframe
from src.module2_data.merge_tables import build_classifier_csv, build_regressor_csv, load_metadata_records


def cmd_embed(args: argparse.Namespace) -> None:
    path = Path(args.metadata_json)
    records = load_metadata_records(path)
    df = embed_records_to_dataframe(records, args.model_dir)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows x {df.shape[1]} cols to {out}")


def cmd_merge(args: argparse.Namespace) -> None:
    meta = Path(args.metadata_json)
    emb = Path(args.embeddings_csv)
    oc = Path(args.out_classifier)
    n_c = build_classifier_csv(meta, emb, oc)
    print(f"Classifier CSV: {n_c} rows -> {oc}")
    or_path = Path(args.out_regressor)
    pheno = Path(args.phenotype_csv) if args.phenotype_csv else None
    n_r = build_regressor_csv(
        meta,
        emb,
        or_path,
        phenotype_path=pheno,
        fallback_seir=bool(args.fallback_seir),
        seir_jitter=float(args.seir_jitter),
        seed=int(args.seed),
    )
    print(f"Regressor CSV: {n_r} rows -> {or_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Module 2 dataset utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    p_e = sub.add_parser("embed", help="Compute DNABERT-pooled embeddings from metadata JSON")
    p_e.add_argument("--model-dir", type=str, required=True, help="Hugging Face model folder (same as mutation tool)")
    p_e.add_argument("--metadata-json", type=str, required=True, help="List of strain records with strain_id, sequence")
    p_e.add_argument("--out-csv", type=str, required=True, help="Output CSV: strain_id, emb_0, ...")

    p_m = sub.add_parser("merge", help="Merge metadata + embeddings into classifier/regressor CSVs")
    p_m.add_argument("--metadata-json", type=str, required=True)
    p_m.add_argument("--embeddings-csv", type=str, required=True)
    p_m.add_argument("--out-classifier", type=str, required=True)
    p_m.add_argument("--out-regressor", type=str, required=True)
    p_m.add_argument("--phenotype-csv", type=str, default="", help="strain_id,beta,gamma,sigma")
    p_m.add_argument("--fallback-seir", action="store_true", help="Fill SEIR targets from config + noise (demo only)")
    p_m.add_argument("--seir-jitter", type=float, default=0.02)
    p_m.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "embed":
        cmd_embed(args)
    elif args.command == "merge":
        cmd_merge(args)


if __name__ == "__main__":
    main()
