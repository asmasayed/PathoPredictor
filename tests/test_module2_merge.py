"""Merge metadata + embeddings into Module 2 CSVs (no torch)."""

import json
import unittest
from pathlib import Path
import tempfile

import pandas as pd

from src.module2_data.merge_tables import build_classifier_csv, build_regressor_csv


class TestMergeModule2(unittest.TestCase):
    def test_build_classifier_and_regressor_with_pheno(self):
        with tempfile.TemporaryDirectory() as tmp:
            td = Path(tmp)
            meta = td / "m.json"
            meta.write_text(
                json.dumps(
                    [
                        {"strain_id": "A/test1/X/1/2020", "host": "Chicken", "sequence": "ATCG"},
                        {"strain_id": "A/test2/X/2/2020", "host": "Human", "sequence": "GCTA"},
                        {"strain_id": "A/skipped/X/3/2020", "host": "UnknownPlanet", "sequence": "AAAC"},
                    ]
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                {
                    "strain_id": ["A/test1/X/1/2020", "A/test2/X/2/2020"],
                    "emb_0": [0.0, 1.0],
                    "emb_1": [1.0, 0.0],
                }
            ).to_csv(td / "e.csv", index=False)
            pd.DataFrame(
                {
                    "strain_id": ["A/test1/X/1/2020", "A/test2/X/2/2020"],
                    "beta": [0.2, 0.25],
                    "gamma": [0.1, 0.1],
                    "sigma": [0.2, 0.22],
                }
            ).to_csv(td / "p.csv", index=False)

            oc = td / "c.csv"
            or_ = td / "r.csv"
            nc = build_classifier_csv(meta, td / "e.csv", oc)
            self.assertEqual(nc, 2)
            nr = build_regressor_csv(meta, td / "e.csv", or_, phenotype_path=td / "p.csv")
            self.assertEqual(nr, 2)
            cdf = pd.read_csv(oc)
            self.assertEqual(set(cdf["label"].tolist()), {0, 1})


if __name__ == "__main__":
    unittest.main()
