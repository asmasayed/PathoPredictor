"""Tests for Module 2 training data IO, features, and inference helpers."""

import unittest
from pathlib import Path
import tempfile

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.module2_classifier.io_utils import load_classifier_training_arrays, write_synthetic_classifier_csv
from src.module2_classifier.predict_host_risk import predict_host_adaptation
from src.module2_common.features import compose_feature_vector, sequence_motif_features
from src.module2_regressor.io_utils import load_regressor_training_arrays, write_synthetic_regressor_csv


class TestModule2(unittest.TestCase):
    def test_sequence_motif_features_empty(self):
        v = sequence_motif_features(None)
        self.assertEqual(v.shape, (5,))
        self.assertTrue(np.allclose(v, 0))

    def test_compose_feature_vector(self):
        emb = np.ones(8)
        v = compose_feature_vector(emb, "ACGTACGT")
        self.assertEqual(v.shape[0], 8 + 5)

    def test_classifier_roundtrip_joblib(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "c.csv"
            write_synthetic_classifier_csv(csv_path, n_samples=80, embed_dim=12)
            X, y, cols = load_classifier_training_arrays(csv_path)
            self.assertEqual(X.shape[1], 12 + 5)
            self.assertEqual(len(cols), 12)

            scaler = StandardScaler().fit(X)
            clf = RandomForestClassifier(n_estimators=30, random_state=0).fit(scaler.transform(X), y)
            bundle_path = Path(tmp) / "b.joblib"
            import joblib

            joblib.dump({"model": clf, "scaler": scaler, "embedding_columns": cols, "feature_dim": X.shape[1]}, bundle_path)

            emb = X[0, :12]
            out = predict_host_adaptation(bundle_path, emb, sequence=None)
            self.assertIn("human_adaptation_probability", out)
            self.assertIn("risk_score_percent", out)

    def test_regressor_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "r.csv"
            write_synthetic_regressor_csv(csv_path, n_samples=60, embed_dim=10)
            X, y, cols = load_regressor_training_arrays(csv_path)
            self.assertEqual(X.shape[1], 10 + 5)
            self.assertEqual(y.shape[1], 3)


if __name__ == "__main__":
    unittest.main()
