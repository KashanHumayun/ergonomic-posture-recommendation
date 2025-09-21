from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergonomic_posture.pipeline import run_pipeline


class ErgonomicPipelineSmokeTest(unittest.TestCase):
    def test_pipeline_creates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            model_dir = output_dir / "models"
            project_root = Path(__file__).resolve().parents[1]
            summary = run_pipeline(project_root=project_root, output_dir=output_dir, model_dir=model_dir)
            self.assertIn("gradient_boosting", summary["models"])
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "posture_predictions.csv").exists())
            self.assertGreater(summary["models"]["knn"]["accuracy"], 0.8)


if __name__ == "__main__":
    unittest.main()
