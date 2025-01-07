from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ergonomic_posture.pipeline import run_demo


class ErgonomicPipelineSmokeTest(unittest.TestCase):
    def test_demo_pipeline_creates_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            summary = run_demo(output_dir=output_dir, n_subjects=6, windows_per_subject=18)
            self.assertIn("gradient_boosting", summary["models"])
            self.assertTrue((output_dir / "metrics.json").exists())
            self.assertTrue((output_dir / "posture_predictions.csv").exists())
            self.assertGreater(summary["best_model"]["accuracy"], 0.6)


if __name__ == "__main__":
    unittest.main()
