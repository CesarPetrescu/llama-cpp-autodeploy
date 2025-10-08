import unittest
from pathlib import Path
from unittest.mock import patch

import autodevops


class PickCudaHomeTests(unittest.TestCase):
    def test_prefers_cuda_home_env_variable(self):
        target = Path("/opt/cuda-12.9")

        with patch.dict(autodevops.os.environ, {"CUDA_HOME": str(target)}):
            with patch("autodevops._is_cuda_home", side_effect=lambda p: p == target):
                result = autodevops.pick_cuda_home()

        self.assertEqual(result, target)

    def test_falls_back_to_highest_versioned_install(self):
        candidates = [Path("/usr/local/cuda-12.9"), Path("/usr/local/cuda-12.8")]

        with patch.dict(autodevops.os.environ, {}, clear=True):
            with patch("autodevops._candidate_cuda_directories", return_value=candidates):
                with patch("autodevops._is_cuda_home", side_effect=lambda p: p == candidates[0]):
                    result = autodevops.pick_cuda_home()

        self.assertEqual(result, candidates[0])

    def test_uses_nvcc_location_as_last_resort(self):
        nvcc_path = Path("/opt/cuda/bin/nvcc")

        with patch.dict(autodevops.os.environ, {}, clear=True):
            with patch("autodevops._candidate_cuda_directories", return_value=[]):
                with patch("autodevops._is_cuda_home", return_value=False):
                    with patch("autodevops.shutil.which", return_value=str(nvcc_path)):
                        result = autodevops.pick_cuda_home()

        self.assertEqual(result, nvcc_path.parent.parent)


if __name__ == "__main__":
    unittest.main()
