import io
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from autodevops_cli import (
    ChoiceOption,
    ToggleOption,
    build_options,
    compile_config,
    SystemInfo,
    cpu_profile_instructions,
    backend_instructions,
    runtime_profile_instructions,
    quantization_notes,
    launch_build,
)


class BuildOptionsTestCase(unittest.TestCase):
    def option_by_key(self, options, key):
        for opt in options:
            if getattr(opt, "key", None) == key:
                return opt
        self.fail(f"Option with key '{key}' not found")

    def make_system_info(
        self,
        *,
        cpu_vendor="intel",
        cpu_flags=None,
        arch="x86_64",
        gpu_vendor="nvidia",
        cuda_home=Path("/usr/local/cuda"),
        has_mkl=True,
        has_openblas=True,
        has_blis=False,
    ):
        return SystemInfo(
            cpu_vendor=cpu_vendor,
            cpu_flags=set(cpu_flags or {}),
            arch=arch,
            gpu_vendor=gpu_vendor,
            cuda_home=cuda_home,
            has_mkl=has_mkl,
            has_openblas=has_openblas,
            has_blis=has_blis,
        )

    def test_cuda_path_enables_fast_math(self):
        info = self.make_system_info(cpu_flags={"avx2", "avx512f"})
        options = build_options(system_info=info)

        backend = self.option_by_key(options, "backend")
        self.assertIsInstance(backend, ChoiceOption)
        cuda_choice = next(choice for choice in backend.choices if choice.value == "cuda")
        self.assertTrue(cuda_choice.enabled)

        fast_math = self.option_by_key(options, "fast_math")
        self.assertIsInstance(fast_math, ToggleOption)
        self.assertFalse(fast_math.disabled)

    def test_missing_cuda_disables_fast_math(self):
        info = self.make_system_info(
            cpu_vendor="amd",
            cpu_flags={"avx2"},
            gpu_vendor="unknown",
            cuda_home=None,
            has_mkl=False,
            has_openblas=True,
            has_blis=True,
        )
        options = build_options(system_info=info)

        backend = self.option_by_key(options, "backend")
        cuda_choice = next(choice for choice in backend.choices if choice.value == "cuda")
        self.assertFalse(cuda_choice.enabled)

        fast_math = self.option_by_key(options, "fast_math")
        self.assertTrue(fast_math.disabled)
        self.assertIn("NVCC", fast_math.reason)

    def test_compile_config_collects_values(self):
        info = self.make_system_info()
        options = build_options(system_info=info)

        backend = self.option_by_key(options, "backend")
        backend.index = next(i for i, c in enumerate(backend.choices) if c.value == "vulkan")

        blas = self.option_by_key(options, "blas")
        blas.index = next(i for i, c in enumerate(blas.choices) if c.value == "openblas")

        fast_math = self.option_by_key(options, "fast_math")
        fast_math.value = True

        config = compile_config(options)
        self.assertEqual(config["backend"], "vulkan")
        self.assertEqual(config["blas"], "openblas")
        self.assertTrue(config["fast_math"])

    def test_instruction_helpers(self):
        self.assertIn("GGML_AVX", cpu_profile_instructions("intel_avx2"))
        self.assertEqual("", cpu_profile_instructions("unknown"))

        self.assertIn("GGML_CUDA", backend_instructions("cuda"))
        self.assertEqual("", backend_instructions("something_else"))

        self.assertIn("llama-cli", runtime_profile_instructions("balanced"))
        self.assertEqual("", runtime_profile_instructions("invalid"))

        self.assertIn("GGUF", quantization_notes("q4_k_m"))
        self.assertEqual("", quantization_notes("auto"))

    def test_launch_build_reports_detection_metadata(self):
        stdout = io.StringIO()
        stderr = io.StringIO()
        config = {
            "ref": "latest",
            "backend": "cuda",
            "cpu_profile": "auto",
            "blas": "auto",
            "detected_gpu_vendor": "nvidia",
            "detected_cuda_home": "/usr/local/cuda",
            "now": False,
        }
        with redirect_stdout(stdout), redirect_stderr(stderr):
            rc = launch_build(config)

        self.assertEqual(rc, 0)
        output = stdout.getvalue()
        self.assertIn("Detected GPU vendor: nvidia", output)
        self.assertIn("Detected CUDA toolkit: /usr/local/cuda", output)


if __name__ == "__main__":
    unittest.main()
