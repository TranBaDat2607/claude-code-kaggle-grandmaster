"""A local NVIDIA GPU with a CPU-only torch build: training silently runs on the CPU, so the brief says so."""

import shutil

import pytest

from kgkit import gpu as G

CPU = """from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip', 'rocm', 'xpu']
__version__ = '2.14.0+cpu'
debug = False
cuda: Optional[str] = None
hip: Optional[str] = None
"""
CUDA = CPU.replace("'2.14.0+cpu'", "'2.14.0+cu130'").replace("cuda: Optional[str] = None",
                                                              "cuda: Optional[str] = '13.0'")
OLD_STYLE = "__version__ = '1.13.1'\ncuda = '11.7'\n"  # older wheels: no annotation


def fake_torch(tmp_path, version_py: str | None) -> list[str]:
    pkg = tmp_path / "site" / "torch"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("raise RuntimeError('must not be imported')\n")
    if version_py is not None:
        (pkg / "version.py").write_text(version_py)
    return [str(tmp_path / "site")]


@pytest.fixture
def with_driver(monkeypatch):
    real = shutil.which
    monkeypatch.setattr(G.shutil, "which", lambda name, *a, **k: "/usr/bin/nvidia-smi" if name == "nvidia-smi"
                        else real(name, *a, **k))


@pytest.mark.parametrize("text, want", [(CPU, ("2.14.0+cpu", None)), (CUDA, ("2.14.0+cu130", "13.0")),
                                        (OLD_STYLE, ("1.13.1", "11.7"))])
def test_installed_torch_reads_version_py_without_importing(tmp_path, text, want):
    t = G.installed_torch(fake_torch(tmp_path, text))
    assert (t["version"], t["cuda"]) == want


def test_no_torch_or_no_version_file(tmp_path):
    assert G.installed_torch([str(tmp_path)]) is None
    assert G.installed_torch(fake_torch(tmp_path, None)) is None


def test_cpu_build_with_a_driver_is_flagged(tmp_path, with_driver):
    line = G.cpu_torch_line(fake_torch(tmp_path, CPU))
    assert line.startswith("LOCAL GPU UNUSED") and "2.14.0+cpu" in line and "ask first" in line


def test_cuda_build_is_quiet(tmp_path, with_driver):
    assert G.cpu_torch_line(fake_torch(tmp_path, CUDA)) == ""


def test_no_driver_is_quiet(tmp_path, monkeypatch):
    monkeypatch.setattr(G.shutil, "which", lambda *a, **k: None)
    assert G.cpu_torch_line(fake_torch(tmp_path, CPU)) == ""


def test_no_torch_is_quiet(tmp_path, with_driver):
    assert G.cpu_torch_line([str(tmp_path)]) == ""
