"""Thin ctypes wrapper exposing forward dispersion computation in Python.

The shared library ``libqedispinv_forward`` is built by CMake and placed under
``lib/``. This module provides a small convenience function to compute phase
velocities without going through the command-line interface.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Iterable

import numpy as np


_LIB_NAME = "libqedispinv_forward.so"
_LIB_PATH = Path(__file__).resolve().parent.parent / "lib" / _LIB_NAME


def _load_library(path: Path) -> ctypes.CDLL:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {_LIB_NAME} at {path}. Build it with `cmake --build "
            "build --target qedispinv_forward` and ensure the library output "
            "directory is available."
        )
    return ctypes.CDLL(str(path))


_lib = _load_library(_LIB_PATH)


_lib.qedi_dispersion_create.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_bool,
]
_lib.qedi_dispersion_create.restype = ctypes.c_void_p

_lib.qedi_dispersion_destroy.argtypes = [ctypes.c_void_p]
_lib.qedi_dispersion_destroy.restype = None

_lib.qedi_dispersion_calculate.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
]
_lib.qedi_dispersion_calculate.restype = ctypes.c_int

_lib.qedi_dispersion_search_mode.argtypes = [
    ctypes.c_void_p,
    ctypes.c_double,
    ctypes.c_int,
]
_lib.qedi_dispersion_search_mode.restype = ctypes.c_double


class DispersionCalculator:
    """A reusable dispersion calculator backed by the C++ implementation."""

    def __init__(self, model: np.ndarray, sh: bool = False) -> None:
        model_arr = np.ascontiguousarray(model, dtype=np.float64)
        if model_arr.ndim != 2:
            raise ValueError("model must be a 2-D array")
        if model_arr.shape[1] < 5:
            raise ValueError("model must have at least 5 columns")

        self._nrow, self._ncol = model_arr.shape
        ptr = model_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        handle = _lib.qedi_dispersion_create(
            ptr, ctypes.c_int(self._nrow), ctypes.c_int(self._ncol), ctypes.c_bool(sh)
        )
        if not handle:
            raise RuntimeError("Failed to initialize dispersion calculator")
        self._handle = handle

    def __del__(self) -> None:  # pragma: no cover - destructor safety
        if getattr(self, "_handle", None):
            _lib.qedi_dispersion_destroy(self._handle)
            self._handle = None

    def compute(self, freqs: Iterable[float], mode_max: int = 0) -> np.ndarray:
        freq_arr = np.ascontiguousarray(freqs, dtype=np.float64)
        if freq_arr.ndim != 1:
            raise ValueError("freqs must be 1-D")
        if mode_max < 0:
            raise ValueError("mode_max must be non-negative")

        nfreq = int(freq_arr.size)
        ncol = mode_max + 1
        out = np.empty((nfreq, ncol), dtype=np.float64)
        ret = _lib.qedi_dispersion_calculate(
            self._handle,
            freq_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(nfreq),
            ctypes.c_int(mode_max),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        if ret != 0:
            raise RuntimeError("Dispersion calculation failed")
        return out

    def search_mode(self, freq: float, mode: int) -> float:
        if mode < 0:
            raise ValueError("mode must be non-negative")
        return float(_lib.qedi_dispersion_search_mode(self._handle, freq, mode))


def compute_dispersion(
    model: np.ndarray, freqs: Iterable[float], mode_max: int = 0, sh: bool = False
) -> np.ndarray:
    """Compute dispersion curves for given frequencies and model.

    Parameters
    ----------
    model:
        2-D ``numpy`` array with at least five columns, matching the
        ``model_data.txt`` layout used by the command-line tools.
    freqs:
        Iterable of frequencies in Hz.
    mode_max:
        Highest mode (starting from 0) to compute. The return array has
        ``mode_max + 1`` columns.
    sh:
        ``True`` to compute Love waves, ``False`` for Rayleigh waves.
    """

    calc = DispersionCalculator(model, sh=sh)
    return calc.compute(freqs, mode_max=mode_max)


__all__ = ["DispersionCalculator", "compute_dispersion"]
