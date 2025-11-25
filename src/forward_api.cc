/*
 QEDispInv: Surface Wave Dispersion Curve Computation and Inversion Toolkit

 GNU General Public License, Version 3, 29 June 2007

 Copyright (c) 2025 Lei Pan

 Xiaofei Chen Research Group,
 Department of Earth and Space Sciences,
 Southern University of Science and Technology, China.
 */

#include "disp.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <limits>

namespace {
struct DispersionHandle {
  explicit DispersionHandle(const Eigen::ArrayXXd &model, bool sh)
      : impl(model, sh) {}
  Dispersion impl;
};

Eigen::ArrayXXd map_model(const double *model, int nrow, int ncol) {
  return Eigen::Map<const Eigen::ArrayXXd>(model, nrow, ncol);
}

const double NaN = std::numeric_limits<double>::quiet_NaN();
} // namespace

extern "C" {
DispersionHandle *qedi_dispersion_create(const double *model, int nrow,
                                         int ncol, bool sh) {
  if (!model || nrow <= 0 || ncol < 5) {
    return nullptr;
  }

  Eigen::ArrayXXd model_mapped = map_model(model, nrow, ncol);
  return new DispersionHandle(model_mapped, sh);
}

int qedi_dispersion_calculate(const DispersionHandle *handle,
                              const double *freqs, int nfreq, int mode_max,
                              double *out) {
  if (!handle || !freqs || !out || nfreq <= 0 || mode_max < 0) {
    return 1;
  }

  const int ncol = mode_max + 1;
  for (int i = 0; i < nfreq; ++i) {
    const auto result = handle->impl.search(freqs[i], ncol);
    for (int m = 0; m < ncol; ++m) {
      const int idx = i * ncol + m;
      out[idx] = (m < static_cast<int>(result.size())) ? result[m] : NaN;
    }
  }
  return 0;
}

double qedi_dispersion_search_mode(const DispersionHandle *handle, double freq,
                                   int mode) {
  if (!handle || mode < 0) {
    return NaN;
  }
  const double vel = handle->impl.search_mode(freq, mode);
  if (std::isnan(vel)) {
    return NaN;
  }
  return vel;
}

void qedi_dispersion_destroy(DispersionHandle *handle) { delete handle; }
}
