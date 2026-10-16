"""PD calibration.

Discrimination and calibration are different problems. A model can rank
borrowers perfectly and still be useless for provisioning if its predicted PD
levels are wrong, because ECL is a *level* calculation:

    ECL = PD x LGD x EAD

So the scorecard's raw probabilities are recalibrated on the test window and the
recalibration is then frozen and applied out of time - the only honest
sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class Calibrator:
    kind: str
    model: object

    def apply(self, probabilities: np.ndarray) -> np.ndarray:
        values = np.asarray(probabilities, dtype=float)
        if self.kind == "isotonic":
            calibrated = self.model.predict(values)
        elif self.kind == "platt":
            log_odds = np.log(np.clip(values, 1e-9, 1 - 1e-9) / np.clip(1 - values, 1e-9, 1))
            coefficient, intercept = self.model
            calibrated = 1.0 / (1.0 + np.exp(-(coefficient * log_odds + intercept)))
        elif self.kind == "identity":
            calibrated = values
        else:  # pragma: no cover
            raise ValueError(f"Unknown calibrator kind: {self.kind}")
        return np.clip(calibrated, 1e-6, 1 - 1e-6)


def fit_isotonic(probabilities: np.ndarray, target: np.ndarray) -> Calibrator:
    model = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    model.fit(np.asarray(probabilities, dtype=float), np.asarray(target, dtype=float))
    return Calibrator(kind="isotonic", model=model)


def fit_platt(probabilities: np.ndarray, target: np.ndarray) -> Calibrator:
    """Two-parameter Platt scaling fitted by gradient descent.

    Kept dependency-free (no scipy optimiser) so the module is trivially
    testable; the parameter count is two, so convergence is not a concern.
    """
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1 - 1e-9)
    log_odds = np.log(values / (1 - values))
    y = np.asarray(target, dtype=float)

    coefficient, intercept = 1.0, 0.0
    learning_rate = 0.05
    for _ in range(4000):
        linear = coefficient * log_odds + intercept
        predicted = 1.0 / (1.0 + np.exp(-linear))
        error = predicted - y
        coefficient -= learning_rate * np.mean(error * log_odds)
        intercept -= learning_rate * np.mean(error)
    return Calibrator(kind="platt", model=(coefficient, intercept))


def identity_calibrator() -> Calibrator:
    return Calibrator(kind="identity", model=None)
