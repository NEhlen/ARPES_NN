from typing import Any, Callable
from numpy.typing import NDArray

import numpy as np

DispersionResult = float | NDArray[np.floating]
DispersionFn = Callable[[NDArray[np.floating], dict[str, Any]], DispersionResult]


class Dispersion:
    def __init__(
        self, params: dict[str, Any], disp_function: DispersionFn | None = None
    ) -> None:
        self.params = params
        if disp_function is None:
            self.disp: DispersionFn = lambda k, _: float(np.sqrt((k**2).sum()))
        else:
            self.disp = disp_function

    def relation(self, k: NDArray[np.floating]) -> DispersionResult:
        return self.disp(k, self.params)


if __name__ == "__main__":

    def Hamiltonian(k, parameters):
        tij = parameters["t"]
        onsite = parameters["on-site"]
        delta_k = (
            np.exp(1j * np.dot(k, parameters["del0"]))
            + np.exp(1j * np.dot(k, parameters["del1"]))
            + np.exp(1j * np.dot(k, parameters["del2"]))
        )
        H = -tij * np.array([[onsite, delta_k], [delta_k.conj(), -onsite]])
        return np.linalg.eigvalsh(H)

    Dsp = Dispersion(
        {
            "t": 1.0,
            "del0": np.array([0.5, 0.5 * np.sqrt(3)]),
            "del1": np.array([0.5, -0.5 * np.sqrt(3)]),
            "del2": np.array([-1, 0]),
            "on-site": 0.2 + 0j,
        },
        Hamiltonian,
    )
    print(Dsp.relation(np.array([0, 2.44])))
