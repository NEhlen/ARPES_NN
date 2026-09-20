from typing import Sequence, TypedDict
from numpy.typing import NDArray

import numpy as np
from dispersion_relations import Dispersion
from dotenv import load_dotenv
from physics_parameters import _kb
from scipy.ndimage import convolve1d
from scipy.signal.windows import gaussian as scp_gaussian

load_dotenv()


class Spectrum:
    def __init__(
        self,
        image_data: NDArray[np.floating],
        kshape: tuple[NDArray[np.floating], NDArray[np.floating]],
        Eshape: tuple[float, float],
    ) -> None:
        self.original = image_data
        self.spectrum = image_data
        self.kshape = kshape
        self.Eshape = Eshape
        self.shape = self.spectrum.shape
        self.kvals = self.linear_vals_from_min_max(*kshape, N=self.shape[1])
        self.Evals = self.linear_vals_from_min_max(*Eshape, N=self.shape[0])

    # linear intepolation between min and max value
    def linear_vals_from_min_max(
        self,
        min_val: float | NDArray[np.floating],
        max_val: float | NDArray[np.floating],
        N: int,
    ) -> NDArray[np.floating]:
        return np.linspace(min_val, max_val, N)

    # multiply with Fermi function of a given temperature in K
    def add_Fermi(self, Temp: float = 20.0) -> NDArray[np.floating]:
        def fermi(E: NDArray[np.floating], T: float) -> NDArray[np.floating]:
            return 1.0 / (np.exp(np.clip(E / (_kb * T), -700.0, 700.0)) + 1.0)

        fermi_distribution = fermi(self.Evals, Temp)
        self.spectrum = self.spectrum * fermi_distribution[:, None]
        self.original = self.original * fermi_distribution[:, None]
        return fermi_distribution

    # convolve with gaussian window function with given standard deviation (in eV)
    # to simulate instrument resolution
    def add_energy_resolution(self, resolution: float) -> float:
        # recalculate resolution into appropriate pixel std by calculating how many
        # pixels correspond to the resolution
        gaussian_std = resolution * self.shape[0] / (self.Eshape[1] - self.Eshape[0])
        # convolve gaussian along axis
        self.spectrum = convolve1d(
            self.spectrum,
            scp_gaussian(
                51,
                gaussian_std,
            ),
            axis=0,
            mode="nearest",
        )
        return gaussian_std

    # regenerate spectrum with poisson-noise of a given noise level
    def add_poisson_noise(self, noise_lvl: float = 0.1) -> NDArray[np.integer]:
        self.spectrum /= self.spectrum.mean()
        noise_mask = np.random.poisson((self.spectrum + noise_lvl) / noise_lvl)
        self.spectrum = noise_mask.astype(float)
        self.spectrum /= self.spectrum.mean()
        return noise_mask


# Model Class for the many-body complex self-energy
# reimplement this class with other self-energy models to include them in the
# spectra generation
class BaseSelfEnergyParams(TypedDict):
    ai: float
    bi: float
    ar: float


class SelfEnergyParams(BaseSelfEnergyParams, total=False):
    Omega_List: list[float]
    CC_List: list[float]


class SelfEnergy:
    def __init__(self, _type: str, params: SelfEnergyParams) -> None:
        self._type = _type
        self.params = params

    def ImSigma(self, E: float) -> float:
        # ee-interaction
        s = -self.params["ai"] - self.params["bi"] * E**2
        # if there are phonon modes and their coupling constants in the parameters
        # include them in the calculation
        omega_list = self.params.get("Omega_List")
        cc_list = self.params.get("CC_List")
        if omega_list is not None and cc_list is not None:
            for omega, cc in zip(omega_list, cc_list):
                if E <= -omega:
                    s -= np.abs(cc * np.pi * omega * 0.5)
        return s

    # if there are phonon modes and their coupling constants in the parameters
    # include them in the calculation
    def RealSigma(self, E: float) -> float:
        # ee-interaction
        s = self.params["ar"] * E
        # eph-interaction
        omega_list = self.params.get("Omega_List")
        cc_list = self.params.get("CC_List")
        if omega_list is not None and cc_list is not None:
            for omega, cc in zip(omega_list, cc_list):
                s -= (cc * omega * 0.5) * np.log(
                    np.abs((E + omega) / np.sqrt((E - omega) ** 2 + 0.001**2))
                )
        return s

    # if the self-energy class instance is called like a function, it returns the complex self-energy
    def __call__(self, E: float) -> complex:
        return self.RealSigma(E) + 1j * self.ImSigma(E)


# generates the Spectrum of a given dispersion-relation for a given self-energy
class SpectralFunction:
    def __init__(
        self,
        dispersion: Dispersion,
        self_energy: SelfEnergy | float | None = None,
    ) -> None:
        self.dispersion = dispersion

        # if no self-energy is given, use a basic self energy
        if self_energy is None:
            self.self_energy = SelfEnergy(
                _type="simple",
                params={"ai": 0.1, "bi": 0.0, "ar": 0.1},
            )
        # if the self-energy is given as a float, generate a simple self-energy
        # with the float as the intrinsic broadening
        elif isinstance(self_energy, (int, float)):
            self.self_energy = SelfEnergy(
                _type="simple",
                params={"ai": float(self_energy), "bi": 0.0, "ar": 0.0001},
            )
        # if the self energy given can be called and returns a complex number
        # use it as a self-energy. Ideally this is a SelfEnergy class!
        elif isinstance(self_energy, SelfEnergy):
            self.self_energy = self_energy
        else:
            raise TypeError("Self Energy is not a supported type")

    # definition of the spectral function
    def _spectral(
        self,
        k: NDArray[np.floating],
        E: float,
        self_energy: SelfEnergy,
        dispersion: Dispersion,
    ) -> float:
        # load the self-energy
        _SE = self_energy(E)

        # generate spectral function
        A = np.imag(-1.0 / ((E - dispersion.relation(k) - _SE)))
        A /= np.pi
        # if the dispersion relation returns an array with multiple energies (because of multiple bands),
        # the individual components need to be summed
        return float(np.sum(A))

    # generate the basic spectrum in the window given by kmin, kmax, Emin, Emax and the resolution given by shape
    def generate_base(
        self,
        kmin: Sequence[float],
        kmax: Sequence[float],
        Emin: float,
        Emax: float,
        shape: tuple[int, int],
    ) -> Spectrum:
        kmin_arr = np.array(kmin, dtype=float)
        kmax_arr = np.array(kmax, dtype=float)
        nE, nK = shape
        # generate the list of k-values
        dk = kmax_arr - kmin_arr
        dk /= nK
        k_list = np.array([kmin_arr + n * dk for n in range(nK)])
        # generate the list of energy values
        dE = (Emax - Emin) / nE
        E_list = [Emin + n * dE for n in range(nE)]

        # Evaluate the dispersion once per k and self-energy once per energy,
        # instead of repeating both inside a Python loop over all E,k pixels.
        bands = np.stack([np.atleast_1d(self.dispersion.relation(k)) for k in k_list]).T
        energies = np.asarray(E_list)
        sigma = np.asarray([self.self_energy(float(E)) for E in energies])
        gamma = -sigma.imag[:, None]
        center = energies[:, None] - sigma.real[:, None]
        img = np.zeros((nE, nK), dtype=float)
        for band in bands:
            img += gamma / ((center - band[None, :]) ** 2 + gamma**2)
        img /= np.pi
        # put the minimum value in the generated spectrum to 0
        img -= np.amin(img)
        img /= img.mean()
        return Spectrum(
            img,
            (kmin_arr, kmax_arr),
            (Emin, Emax),
        )
