import inspect
import json
import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from dispersion_relations import Dispersion
from dotenv import load_dotenv
from physics_parameters import _kb
from spectral_function import SelfEnergy, SpectralFunction

load_dotenv()


class SpectrumGenerator:
    def __init__(
        self,
        dispersion_relation: Callable,
        parameters: dict,
    ):
        self.parameters = parameters
        self.Dispersion = Dispersion(
            params=parameters["dispersion_params"],
            disp_function=dispersion_relation,
        )
        self.Self_Energy = SelfEnergy(
            _type="self-energy",
            params=parameters["selfenergy_params"],
        )
        self.Spectral_Function = SpectralFunction(
            self.Dispersion,
            self.Self_Energy,
        )
        self.Spectrum = self.Spectral_Function.generate_base(
            **parameters["spectrum_params"]
        )

    def apply(self):
        simulation_params = self.parameters["simulation_data"]
        # add broadening due to energy resolution of analyzer
        self.Spectrum.add_energy_resolution(simulation_params["energy_resolution"])
        # add broadening due to temperature
        self.Spectrum.add_energy_resolution(_kb * simulation_params["temperature"])
        # Multiply with Fermi function
        self.Spectrum.add_Fermi(simulation_params["temperature"])
        # add poisson noise according to noise level
        self.Spectrum.add_poisson_noise(simulation_params["noise_level"])

    # save spectrum
    def save_spectra(self, savestring: str):
        os.makedirs("/".join(savestring.split("/")[:-1]), exist_ok=True)
        np.savetxt(savestring + "_target.txt", self.Spectrum.original)
        np.savetxt(savestring + "_input.txt", self.Spectrum.spectrum)
        with open(savestring + "_parameters.json", "w") as f:
            temp_params = self.parameters
            temp_params["dispersion_relation"] = inspect.getsource(self.Dispersion.disp)
            json.dump(self.parameters, f)

    # plot the spectrum before and after applying broadening and poisson noise
    def plot_spectrum_before_after(self, savestring: str | None = None):
        fig, axarr = plt.subplots(1, 2)
        axarr[0].imshow(
            self.Spectrum.original[::-1, :],
            extent=[np.linalg.norm(k) for k in self.Spectrum.kshape]
            + list(self.Spectrum.Eshape),
        )
        axarr[1].imshow(
            self.Spectrum.spectrum[::-1, :],
            extent=[np.linalg.norm(k) for k in self.Spectrum.kshape]
            + list(self.Spectrum.Eshape),
        )
        if savestring is not None:
            fig.savefig(savestring)


class SpectrumGeneratorBareband:
    def __init__(
        self,
        dispersion_relation: Callable,
        parameters: dict,
    ) -> None:
        self.parameters = parameters
        self.Dispersion = Dispersion(
            params=parameters["dispersion_params"],
            disp_function=dispersion_relation,
        )
        self.Self_Energy_kink = SelfEnergy(
            _type="self-energy",
            params=parameters["selfenergy_params_kink"],
        )
        self.Self_Energy_bare = SelfEnergy(
            _type="self-energy",
            params=parameters["selfenergy_params_bare"],
        )
        self.Spectral_Function_kink = SpectralFunction(
            self.Dispersion,
            self.Self_Energy_kink,
        )
        self.Spectral_Function_bare = SpectralFunction(
            self.Dispersion,
            self.Self_Energy_bare,
        )
        self.Spectrum_kink = self.Spectral_Function_kink.generate_base(
            **parameters["spectrum_params"]
        )
        self.Spectrum_bare = self.Spectral_Function_bare.generate_base(
            **parameters["spectrum_params"]
        )

    def apply(self) -> None:
        simulation_params = self.parameters["simulation_data"]
        # add broadening due to energy resolution of analyzer
        self.Spectrum_kink.add_energy_resolution(simulation_params["energy_resolution"])
        # self.Spectrum_bare.add_energy_resolution(simulation_params["energy_resolution"])
        # add broadening due to temperature
        self.Spectrum_kink.add_energy_resolution(_kb * simulation_params["temperature"])
        # self.Spectrum_bare.add_energy_resolution(_kb * simulation_params["temperature"])
        # Multiply with Fermi function
        self.Spectrum_kink.add_Fermi(simulation_params["temperature"])
        self.Spectrum_bare.add_Fermi(simulation_params["temperature"])
        # add poisson noise according to noise level
        self.Spectrum_kink.add_poisson_noise(simulation_params["noise_level"])
        # self.Spectrum_bare.add_poisson_noise(simulation_params["noise_level"])

    # save spectrum
    def save_spectra(self, savestring: str) -> None:
        os.makedirs("/".join(savestring.split("/")[:-1]), exist_ok=True)
        np.savetxt(savestring + "_target.txt", self.Spectrum_bare.spectrum)
        np.savetxt(savestring + "_input.txt", self.Spectrum_kink.spectrum)
        with open(savestring + "_parameters.json", "w") as f:
            temp_params = self.parameters
            temp_params["dispersion_relation"] = inspect.getsource(self.Dispersion.disp)
            json.dump(self.parameters, f)

    # plot the spectrum before and after applying broadening and poisson noise
    def plot_spectrum_before_after(self, savestring: str | None = None) -> None:
        fig, axarr = plt.subplots(1, 2)
        axarr[0].imshow(
            self.Spectrum_kink.spectrum[::-1, :],
            extent=[np.linalg.norm(k) for k in self.Spectrum_kink.kshape]
            + list(self.Spectrum_kink.Eshape),
        )
        axarr[1].imshow(
            self.Spectrum_bare.spectrum[::-1, :],
            extent=[np.linalg.norm(k) for k in self.Spectrum_bare.kshape]
            + list(self.Spectrum_bare.Eshape),
        )
        if savestring is not None:
            fig.savefig(savestring)
            plt.close(fig)


def param_randomizer(params: dict, change_params: dict):
    for param_type, param_dict in change_params.items():
        if param_type in params:
            for key, value in param_dict.items():
                # if it's a list that means we'll have to interpolate the vectors
                if param_type == "selfenergy_params_kink":
                    if (key == "CC_List") or (key == "Omega_List"):
                        temp_list = []
                        for i in range(len(value)):
                            temp_list.append(
                                np.random.uniform(low=value[i][0], high=value[i][1])
                            )
                        params[param_type][key] = temp_list
                    if (key == "ai") or (key == "bi") or (key == "ar"):
                        params[param_type][key] = np.random.uniform(
                            low=value[0], high=value[0]
                        )
                        params["selfenergy_params_bare"][key] = params[param_type][key]
                elif param_type == "selfenergy_params":
                    if (key == "CC_List") or (key == "Omega_List"):
                        temp_list = []
                        for i in range(len(value)):
                            temp_list.append(
                                np.random.uniform(low=value[i][0], high=value[i][1])
                            )
                        params[param_type][key] = temp_list
                    else:
                        params[param_type][key] = np.random.uniform(
                            low=value[0], high=value[1]
                        )
                else:
                    if type(value[0]) == list:
                        params[param_type][key] = list(
                            np.random.uniform(low=value[0], high=value[1])
                        )
                    else:
                        params[param_type][key] = np.random.uniform(
                            low=value[0], high=value[1]
                        )
    return params


if __name__ == "__main__":

    def Hamiltonian_graphene(k: np.ndarray, parameters: dict):
        tij = parameters["t"]
        onsite = parameters["on-site"]
        delta_k = (
            np.exp(1j * np.dot(k, np.array(parameters["del0"])))
            + np.exp(1j * np.dot(k, np.array(parameters["del1"])))
            + np.exp(1j * np.dot(k, np.array(parameters["del2"])))
        )
        H = (
            -tij * np.array([[onsite, delta_k], [delta_k.conj(), -onsite]])
            + parameters["energy_shift"]
        )
        return np.linalg.eigvalsh(H)

    params = {
        "dispersion_params": {
            "t": 1.0,
            "del0": [0.5, 0.5 * np.sqrt(3)],
            "del1": [0.5, -0.5 * np.sqrt(3)],
            "del2": [-1, 0],
            "on-site": 0.05,
            "energy_shift": -0.3,
        },
        "selfenergy_params_kink": {
            "ai": 0.001,  # intrinsic broadening
            "bi": 0.02,  # imaginary self-energy ee-coupling
            "ar": 0.01,  # real self-energy ee-coupling param
            "CC_List": [0.05],  # eph-coupling params
            "Omega_List": [0.18],  # phonon energies
        },
        "selfenergy_params_bare": {
            "ai": 0.001,  # intrinsic broadening
            "bi": 0.02,  # imaginary self-energy ee-coupling
            "ar": 0.01,  # real self-energy ee-coupling param
        },
        "spectrum_params": {
            "kmin": [0.0, 2.2],
            "kmax": [0.0, 2.7],
            "Emin": -0.3,
            "Emax": 0.1,
            "shape": (256, 256),
        },
        "simulation_data": {
            "energy_resolution": 0.005,  # eV
            "temperature": 30,  # K
            "noise_level": 0.5,  # percentage
        },
    }
    change_params = {
        "dispersion_params": {
            "t": [0.8, 1.2],
            "on-site": [0.0, 0.05],
            "energy_shift": [0.5, -0.5],
        },
        "selfenergy_params_kink": {
            "ai": [0.001, 0.005],  # intrinsic broadening
            "bi": [0.01, 0.03],  # imaginary self-energy ee-coupling
            "ar": [0.01, 0.03],  # real self-energy ee-coupling param
            "CC_List": [[0.04, 0.2]],  # eph-coupling params
            "Omega_List": [[0.1, 0.3]],  # phonon energies
        },
        "spectrum_params": {
            "kmin": [[0.0, 2.0], [0.0, 2.4]],
            "kmax": [[0.0, 3.0], [0.0, 2.5]],
            "Emin": [-0.6, -0.3],
            "Emax": [0.05, 0.1],
        },
        "simulation_data": {
            "temperature": [10, 150],  # K
            "noise_level": [0.1, 0.5],  # percentage
        },
    }
    for i in range(1, 51):
        print(i)
        param_randomizer(params, change_params)
        SG = SpectrumGeneratorBareband(Hamiltonian_graphene, params)
        SG.apply()

        SG.save_spectra(
            r"{}".format(os.environ["DATASET_PATH"]) + f"{i:03d}/graphene_test"
        )
        SG.plot_spectrum_before_after(
            r"{}".format(os.environ["DATASET_PATH"]) + f"{i:03d}/graphene_test_image"
        )
    plt.show()
