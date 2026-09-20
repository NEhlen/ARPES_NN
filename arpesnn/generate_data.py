import inspect
import json
import os
from copy import deepcopy
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from dispersion_relations import Dispersion
from dotenv import load_dotenv
from physics_parameters import _kb
from spectral_function import SelfEnergy, SpectralFunction

load_dotenv()


def parabolic_band(k: np.ndarray, parameters: dict) -> np.ndarray:
    q = k[0] - parameters["k0"]
    bands = [
        parameters["energy_shift"] + curvature * q**2 + offset
        for curvature, offset in zip(parameters["curvatures"], parameters["offsets"])
    ]
    return np.array(bands)


def graphene_cone(k: np.ndarray, parameters: dict) -> np.ndarray:
    q = k[0] - parameters["k0"]
    gap = parameters["gap"]
    energy = np.sqrt((parameters["velocity"] * q) ** 2 + (0.5 * gap) ** 2)
    return np.array(
        [
            parameters["energy_shift"] - energy,
            parameters["energy_shift"] + energy,
        ]
    )


def bilayer_graphene(k: np.ndarray, parameters: dict) -> np.ndarray:
    q = k[0] - parameters["k0"]
    gap = parameters["gap"]
    split = parameters["split"]
    curvature = parameters["curvature"]
    base = np.sqrt((curvature * q**2) ** 2 + (0.5 * gap) ** 2)
    return np.array(
        [
            parameters["energy_shift"] - base - 0.5 * split,
            parameters["energy_shift"] - base + 0.5 * split,
            parameters["energy_shift"] + base - 0.5 * split,
            parameters["energy_shift"] + base + 0.5 * split,
        ]
    )


def mexican_hat_band(k: np.ndarray, parameters: dict) -> np.ndarray:
    q = k[0] - parameters["k0"]
    band = (
        parameters["energy_shift"]
        + parameters["depth"]
        + parameters["quartic"] * (q**2 - parameters["radius"] ** 2) ** 2
    )
    return np.array([band])


def mos2_valence(k: np.ndarray, parameters: dict) -> np.ndarray:
    q = k[0] - parameters["k0"]
    top = parameters["energy_shift"] - parameters["curvature"] * q**2
    return np.array([top, top - parameters["spin_split"]])


BAND_FAMILIES = {
    "parabolic": parabolic_band,
    "four_layer_cs": parabolic_band,
    "graphene": graphene_cone,
    "bilayer_graphene": bilayer_graphene,
    "mexican_hat": mexican_hat_band,
    "mos2": mos2_valence,
}


def random_energy_window() -> tuple[float, float]:
    windows = [
        (-0.35, 0.15),
        (-0.7, 0.3),
        (-1.0, 1.0),
        (-0.5, 2.5),
        (-1.0, 3.0),
        (0.0, 5.0),
    ]
    emin, emax = windows[np.random.randint(len(windows))]
    jitter = 0.1 * (emax - emin)
    return (
        float(emin + np.random.uniform(-jitter, jitter)),
        float(emax + np.random.uniform(-jitter, jitter)),
    )


def random_band_parameters() -> tuple[str, Callable, dict]:
    family = str(np.random.choice(list(BAND_FAMILIES)))
    k0 = float(np.random.uniform(-0.2, 0.2))

    if family == "parabolic":
        params = {
            "k0": k0,
            "energy_shift": float(np.random.uniform(-0.4, 2.5)),
            "curvatures": [
                float(np.random.choice([-1.0, 1.0]) * np.random.uniform(0.4, 4.0))
            ],
            "offsets": [0.0],
        }
    elif family == "four_layer_cs":
        base = float(np.random.uniform(0.0, 4.0))
        spacing = float(np.random.uniform(0.15, 0.8))
        params = {
            "k0": k0,
            "energy_shift": base,
            "curvatures": [
                float(np.random.choice([-1.0, 1.0]) * np.random.uniform(0.15, 1.4))
                for _ in range(4)
            ],
            "offsets": [float((i - 1.5) * spacing) for i in range(4)],
        }
    elif family == "graphene":
        params = {
            "k0": k0,
            "energy_shift": float(np.random.uniform(-0.3, 2.5)),
            "velocity": float(np.random.uniform(1.5, 7.0)),
            "gap": float(np.random.uniform(0.0, 0.25)),
        }
    elif family == "bilayer_graphene":
        params = {
            "k0": k0,
            "energy_shift": float(np.random.uniform(-0.3, 2.5)),
            "curvature": float(np.random.uniform(1.0, 6.0)),
            "gap": float(np.random.uniform(0.0, 0.35)),
            "split": float(np.random.uniform(0.0, 0.35)),
        }
    elif family == "mexican_hat":
        params = {
            "k0": k0,
            "energy_shift": float(np.random.uniform(-0.2, 2.5)),
            "depth": float(np.random.uniform(-0.2, 0.6)),
            "quartic": float(np.random.uniform(2.0, 18.0)),
            "radius": float(np.random.uniform(0.12, 0.55)),
        }
    elif family == "mos2":
        params = {
            "k0": k0,
            "energy_shift": float(np.random.uniform(-0.2, 2.5)),
            "curvature": float(np.random.uniform(0.5, 5.0)),
            "spin_split": float(np.random.uniform(0.08, 0.55)),
        }
    else:
        raise ValueError(f"Unknown band family: {family}")

    return family, BAND_FAMILIES[family], params


def random_self_energy(kink: bool) -> dict:
    params = {
        "ai": float(np.random.uniform(0.001, 0.01)),
        "bi": float(np.random.uniform(0.005, 0.05)),
        "ar": float(np.random.uniform(0.0, 0.04)),
    }
    if kink:
        n_modes = int(np.random.randint(1, 4))
        params["CC_List"] = [
            float(np.random.uniform(0.02, 0.18)) for _ in range(n_modes)
        ]
        params["Omega_List"] = [
            float(np.random.uniform(0.04, 0.35)) for _ in range(n_modes)
        ]
    return params


def add_matrix_and_background(spectrum: np.ndarray) -> np.ndarray:
    rows, cols = spectrum.shape
    k = np.linspace(-1.0, 1.0, cols)
    e = np.linspace(-1.0, 1.0, rows)
    k_center = np.random.uniform(-0.5, 0.5)
    e_center = np.random.uniform(-0.5, 0.5)
    k_width = np.random.uniform(0.35, 1.4)
    e_width = np.random.uniform(0.6, 2.0)
    k_envelope = 0.35 + 0.65 * np.exp(-((k - k_center) ** 2) / (2.0 * k_width**2))
    e_envelope = 0.5 + 0.5 * np.exp(-((e - e_center) ** 2) / (2.0 * e_width**2))
    output = spectrum * e_envelope[:, None] * k_envelope[None, :]
    output += np.random.uniform(0.0, 0.12) * np.mean(output)

    if np.random.random() < 0.45:
        edge_width = np.random.randint(3, 18)
        output[:, :edge_width] = 0.0
    if np.random.random() < 0.45:
        edge_width = np.random.randint(3, 18)
        output[:, -edge_width:] = 0.0

    output -= np.amin(output)
    mean = output.mean()
    if mean > 0.0:
        output /= mean
    return output


def coordinate_channels(spectrum_params: dict) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = spectrum_params["shape"]
    e_vals = np.linspace(
        spectrum_params["Emin"],
        spectrum_params["Emax"],
        rows,
        dtype=np.float32,
    )
    k_vals = np.linspace(
        spectrum_params["kmin"][0],
        spectrum_params["kmax"][0],
        cols,
        dtype=np.float32,
    )
    e_grid = np.repeat(e_vals[:, None], cols, axis=1)
    k_grid = np.repeat(k_vals[None, :], rows, axis=0)
    return e_grid, k_grid


def build_model_input(intensity: np.ndarray, spectrum_params: dict) -> np.ndarray:
    e_grid, k_grid = coordinate_channels(spectrum_params)
    return np.stack([intensity.astype(np.float32), e_grid, k_grid])


def save_training_sample(
    output_prefix: str,
    intensity_input: np.ndarray,
    target: np.ndarray,
    parameters: dict,
    file_format: str,
    coordinate_input: bool,
) -> None:
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    if coordinate_input:
        if file_format != "npy":
            raise ValueError("Coordinate-channel inputs require DATA_FILE_FORMAT=npy.")
        input_data = build_model_input(intensity_input, parameters["spectrum_params"])
    else:
        input_data = intensity_input

    if file_format == "npy":
        np.save(output_prefix + "_input.npy", input_data)
        np.save(output_prefix + "_target.npy", target)
    else:
        np.savetxt(output_prefix + "_input.txt", input_data)
        np.savetxt(output_prefix + "_target.txt", target)

    with open(output_prefix + "_parameters.json", "w") as f:
        json.dump(parameters, f, indent=2)


def generate_diverse_sample() -> tuple[np.ndarray, np.ndarray, dict]:
    family, dispersion_fn, dispersion_params = random_band_parameters()
    emin, emax = random_energy_window()
    k_span = float(np.random.uniform(0.7, 3.2))
    k_center = float(np.random.uniform(-0.2, 0.8))
    spectrum_params = {
        "kmin": [k_center - 0.5 * k_span, 0.0],
        "kmax": [k_center + 0.5 * k_span, 0.0],
        "Emin": emin,
        "Emax": emax,
        "shape": (256, 256),
    }
    simulation_data = {
        "energy_resolution": float(np.random.uniform(0.004, 0.06)),
        "temperature": float(np.random.uniform(10.0, 180.0)),
        "noise_level": float(np.random.uniform(0.05, 0.8)),
    }
    parameters = {
        "family": family,
        "dispersion_params": dispersion_params,
        "selfenergy_params_kink": random_self_energy(kink=True),
        "selfenergy_params_bare": random_self_energy(kink=False),
        "spectrum_params": spectrum_params,
        "simulation_data": simulation_data,
    }

    dispersion = Dispersion(dispersion_params, dispersion_fn)
    spectrum_kink = SpectralFunction(
        dispersion,
        SelfEnergy("self-energy", parameters["selfenergy_params_kink"]),
    ).generate_base(**spectrum_params)
    spectrum_bare = SpectralFunction(
        dispersion,
        SelfEnergy("self-energy", parameters["selfenergy_params_bare"]),
    ).generate_base(**spectrum_params)

    spectrum_kink.add_energy_resolution(simulation_data["energy_resolution"])
    spectrum_kink.add_energy_resolution(_kb * simulation_data["temperature"])
    spectrum_kink.add_Fermi(simulation_data["temperature"])
    spectrum_kink.add_poisson_noise(simulation_data["noise_level"])
    spectrum_kink.spectrum = add_matrix_and_background(spectrum_kink.spectrum)

    spectrum_bare.add_Fermi(simulation_data["temperature"])
    target = deepcopy(spectrum_bare.spectrum)
    target -= target.min()
    if target.mean() > 0.0:
        target /= target.mean()

    return spectrum_kink.spectrum, target, parameters


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
    def save_spectra(self, savestring: str, file_format: str = "txt"):
        os.makedirs("/".join(savestring.split("/")[:-1]), exist_ok=True)
        if file_format == "npy":
            np.save(savestring + "_target.npy", self.Spectrum.original)
            np.save(savestring + "_input.npy", self.Spectrum.spectrum)
        else:
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
    def save_spectra(self, savestring: str, file_format: str = "txt") -> None:
        os.makedirs("/".join(savestring.split("/")[:-1]), exist_ok=True)
        if file_format == "npy":
            np.save(savestring + "_target.npy", self.Spectrum_bare.spectrum)
            np.save(savestring + "_input.npy", self.Spectrum_kink.spectrum)
        else:
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


def get_max_dataset_index(dataset_path: str) -> int:
    max_index = 0
    if not os.path.isdir(dataset_path):
        return max_index
    for entry in os.listdir(dataset_path):
        entry_path = os.path.join(dataset_path, entry)
        if os.path.isdir(entry_path) and entry.isdigit():
            max_index = max(max_index, int(entry))
    return max_index


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


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
    dataset_path = os.environ["DATASET_PATH"]
    n_new_datasets = int(os.getenv("N_NEW_DATASETS", "50"))
    save_plots = env_bool("SAVE_DATA_PLOTS", False)
    file_format = os.getenv("DATA_FILE_FORMAT", "npy").strip().lower()
    generator_mode = os.getenv("DATASET_GENERATOR", "diverse").strip().lower()
    coordinate_input = env_bool("COORDINATE_INPUT", file_format == "npy")
    if file_format not in {"txt", "npy"}:
        raise ValueError("DATA_FILE_FORMAT must be 'txt' or 'npy'.")
    if coordinate_input and file_format != "npy":
        raise ValueError("COORDINATE_INPUT=true requires DATA_FILE_FORMAT=npy.")
    if generator_mode not in {"diverse", "graphene"}:
        raise ValueError("DATASET_GENERATOR must be 'diverse' or 'graphene'.")
    max_existing_index = get_max_dataset_index(dataset_path)
    start_index = max_existing_index + 1

    print(f"Found highest existing dataset index: {max_existing_index:03d}")
    print(f"Generating new datasets from index: {start_index:03d}")
    print(
        f"n_new_datasets={n_new_datasets}, save_plots={save_plots}, "
        f"file_format={file_format}, generator_mode={generator_mode}, "
        f"coordinate_input={coordinate_input}"
    )

    for i in range(1, n_new_datasets + 1):
        current_index = max_existing_index + i
        if generator_mode == "diverse":
            intensity_input, target, sample_params = generate_diverse_sample()
            output_prefix = os.path.join(
                dataset_path,
                f"{current_index:03d}",
                f"{sample_params['family']}_test",
            )
            save_training_sample(
                output_prefix,
                intensity_input,
                target,
                sample_params,
                file_format=file_format,
                coordinate_input=coordinate_input,
            )
            print(f"{current_index:03d} {sample_params['family']}")
        else:
            print(current_index)
            param_randomizer(params, change_params)
            SG = SpectrumGeneratorBareband(Hamiltonian_graphene, params)
            SG.apply()

            output_prefix = os.path.join(
                dataset_path, f"{current_index:03d}", "graphene_test"
            )
            SG.save_spectra(output_prefix, file_format=file_format)
            if save_plots:
                SG.plot_spectrum_before_after(
                    os.path.join(
                        dataset_path, f"{current_index:03d}", "graphene_test_image"
                    )
                )
    if save_plots and generator_mode == "graphene":
        plt.show()
