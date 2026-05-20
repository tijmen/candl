from candl.lib import *
import candl.transformations.abstract_base

import os
from pathlib import Path


# --------------------------------------#
# LEAKAGE PRIORS
# --------------------------------------#

ELL_PIVOT = 1000.0

altt2p_par_order = [
    "te_z_90",
    "te_z2_90",
    "te_z_150",
    "te_z2_150",
    "te_z_220",
    "te_z2_220",
    "u_z2_90",
    "u_z2_150",
    "u_z2_220",
]

BAND_TO_IDX = {
    "90": 0,
    "150": 1,
    "220": 2,
    "90GHz": 0,
    "150GHz": 1,
    "220GHz": 2,
}

_LEAKAGE_CACHE = None


def _canonical_band(band):
    band = str(band)
    if band.endswith("GHz"):
        return band[:-3]
    return band


def _default_leakage_file():
    env_path = os.environ.get("ALTT2P_LEAKAGE_FUNCTIONS_FILE")
    if env_path:
        return Path(env_path)

    return (
        Path(__file__).resolve().parents[3]
        / "data"
        / "leakage_functions_cleaned_coherent_u_outer.py"
    )


def _load_leakage_prior_data():
    leakage_file = _default_leakage_file()
    if not leakage_file.is_file():
        raise FileNotFoundError(
            "Could not find altt2p leakage source file. "
            f"Expected {leakage_file}. "
            "Set ALTT2P_LEAKAGE_FUNCTIONS_FILE to override."
        )

    spec = importlib.util.spec_from_file_location(
        f"candl_altt2p_leakage_{abs(hash(str(leakage_file.resolve())))}",
        str(leakage_file),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import leakage source file: {leakage_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "PARAMS_BEST") or not hasattr(module, "COV"):
        raise AttributeError(
            f"Leakage source file {leakage_file} must define PARAMS_BEST and COV."
        )

    params_best = np.asarray(module.PARAMS_BEST, dtype=float)
    cov = np.asarray(module.COV, dtype=float)

    if params_best.shape != (9,):
        raise ValueError(
            f"Expected PARAMS_BEST shape (9,), got {params_best.shape} from {leakage_file}."
        )
    if cov.shape != (9, 9):
        raise ValueError(
            f"Expected COV shape (9, 9), got {cov.shape} from {leakage_file}."
        )

    return params_best, cov, np.linalg.inv(cov)


def _get_leakage_prior_data():
    global _LEAKAGE_CACHE
    if _LEAKAGE_CACHE is None:
        _LEAKAGE_CACHE = _load_leakage_prior_data()
    return _LEAKAGE_CACHE


def _z(ell):
    return jnp.asarray(ell, dtype=jnp.float64) / ELL_PIVOT


def _te_coeffs(params, band):
    idx = BAND_TO_IDX[band]
    return params[2 * idx : 2 * idx + 2]


def _u_coeff(params, band):
    return params[6 + BAND_TO_IDX[band]]


def lambda_TE(ell, band, params=None):
    """Compute lambda_TE^band(ell) = A_band z + B_band z^2."""
    if params is None:
        params = _get_leakage_prior_data()[0]
    params = jnp.asarray(params)

    z = _z(ell)
    a, b = _te_coeffs(params, band)
    return a * z + b * z**2


def u_incoherent(ell, band, params=None):
    """Compute hidden incoherent amplitude u_band(ell) = U_band z^2."""
    if params is None:
        params = _get_leakage_prior_data()[0]
    params = jnp.asarray(params)

    z = _z(ell)
    return _u_coeff(params, band) * z**2


def lambda_EE2(ell, band_a, band_b, params=None):
    """Compute lambda_EE2 = lambda_TE_a lambda_TE_b + u_a u_b."""
    if params is None:
        params = _get_leakage_prior_data()[0]

    te_a = lambda_TE(ell, band_a, params=params)
    te_b = lambda_TE(ell, band_b, params=params)
    u_a = u_incoherent(ell, band_a, params=params)
    u_b = u_incoherent(ell, band_b, params=params)
    return te_a * te_b + u_a * u_b


def log_prior(params):
    """Gaussian log prior: -0.5 * (theta - theta_best)^T C^-1 (theta - theta_best)."""
    params_best, _, cov_inv = _get_leakage_prior_data()
    delta = np.asarray(params) - params_best
    return -0.5 * delta @ cov_inv @ delta


class altt2pT2PLeakage(candl.transformations.abstract_base.Transformation):
    """Alternative T->E leakage model using the cleaned 9-parameter form."""

    def __init__(
        self,
        ells,
        spec_freqs,
        spec_types,
        spec_order,
        long_ells=None,
        data_set_dict=None,
        modes_params=(
            "beta_1",
            "beta_2",
            "beta_3",
            "beta_4",
            "beta_5",
            "beta_6",
            "beta_7",
            "beta_8",
            "beta_9",
        ),
        pol_params=("beta_pol_90", "beta_pol_150", "beta_pol_220"),
        beam_eigenmodes="beams_templates/cov_eigenmodes_300_4100.npz",
        beam_main_temperature="/beams_templates/B_ell_300_4100_main_rc4.npz",
        mode_index=None,
        alpha=0.0,
        beam_renorm=True,
        apply_bp_correction=True,
        descriptor="T2P leakage (altt2p)",
        operation_hint="additive",
    ):
        super().__init__(
            ells=ells,
            descriptor=descriptor,
            param_names=list(altt2p_par_order),
            operation_hint=operation_hint,
        )
        self.spec_order = spec_order
        self.spec_freqs = spec_freqs
        self.spec_types = spec_types
        self.ells = ells
        self.long_ells = long_ells
        self.alpha = alpha
        self.apply_bp_correction = apply_bp_correction

        self.freqs = sorted(
            {_canonical_band(freq) for pair in self.spec_freqs for freq in pair}
        )
        self.modes_params = list(modes_params)
        self.pol_params = list(pol_params)
        self._band_to_pol_param = {}
        for freq in self.freqs:
            for par in self.pol_params:
                if freq in par:
                    self._band_to_pol_param[freq] = par
                    break

        if self.apply_bp_correction:
            self._init_bp_model(
                data_set_dict=data_set_dict,
                beam_eigenmodes=beam_eigenmodes,
                beam_main_temperature=beam_main_temperature,
                mode_index=mode_index,
                beam_renorm=beam_renorm,
            )

        self._validate_required_spectra()

    def _init_bp_model(
        self,
        data_set_dict,
        beam_eigenmodes,
        beam_main_temperature,
        mode_index,
        beam_renorm,
    ):
        if data_set_dict is None:
            raise ValueError(
                "altt2pT2PLeakage: data_set_dict is required when apply_bp_correction=True."
            )

        load = np.load(f"{data_set_dict['data_set_path']}{beam_eigenmodes}")
        beam_modes_all = load["modes"]
        beam_ells = load["ell"]
        del load

        ell_mask = np.isin(beam_ells, self.ells)
        beam_modes_all = beam_modes_all[np.tile(ell_mask, 3)]

        if mode_index is None:
            mode_index = list(range(len(self.modes_params)))
        if len(mode_index) != len(self.modes_params):
            raise ValueError(
                "altt2pT2PLeakage: mode_index must match modes_params length."
            )

        mode_matrix = beam_modes_all[:, mode_index]
        n_ell = len(self.ells)
        freq_to_ix = {"90": 0, "150": 1, "220": 2}

        self._mode_templates_by_band = {}
        for freq in self.freqs:
            if freq not in freq_to_ix:
                raise ValueError(
                    f"altt2pT2PLeakage: unsupported frequency '{freq}' for beam correction."
                )
            i0 = freq_to_ix[freq] * n_ell
            i1 = (freq_to_ix[freq] + 1) * n_ell
            self._mode_templates_by_band[freq] = jnp.asarray(
                mode_matrix[i0:i1, :], dtype=jnp.float64
            )

        load = np.load(f"{data_set_dict['data_set_path']}{beam_main_temperature}")
        beam_ells = load["ell"]
        ell_mask = np.isin(beam_ells, self.ells)
        self._beam_main_by_band = {}
        self._beam_main_800_by_band = {}
        for freq in self.freqs:
            if freq not in load:
                raise ValueError(
                    f"altt2pT2PLeakage: missing '{freq}' in beam_main_temperature file."
                )
            beam_main = jnp.asarray(load[freq][ell_mask], dtype=jnp.float64)
            self._beam_main_by_band[freq] = beam_main
            if beam_renorm:
                norm_vals = load[freq][beam_ells == 800]
                if norm_vals.size == 0:
                    raise ValueError(
                        "altt2pT2PLeakage: could not find ell=800 in "
                        f"beam_main_temperature for {freq}."
                    )
                self._beam_main_800_by_band[freq] = jnp.asarray(
                    norm_vals[0], dtype=jnp.float64
                )
            else:
                self._beam_main_800_by_band[freq] = jnp.asarray(1.0, dtype=jnp.float64)
        del load

    def _alpha_value(self, sample_params):
        return sample_params[self.alpha] if isinstance(self.alpha, str) else self.alpha

    def _band_ratio_bt_over_bp(self, sample_params):
        if not self.apply_bp_correction:
            return {freq: jnp.ones_like(self.ells, dtype=jnp.float64) for freq in self.freqs}

        alpha_val = self._alpha_value(sample_params)
        ell_factor = alpha_val * self.ells / 4000.0 + 1.0 - alpha_val

        mode_vals = jnp.asarray(
            [sample_params[p] for p in self.modes_params],
            dtype=jnp.float64,
        )

        ratios = {}
        for freq in self.freqs:
            bt = 1.0 + self._mode_templates_by_band[freq] @ mode_vals
            pol_par = self._band_to_pol_param.get(freq)
            if pol_par is None:
                ratios[freq] = jnp.ones_like(bt, dtype=jnp.float64)
                continue

            beta_pol = sample_params[pol_par]
            bmain = self._beam_main_by_band[freq]
            bmain_800 = self._beam_main_800_by_band[freq]
            bp = (bmain + beta_pol * ell_factor * (bt - bmain)) / (
                bmain_800 + beta_pol * ell_factor * (1.0 - bmain_800)
            )
            ratios[freq] = bt / bp

        return ratios

    def _find_tt_index(self, freq_a, freq_b):
        freq_a = _canonical_band(freq_a)
        freq_b = _canonical_band(freq_b)
        for i, (stype, (f0, f1)) in enumerate(zip(self.spec_types, self.spec_freqs)):
            f0 = _canonical_band(f0)
            f1 = _canonical_band(f1)
            if stype == "TT" and f0 == freq_a and f1 == freq_b:
                return i
        for i, (stype, (f0, f1)) in enumerate(zip(self.spec_types, self.spec_freqs)):
            f0 = _canonical_band(f0)
            f1 = _canonical_band(f1)
            if stype == "TT" and f0 == freq_b and f1 == freq_a:
                return i
        raise ValueError(
            f"altt2pT2PLeakage: missing TT {freq_a}x{freq_b} (or swapped)."
        )

    def _find_te_like_index(self, t_band, e_band):
        t_band = _canonical_band(t_band)
        e_band = _canonical_band(e_band)
        for i, (stype, (f0, f1)) in enumerate(zip(self.spec_types, self.spec_freqs)):
            f0 = _canonical_band(f0)
            f1 = _canonical_band(f1)
            if stype == "TE" and f0 == t_band and f1 == e_band:
                return i
            if stype == "ET" and f0 == e_band and f1 == t_band:
                return i

        swapped = []
        for i, (stype, (f0, f1)) in enumerate(zip(self.spec_types, self.spec_freqs)):
            f0 = _canonical_band(f0)
            f1 = _canonical_band(f1)
            if stype == "TE" and f0 == e_band and f1 == t_band:
                swapped.append(i)
            if stype == "ET" and f0 == t_band and f1 == e_band:
                swapped.append(i)
        if len(swapped) == 1:
            return swapped[0]
        if len(swapped) > 1:
            raise ValueError(
                f"altt2pT2PLeakage: ambiguous TE/ET lookup for T={t_band}, E={e_band}."
            )
        raise ValueError(
            f"altt2pT2PLeakage: need TE/ET spectrum for T={t_band}, E={e_band}."
        )

    def _te_t_and_e_bands(self, spec_index):
        stype = self.spec_types[spec_index]
        f0, f1 = self.spec_freqs[spec_index]
        f0 = _canonical_band(f0)
        f1 = _canonical_band(f1)
        if stype == "TE":
            return f0, f1
        if stype == "ET":
            return f1, f0
        raise ValueError(
            f"altt2pT2PLeakage: requested TE/ET parsing for spectrum type '{stype}'."
        )

    def _validate_required_spectra(self):
        missing = []
        for i, stype in enumerate(self.spec_types):
            if stype in ("TE", "ET"):
                t_freq, e_freq = self._te_t_and_e_bands(i)
                try:
                    self._find_tt_index(e_freq, t_freq)
                except ValueError as err:
                    missing.append(str(err))

            elif stype == "EE":
                freq_1, freq_2 = self.spec_freqs[i]
                freq_1 = _canonical_band(freq_1)
                freq_2 = _canonical_band(freq_2)
                for t_band, e_band in ((freq_1, freq_2), (freq_2, freq_1)):
                    try:
                        self._find_te_like_index(t_band=t_band, e_band=e_band)
                    except ValueError as err:
                        missing.append(str(err))
                try:
                    self._find_tt_index(freq_1, freq_2)
                except ValueError as err:
                    missing.append(str(err))

        if missing:
            unique_missing = list(dict.fromkeys(missing))
            raise ValueError(
                "altt2pT2PLeakage cannot be used with this spectrum subset. "
                "The altt2p equations require TT for TE leakage, and TT plus "
                "both TE directions for EE leakage. Missing dependencies:\n  - "
                + "\n  - ".join(unique_missing)
            )

    def output(self, Dls, sample_params):
        n_ell = len(self.ells)
        full_spec_shift = jnp.zeros(
            len(self.ells) * len(self.spec_order), dtype=jnp.float64
        )
        altt2p_params = jnp.asarray([sample_params[p] for p in altt2p_par_order])
        bt_over_bp = self._band_ratio_bt_over_bp(sample_params)

        for i, _spec in enumerate(self.spec_order):
            spec_shift = 0.0
            stype = self.spec_types[i]

            if stype in ("TE", "ET"):
                t_freq, e_freq = self._te_t_and_e_bands(i)
                tt_ix = self._find_tt_index(e_freq, t_freq)
                tt_spec = Dls[tt_ix * n_ell : (tt_ix + 1) * n_ell]
                spec_shift = (
                    lambda_TE(self.ells, e_freq, altt2p_params)
                    * bt_over_bp[e_freq]
                    * tt_spec
                )

            elif stype == "EE":
                freq_1, freq_2 = self.spec_freqs[i]
                freq_1 = _canonical_band(freq_1)
                freq_2 = _canonical_band(freq_2)

                te_ix = self._find_te_like_index(t_band=freq_1, e_band=freq_2)
                spec_shift += (
                    lambda_TE(self.ells, freq_1, altt2p_params)
                    * bt_over_bp[freq_1]
                    * Dls[n_ell * te_ix : n_ell * (te_ix + 1)]
                )

                te_ix = self._find_te_like_index(t_band=freq_2, e_band=freq_1)
                spec_shift += (
                    lambda_TE(self.ells, freq_2, altt2p_params)
                    * bt_over_bp[freq_2]
                    * Dls[n_ell * te_ix : n_ell * (te_ix + 1)]
                )

                tt_ix = self._find_tt_index(freq_1, freq_2)
                tt_spec = Dls[tt_ix * n_ell : (tt_ix + 1) * n_ell]
                spec_shift += (
                    lambda_EE2(self.ells, freq_1, freq_2, altt2p_params)
                    * bt_over_bp[freq_1]
                    * bt_over_bp[freq_2]
                    * tt_spec
                )

            full_spec_shift = full_spec_shift.at[i * n_ell : (i + 1) * n_ell].set(
                spec_shift
            )

        return full_spec_shift

    def transform(self, Dls, sample_params):
        return Dls + self.output(Dls, sample_params)
