# Dementia progression model (hazard-based), using basline data from 2023

import random
import math
import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import Counter, defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# General configuration

REPORTING_AGE_BANDS: List[Tuple[int, Optional[int]]] = [
    (35, 49),
    (50, 64),
    (65, 79),
    (80, None),
]

# Hazard reporting age bands for incidence outputs (inclusive upper bounds)
INCIDENCE_AGE_BANDS: List[Tuple[int, Optional[int]]] = [
    (0, 39),
    (40, 44),
    (45, 49),
    (50, 54),
    (55, 59),
    (60, 64),
    (65, 69),
    (70, 74),
    (75, 79),
    (80, 84),
    (85, 89),
    (90, None),
]


def assign_age_to_reporting_band(age: float,
                                 bands: Optional[List[Tuple[int, Optional[int]]]] = None
                                 ) -> Optional[Tuple[int, Optional[int]]]:
    """Return the first reporting age band that contains the provided age."""
    lookup = bands if bands is not None else REPORTING_AGE_BANDS
    for lower, upper in lookup:
        if upper is None:
            if age >= lower:
                return (lower, upper)
        elif lower <= age <= upper:
            return (lower, upper)
    return None


def age_band_key(band: Tuple[int, Optional[int]]) -> str:
    """Stable key for dictionary columns derived from an age band."""
    lower, upper = band
    upper_str = str(upper) if upper is not None else "plus"
    return f"{lower}_{upper_str}"


def age_band_label(band: Tuple[int, Optional[int]]) -> str:
    """Human-readable label for age band legends."""
    lower, upper = band
    if upper is None:
        return f"{lower}+"
    return f"{lower}-{upper}"


def age_band_midpoint(band: Tuple[int, Optional[int]]) -> Optional[float]:
    """Return numeric midpoint for a closed interval; None for open-ended bands."""
    lower, upper = band
    if upper is None:
        return None
    return (lower + upper) / 2.0


def smooth_series(values: List[float], window: int = 3) -> List[float]:
    """Simple centered moving average smoothing."""
    if window <= 1 or not values:
        return values
    series = pd.Series(values, dtype=float)
    smoothed = (
        series.rolling(window=window, center=True, min_periods=1)
        .mean()
        .tolist()
    )
    return smoothed


def _beta_params_from_mean_rel_sd(mean: float, rel_sd: float) -> Optional[Tuple[float, float]]:
    """Return (alpha, beta) for a beta distribution given mean and relative SD."""
    if not (0.0 < mean < 1.0) or rel_sd <= 0.0:
        return None
    variance = (rel_sd * mean) ** 2
    max_variance = mean * (1.0 - mean)
    if variance >= max_variance:
        variance = max_variance * 0.999
    if variance <= 0.0:
        return None
    common = (mean * (1.0 - mean) / variance) - 1.0
    if common <= 0.0:
        return None
    alpha = mean * common
    beta_param = (1.0 - mean) * common
    if alpha <= 0.0 or beta_param <= 0.0:
        return None
    return alpha, beta_param


def _sample_probability_value(value: Any,
                              rel_sd: float,
                              rng: np.random.Generator) -> Any:
    """Sample a beta-distributed value around the provided probability."""
    try:
        base = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(base) or base <= 0.0:
        return value
    epsilon = 1e-6
    clipped = min(max(base, epsilon), 1.0 - epsilon)
    params = _beta_params_from_mean_rel_sd(clipped, rel_sd)
    if params is None:
        return value
    alpha, beta_param = params
    return float(rng.beta(alpha, beta_param))


def _gamma_params_from_mean_rel_sd(mean: float, rel_sd: float) -> Optional[Tuple[float, float]]:
    """Return (shape, scale) for a gamma distribution given mean and relative SD."""
    if mean <= 0.0 or rel_sd <= 0.0:
        return None
    variance = (rel_sd * mean) ** 2
    if variance <= 0.0:
        return None
    shape = (mean ** 2) / variance
    scale = variance / mean
    if shape <= 0.0 or scale <= 0.0:
        return None
    return shape, scale


def _sample_gamma_value(value: Any,
                        rel_sd: float,
                        rng: np.random.Generator) -> Any:
    """Sample a gamma-distributed value around the provided positive mean."""
    try:
        base = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(base) or base <= 0.0:
        return value
    params = _gamma_params_from_mean_rel_sd(base, rel_sd)
    if params is None:
        return value
    shape, scale = params
    return float(rng.gamma(shape, scale))


def _lognormal_params_from_ci(point_estimate: float,
                              lower: float,
                              upper: float) -> Optional[Tuple[float, float]]:
    """Return (mu, sigma) for lognormal given point estimate and symmetric 95% CI."""
    if point_estimate <= 0.0 or lower <= 0.0 or upper <= 0.0:
        return None
    sigma = (math.log(upper) - math.log(lower)) / (2.0 * 1.96)
    if sigma <= 0.0:
        return None
    mu = math.log(point_estimate)
    return mu, sigma


def _sample_lognormal_from_ci(point_estimate: float,
                              lower: float,
                              upper: float,
                              rng: np.random.Generator) -> float:
    """Sample a lognormal value using the supplied CI; fall back to point estimate."""
    params = _lognormal_params_from_ci(point_estimate, lower, upper)
    if params is None:
        return point_estimate
    mu, sigma = params
    return float(rng.lognormal(mean=mu, sigma=sigma))


def _apply_beta_to_mapping(mapping: Dict[Any, Any],
                           rel_sd: float,
                           rng: np.random.Generator) -> None:
    """Recursively sample beta-distributed values for all numeric leaves."""
    for key, val in mapping.items():
        if isinstance(val, dict):
            _apply_beta_to_mapping(val, rel_sd, rng)
        else:
            mapping[key] = _sample_probability_value(val, rel_sd, rng)


def _sample_cost_structure(costs_cfg: Dict[str, Any],
                           rel_sd: float,
                           rng: np.random.Generator) -> None:
    """Apply gamma sampling to nested cost dictionaries."""
    for stage_meta in costs_cfg.values():
        if not isinstance(stage_meta, dict):
            continue
        for setting_meta in stage_meta.values():
            if not isinstance(setting_meta, dict):
                continue
            for payer, amount in setting_meta.items():
                setting_meta[payer] = _sample_gamma_value(amount, rel_sd, rng)


def _sample_risk_factor_prevalence(risk_defs: Dict[str, dict],
                                   rel_sd: float,
                                   rng: np.random.Generator) -> None:
    """Apply beta sampling to each risk-factor prevalence entry."""
    for meta in risk_defs.values():
        prevalence = meta.get('prevalence')
        if not isinstance(prevalence, dict):
            continue
        for sex_key, value in prevalence.items():
            prevalence[sex_key] = _sample_probability_value(value, rel_sd, rng)


def _sample_risk_factor_relative_risks(risk_defs: Dict[str, dict],
                                       rng: np.random.Generator) -> None:
    """Apply lognormal sampling to risk-factor hazard ratios when CI data is available."""
    for risk_name, meta in risk_defs.items():
        ci_lookup = RISK_FACTOR_HR_INTERVALS.get(risk_name, {})
        rr_def = meta.get('relative_risks')
        if not isinstance(rr_def, dict) or not ci_lookup:
            continue
        for transition, sex_map in rr_def.items():
            if not isinstance(sex_map, dict):
                continue
            ci_transition = ci_lookup.get(transition, {})
            if not ci_transition:
                continue
            for sex_key, value in sex_map.items():
                ci_tuple = ci_transition.get(sex_key) or ci_transition.get('all')
                if not ci_tuple:
                    continue
                point, lower, upper = ci_tuple
                sex_map[sex_key] = _sample_lognormal_from_ci(point, lower, upper, rng)


general_config = {
    'number_of_timesteps': 18,
    'population': 33167098,
         
    'time_step_years': 1,

     #(VERIFIED, ONS) # Anchor baseline stage mix ---
    'base_year': 2023,  # t=0 will be this year; t=1 => 2024, etc.

    "open_population": {
        "use": True,                 # set True to enable new entrants
        "entrants_per_year": 800000,
        "fixed_entry_age": None,     # allow age distribution to evolve over time
        # entrant age-bands scale relative to baseline weights by milestone year
        "age_band_multiplier_schedule": {
            2025: {
                (35, 49): 1.03,
                (50, 64): 1.00,
                (65, 69): 1.05,
                (70, 74): 0.99,
                (75, 79): 1.03,
                (80, 84): 1.09,
                (85, 100): 1.04,
            },
            2030: {
                (35, 49): 1.10,
                (50, 64): 0.97,
                (65, 69): 1.21,
                (70, 74): 1.09,
                (75, 79): 0.95,
                (80, 84): 1.37,
                (85, 100): 1.21,
            },
            2035: {
                (35, 49): 1.14,
                (50, 64): 0.97,
                (65, 69): 1.23,
                (70, 74): 1.26,
                (75, 79): 1.06,
                (80, 84): 1.28,
                (85, 100): 1.52,
            },
            2040: {
                (35, 49): 1.15,
                (50, 64): 1.01,
                (65, 69): 1.16,
                (70, 74): 1.29,
                (75, 79): 1.23,
                (80, 84): 1.44,
                (85, 100): 1.62,
            },
        },
        "sex_distribution": None      # e.g. {"female":0.52,"male":0.48}
    },

    'store_individual_survival': False,    # disable huge per-person survival records by default
    'report_paf_to_terminal': False,      # suppress console logging of PAF summary
    'compute_paf_in_main': False,         # skip running the heavy PAF counterfactual unless needed

    # Probabilistic sensitivity analysis configuration (sampling specs defined below)
    'psa': {
        'use': True,
        'iterations': 1000,
        'seed': 20231113,
        'relative_sd_beta': 0.10,   # +/-10% relative SD for beta-distributed parameters
        'relative_sd_gamma': 0.10,  # +/-10% relative SD for gamma-distributed parameters
    },

    # Optional: override baseline (time step 0) summary metrics with known real-world data
    'initial_summary_overrides': {
        'deaths': 750000,
        'entrants': 800000,
    },

    'initial_stage_mix': {     #(VERIFIED, (Primary Care Dementia Data & Economic Impact Of Dementia CF Report))    # proportions by sex; each nested dict must sum to 1,
        'female': {
            'cognitively_normal': 0.98,
            'mild': 0.00994,
            'moderate': 0.000745,
            'severe': 0.00261,
        },
        'male': {
            'cognitively_normal': 0.98,
            'mild': 0.00994,
            'moderate': 0.000745,
            'severe': 0.00261,
        },
    },
    'initial_dementia_prevalence_by_age_band': {
        (35, 49): {
            'female': 0.000122881250,
            'male': 0.000359039314,
        },
        (50, 64): {
            'female': 0.001724867141,
            'male': 0.001960687970,
        },
        (65, 79): {
            'female': 0.025045935225,
            'male': 0.023442322769,
        },
        (80, 100): {
            'female': 0.177376040863,
            'male': 0.128446668155,
        },
    },

    # (VERIFIED, ONS)  #  Sex mix used when initialising the cohort 
    'sex_distribution': {
        'female': 0.52,
        'male': 0.48,
    },

    # (VERIFIED, ONS)  #  Example: band weights (uniform draw within each band)
    'initial_age_band_weights': {
        (35, 49): 0.34,
        (50, 64): 0.34,
        (65, 79): 0.24,
        (80, 100): 0.09,
    },

    # #(VERIFED, NHS England)  #   Baseline annual probability of onset if no duration is provided for normal->mild
    'base_onset_probability': 0.0025,

    # Optional macro incidence growth (compounds onset hazard per calendar year)
    'incidence_growth': {
        'use': True,
        'annual_rate': 0.02,
        'reference_year': 2023,
    },

    ## (VERIFIED, Tariot et al(2024)) # Mean durations (years) => baseline hazards = 1/duration under exponential assumption 
    'stage_transition_durations': {
        # 'normal_to_mild': 6,
        'mild_to_moderate': 2.2,
        'moderate_to_severe': 2,
        'severe_to_death': 4,
    },

    # (VERIFIED, ONS) # Background mortality (annual absolute hazards by age band; replace with ONS by converting annual probability to hazard h = -ln(1-prob)) 
    'background_mortality_hazards': {
        'female': {
            35: 0.000631199,
            36: 0.000622194,
            37: 0.000744277,
            38: 0.000870379,
            39: 0.000943445,
            40: 0.001004504,
            41: 0.00107758,
            42: 0.001199719,
            43: 0.001238767,
            44: 0.001387963,
            45: 0.001474086,
            46: 0.00163133,
            47: 0.00191884,
            48: 0.001990981,
            49: 0.002132272,
            50: 0.002372813,
            51: 0.002537216,
            52: 0.002685603,
            53: 0.002974419,
            54: 0.003176038,
            55: 0.00349209,
            56: 0.003817277,
            57: 0.004150602,
            58: 0.004402678,
            59: 0.004811557,
            60: 0.005274888,
            61: 0.005626801,
            62: 0.006349113,
            63: 0.00701253,
            64: 0.007683442,
            65: 0.008298336,
            66: 0.009275888,
            67: 0.010128117,
            68: 0.011140829,
            69: 0.012055375,
            70: 0.013366941,
            71: 0.014398158,
            72: 0.015880429,
            73: 0.017527717,
            74: 0.019302093,
            75: 0.021814209,
            76: 0.024221998,
            77: 0.027727889,
            78: 0.031303894,
            79: 0.035050142,
            80: 0.040569943,
            81: 0.045992631,
            82: 0.052261131,
            83: 0.058631734,
            84: 0.066224208,
            85: 0.075113709,
            86: 0.086476611,
            87: 0.097566523,
            88: 0.112847361,
            89: 0.127565226,
            90: 0.146074877,
            91: 0.166562387,
            92: 0.190102953,
            93: 0.213685916,
            94: 0.237158808,
            95: 0.265026004,
            96: 0.291911002,
            97: 0.327432419,
            98: 0.364047771,
            99: 0.39358644,
            100: 0.438461552,
        },
        'male': {
            35: 0.001129638,
            36: 0.001270807,
            37: 0.001342901,
            38: 0.001447046,
            39: 0.001617307,
            40: 0.0017325,
            41: 0.001872753,
            42: 0.001971943,
            43: 0.00211724,
            44: 0.002277592,
            45: 0.002506138,
            46: 0.002683598,
            47: 0.002940319,
            48: 0.003267332,
            49: 0.003525206,
            50: 0.003829323,
            51: 0.00406224,
            52: 0.004496092,
            53: 0.004703042,
            54: 0.005099983,
            55: 0.00545284,
            56: 0.005884278,
            57: 0.006429626,
            58: 0.006883638,
            59: 0.007414419,
            60: 0.008167261,
            61: 0.009077072,
            62: 0.009761489,
            63: 0.010590886,
            64: 0.011667805,
            65: 0.012624353,
            66: 0.013983312,
            67: 0.015533015,
            68: 0.017026125,
            69: 0.018353398,
            70: 0.020152709,
            71: 0.022175059,
            72: 0.024319332,
            73: 0.026198195,
            74: 0.02844783,
            75: 0.03174044,
            76: 0.035692465,
            77: 0.039667453,
            78: 0.044963894,
            79: 0.049874302,
            80: 0.057289128,
            81: 0.064013859,
            82: 0.072640588,
            83: 0.07986389,
            84: 0.089929084,
            85: 0.10094583,
            86: 0.113810561,
            87: 0.12878951,
            88: 0.14638855,
            89: 0.164157919,
            90: 0.187110606,
            91: 0.207099203,
            92: 0.235637527,
            93: 0.260684476,
            94: 0.29167403,
            95: 0.323528963,
            96: 0.353340509,
            97: 0.397973242,
            98: 0.414895939,
            99: 0.475423088,
            100: 0.514090949,
        }
    },

    # (VERIFIED, Crowell et al. 2023) # Dementia severity multipliers on background mortality hazard
    'dementia_mortality_multipliers': {
        'cognitively_normal': 1.0,   
        'mild': 1.0,          #(1.49-2.91)
        'moderate': 1.0,      #(1.94-3.35)
        'severe': 1.0,        #(4.36-8.80)  ###
    },

    # Risk factor definitions with prevalence and hazard ratios by transition, meta-analysis and large cohort studies for onset, will be harder to find for progression HRs
    'risk_factors': {
        # Sex-specific constants (no age buckets) until more granular evidence is available.
        'smoking': {
            'prevalence': {
                'female': 0.114, #(VERIFIED, ONS)
                'male': 0.096,    #(VERIFIED, ONS)
            },
            'relative_risks': {
                'onset': {
                    'female': 1.49,
                    'male': 1.34,
                },
                'mild_to_moderate': {
                    'female': 1.00,
                    'male': 1.00,
                },
                'moderate_to_severe': {
                    'female': 1.00,
                    'male': 1.00,
                },
                'severe_to_death': {
                    'female': 1.75,         #(VERIFIED, LITERATURE)
                    'male': 1.75,          #(VERIFIED, LITERATURE)
                },
            },
        },
        'periodontal_disease': {
            'prevalence': {
                'female': 0.50,
                'male': 0.50,
            },
            'relative_risks': {
                'onset': {
                    'female': 1.47,
                    'male': 1.47,
                },
                'mild_to_moderate': {
                    'female': 1.00,
                    'male': 1.00,
                },
                'moderate_to_severe': {
                    'female': 1.00,
                    'male': 1.00,
                },
                'severe_to_death': {
                    'female': 1.36,      #(VERIFIED, LITERATURE)
                    'male': 1.36,        #(VERIFIED, LITERATURE)
                },
            },
        },
        'cerebrovascular_disease': {
            'prevalence': {
                'female': 0.018,  #(VERIFIED, British Heart Foundation)
                'male': 0.018,    #(VERIFIED, British Heart Foundation)
            },
            'relative_risks': {
                'onset': {
                    'female': 2.40,
                    'male': 2.24,
                },
                'mild_to_moderate': {
                    'female': 1.00,
                    'male': 1.00,
                },
                'moderate_to_severe': {
                    'female': 1.00,
                    'male': 1.00,
                },
                'severe_to_death': {
                    'female': 1.18,      #(VERIFIED, LITERATURE)
                    'male': 1.18,        #(VERIFIED, LITERATURE)
                },
            },
        },
        'CVD_disease': {
            'prevalence': {
                'female': 0.003,   #(VERIFIED, British Heart Foundation)
                'male': 0.003,     #(VERIFIED, British Heart Foundation)
            },
            'relative_risks': {
                'onset': {
                    'female': 2.10,
                    'male': 2.14,
                },
                'mild_to_moderate': {
                    'female': 1.00,
                    'male': 1.00,
                },
                'moderate_to_severe': {
                    'female': 1.00,
                    'male': 1.00,
                },
                'severe_to_death': {
                    'female': 1.34,    #(VERIFIED, LITERATURE)
                    'male': 1.34,     #(VERIFIED, LITERATURE)
                },
            },
        },
        'diabetes': {
            'prevalence': {
                'female': 0.116,   #(VERIFIED, DoHSC)
                'male': 0.116,     #(VERIFIED, DoHSC)
            },
            'relative_risks': {
                'onset': {
                    'female': 1.36,
                    'male': 1.48,
                },
                'mild_to_moderate': {
                    'female': 1.00,
                    'male': 1.00,
                },
                'moderate_to_severe': {
                    'female': 1.00,
                    'male': 1.00,
                },
                'severe_to_death': {
                    'female': 1.79,  #(VERIFIED, LITERATURE)
                    'male': 1.79,   #(VERIFIED, LITERATURE)
                },
            },
        },
    },

    # Living setting transitions (per-cycle probabilities; one-directional per current setting, fair assumption given nature of disease)
    # keyed by stage, then by (lower_age_inclusive, upper_age_inclusive_or_None) band  (VERIFIED, OHE)
    'living_setting_transition_probabilities': {
        'mild': {
            (35, 65): {'to_institution': 0.0087, 'to_home': 0.469},
            (65, None): {'to_institution': 0.066, 'to_home': 0.105},
        },
        'moderate': {
            (35, 65): {'to_institution': 0.055, 'to_home': 0.28},
            (65, None): {'to_institution': 0.143, 'to_home': 0.225},
        },
        'severe': {
            (35, 65): {'to_institution': 0.117, 'to_home': 0.094},
            (65, None): {'to_institution': 0.179, 'to_home': 0.282},
        },
    },

    # Utility norms by age band (baseline utilities, EQ-5D) split by sex
    'utility_norms_by_age': {
        'female': {
            35: 0.91,
            45: 0.85,
            55: 0.81,
            65: 0.78,
            75: 0.71,
        },
        'male': {
            35: 0.91,
            45: 0.84,
            55: 0.78,
            65: 0.78,
            75: 0.75,
        },
    },

    # OFF ## Age hazard ratios (multipliers) for each transition (banded; fallback if parametric off), will need to update with younger multipliers if I turn off Cox-style poarametric age effects
    'age_risk_multipliers': {
        'onset': {
            60: 0.8,
            65: 1.0,  # reference
            70: 1.2,
            75: 1.5,
            80: 1.8,
        },
        'mild_to_moderate': {
            60: 0.8,
            65: 1.0,
            75: 1.2,
            80: 1.35,
        },
        'moderate_to_severe': {
            60: 0.8,
            65: 1.0,
            75: 1.15,
            80: 1.3,
        },
        'severe_to_death': {
            60: 0.8,
            65: 1.0,
            75: 1.1,
            80: 1.2,
        },
    },

    # --- NEW: Parametric (Cox-style) age effects for hazards ---
    'age_hr_parametric': {
        'use': True,      # flip to False to use banded HRs above
        'ref_age': 70,    # baseline age where HR(age)=1
        'betas': {        # per-year log-hazard slopes (increase) (tune to data/literature)
            'onset': 0.06,     #(VERIFIED) 
            'mild_to_moderate': 0.030,   #(VERIFIED, Biondo et al., 2022 - 3% increase in hazard of dementia per year of age = beta = ln(1.03)=0.03)
            'moderate_to_severe': 0.015,    #(VERIFIED, a 10-year age difference gives only 16% higher hazard of progressing to severe, observe slower symptom progression in very old patients - 33% increase over a 10-year age gap corresponds to ln(1.33)/10 = 0.029, which is too high)
            'severe_to_death': 0.010,   #(VERIFIED, an 80-year old with severe dementia has only slightly higher dementia-specific death hazard than a 70-year old with severe dementia, additional mortality so don't need to use such a large effect)
        },
    },

    'discount_rate_annual': 0.035,

    # Utility multipliers by stage/setting (patient & caregiver)
    'utility_multipliers': {
        'patient': {
            'cognitively_normal': {'home': 1.0, 'institution': 0.95},
            'mild': {'home': 0.85, 'institution': 0.80},
            'moderate': {'home': 0.70, 'institution': 0.65},
            'severe': {'home': 0.50, 'institution': 0.45},
        },
        'caregiver': {
            'cognitively_normal': {'home': 1.0, 'institution': 1.0},
            'mild': {'home': 0.90, 'institution': 0.95},
            'moderate': {'home': 0.80, 'institution': 0.90},
            'severe': {'home': 0.70, 'institution': 0.85},
        },
    },
    # Caregiver utility table (stage/setting specific, age-invariant approximation) (VERIFIED, Reed et al., 2017)
    'stage_age_qalys': {
        'caregiver': {
            'cognitively_normal': {'default': {0: 0.91}},
            'mild': {
                'home': {0: 0.86},
            },
            'moderate': {
                'home': {0: 0.85},
            },
            'severe': {
                'home': {0: 0.82},
            },
        },
    },
    # QALY weights for dementia stages (fixed per stage or per sex) (VERIFIED, Mukadam et al., 2024)
    'dementia_stage_qalys': {
        'mild': 0.71,
        'moderate': 0.64,
        'severe': 0.38,
    },

    # Annual costs (GBP) by stage/setting  (VERIFIED, Annual costs of dementia)
    'costs': {
        'cognitively_normal': {
            'home': {'nhs': 0, 'informal': 0},
            'institution': {'nhs': 0, 'informal': 0},
        },
        'mild': {
            'home': {'nhs': 7466.70, 'informal': 10189.55},
            'institution': {'nhs': 23144.27, 'informal': 874.93},
        },
        'moderate': {
            'home': {'nhs': 7180.18, 'informal': 33726.09},
            'institution': {'nhs': 15552.58, 'informal': 1643.14},
        },
        'severe': {
            'home': {'nhs': 7668.60, 'informal': 31523.39},
            'institution': {'nhs': 53084.13, 'informal': 501.88},
        },
    },
}

PSA_DEFAULT_RELATIVE_SD = 0.10

RISK_FACTOR_HR_INTERVALS: Dict[str, Dict[str, Dict[str, Tuple[float, float, float]]]] = {
    'periodontal_disease': {
        'onset': {
            'female': (1.47, 1.32, 1.65),
            'male': (1.47, 1.32, 1.65),
        },
        'severe_to_death': {
            'female': (1.36, 1.10, 1.69),
            'male': (1.36, 1.10, 1.69),
        },
    },
    'smoking': {
        'onset': {
            'female': (1.49, 1.29, 1.72),
            'male': (1.34, 1.19, 1.51),
        },
        'severe_to_death': {
            'female': (1.75, 1.33, 2.29),
            'male': (1.75, 1.33, 2.29),
        },
    },
    'cerebrovascular_disease': {
        'onset': {
            'female': (2.40, 1.91, 3.02),
            'male': (2.24, 1.86, 2.71),
        },
        'severe_to_death': {
            'female': (1.18, 1.12, 1.25),
            'male': (1.18, 1.12, 1.25),
        },
    },
    'CVD_disease': {
        'onset': {
            'female': (2.10, 1.73, 2.54),
            'male': (2.14, 1.85, 2.47),
        },
        'severe_to_death': {
            'female': (1.34, 1.05, 1.71),
            'male': (1.34, 1.05, 1.71),
        },
    },
    'diabetes': {
        'onset': {
            'female': (1.36, 1.02, 1.83),
            'male': (1.48, 1.02, 2.15),
        },
        'severe_to_death': {
            'female': (1.79, 1.56, 2.06),
            'male': (1.79, 1.56, 2.06),
        },
    },
}


# Constants & seed

DEMENTIA_STAGES = ['cognitively_normal', 'mild', 'moderate', 'severe', 'death']
DEMENTIA_DISEASE_STAGES = ('mild', 'moderate', 'severe')  # Active disease stages (excludes normal and death)
LIVING_SETTINGS = ['home', 'institution']
random.seed(42)  # reproducibility

# -------- Weighted age samplers --------

def _normalize_weights(d: dict) -> dict:
    """Return a new dict with values normalized to sum to 1.0 (ignores non-positive weights)."""
    positive = {k: float(v) for k, v in d.items() if float(v) > 0}
    total = sum(positive.values())
    if total <= 0:
        raise ValueError("All provided weights are zero or negative.")
    return {k: v / total for k, v in positive.items()}

def sample_age_from_weighted_ages(age_weights: Dict[int, float]) -> int:
    """
    Sample an exact age from a {age: weight} mapping.
    Example: {40: 0.2, 50: 0.15, 60: 0.25, ...}
    """
    w = _normalize_weights(age_weights)
    ages = list(w.keys())
    probs = list(w.values())
    # Use random.choices for weighted categorical draw
    return random.choices(population=ages, weights=probs, k=1)[0]

def sample_age_from_band_weights(band_weights: Dict[Tuple[int, int], float]) -> int:
    """
    Sample an age band by weight, then draw a uniform integer age within that band (inclusive).
    Example: {(40, 44): 0.2, (45, 49): 0.15, (50, 54): 0.25, ...}
    """
    w = _normalize_weights(band_weights)
    bands = list(w.keys())
    probs = list(w.values())
    low, high = random.choices(population=bands, weights=probs, k=1)[0]
    return random.randint(low, high)  # uniform within chosen band

# Sex and risk-factor helpers

def age_band_weights_for_year(open_pop_cfg: dict,
                              year: int,
                              baseline_weights: Dict[Tuple[int, int], float]) -> Optional[Dict[Tuple[int, int], float]]:
    """
    Return entrant age-band weights for the provided calendar year by scaling baseline
    weights with any milestone multipliers supplied in the open-population configuration.
    """
    multiplier_schedule = open_pop_cfg.get("age_band_multiplier_schedule")
    if not multiplier_schedule:
        # fall back to legacy fixed weights if provided
        schedule = open_pop_cfg.get("age_band_weights_schedule")
        if schedule:
            milestone_years = sorted(schedule)
            if not milestone_years:
                return open_pop_cfg.get("age_band_weights")
            if year <= milestone_years[0]:
                return schedule[milestone_years[0]]
            if year >= milestone_years[-1]:
                return schedule[milestone_years[-1]]
            for start, end in zip(milestone_years, milestone_years[1:]):
                if start <= year <= end:
                    span = end - start
                    weight = 0.0 if span <= 0 else (year - start) / span
                    bands = set(schedule[start]) | set(schedule[end])
                    return {
                        band: ((1.0 - weight) * schedule[start].get(band, 0.0) +
                               weight * schedule[end].get(band, 0.0))
                        for band in bands
                    }
        return open_pop_cfg.get("age_band_weights")

    milestone_years = sorted(multiplier_schedule)
    if not milestone_years:
        return baseline_weights

    if year <= milestone_years[0]:
        multipliers = multiplier_schedule[milestone_years[0]]
    elif year >= milestone_years[-1]:
        multipliers = multiplier_schedule[milestone_years[-1]]
    else:
        multipliers = None
        for start, end in zip(milestone_years, milestone_years[1:]):
            if start <= year <= end:
                span = end - start
                frac = 0.0 if span <= 0 else (year - start) / span
                bands = set(multiplier_schedule[start]) | set(multiplier_schedule[end])
                multipliers = {
                    band: ((1.0 - frac) * multiplier_schedule[start].get(band, 1.0) +
                           frac * multiplier_schedule[end].get(band, 1.0))
                    for band in bands
                }
                break
        if multipliers is None:
            multipliers = multiplier_schedule.get(year, {})

    scaled = {
        band: baseline_weights.get(band, 0.0) * multipliers.get(band, 1.0)
        for band in baseline_weights
    }
    total = sum(scaled.values())
    if total <= 0:
        return baseline_weights
    return {band: value / total for band, value in scaled.items()}

def add_new_entrants(population_state: Dict[int, dict],
                     config: dict,
                     next_id_start: int,
                     calendar_year: int) -> Tuple[int, int]:
    """Optionally add new individuals at the *start* of this timestep.

    Returns the next unused ID and the number of entrants added."""
    op = config.get("open_population", {}) or {}
    if not op.get("use", False):
        return next_id_start, 0

    n_new = int(op.get("entrants_per_year", 0))
    if n_new <= 0:
        return next_id_start, 0

    # fall back to global config if open-pop overrides are not provided
    baseline_weights = config.get("initial_age_band_weights", {})
    age_band_weights = age_band_weights_for_year(op, calendar_year, baseline_weights) or baseline_weights
    sex_dist = op.get("sex_distribution") or config.get("sex_distribution", {})
    base_year = int(config.get('base_year', calendar_year))
    fixed_entry_age = op.get("fixed_entry_age")
    age_sampling_config = {
        "initial_age_band_weights": age_band_weights,
        "initial_age_range": config.get("initial_age_range", (35, 100)),
    }

    entrants_added = 0
    for j in range(n_new):
        if fixed_entry_age is not None:
            age = int(fixed_entry_age)
        else:
            age = sample_age(age_sampling_config)
        sex = sample_sex(sex_dist)
        population_state[next_id_start] = {
            'ID': next_id_start,
            'age': age,
            'sex': sex,
            'risk_factors': assign_risk_factors(config['risk_factors'], age, sex),
            'dementia_stage': 'cognitively_normal',
            'time_in_stage': 0,
            'living_setting': 'home',
            'alive': True,
            'cumulative_qalys_patient': 0.0,
            'cumulative_qalys_caregiver': 0.0,
            'cumulative_costs_nhs': 0.0,
            'cumulative_costs_informal': 0.0,
            'calendar_year': calendar_year,
            'baseline_stage': 'cognitively_normal',
            'entry_age': age,
            'entry_time_step': max(0, calendar_year - base_year),
            'time_since_entry': 0.0,
            'ever_dementia': False,
            'age_at_onset': None,
        }
        next_id_start += 1
        entrants_added += 1

    return next_id_start, entrants_added

def _canonical_sex_label(sex: Optional[str]) -> str:
    """Map free-text sex labels onto a small canonical vocabulary."""
    if sex is None:
        return 'unspecified'
    label = str(sex).strip().lower()
    if not label:
        return 'unspecified'
    if label in {'f', 'female', 'woman', 'women'}:
        return 'female'
    if label in {'m', 'male', 'man', 'men'}:
        return 'male'
    if label in {'all', 'any', 'either', 'both'}:
        return 'all'
    return label


def _resolve_by_sex(mapping: dict, sex: Optional[str], fallbacks: tuple = ('all', 'any', 'either', 'both', 'default')) -> Any:
    """
    Resolve a value from a sex-keyed dictionary with standardized fallback logic.

    Args:
        mapping: Dictionary potentially keyed by sex labels
        sex: The sex to look up
        fallbacks: Tuple of fallback keys to try if exact match fails

    Returns:
        The resolved value, or None if no match found
    """
    if not mapping or not isinstance(mapping, dict):
        return None

    # Try exact sex match
    if sex is not None:
        target = _canonical_sex_label(sex)
        for key, value in mapping.items():
            if _canonical_sex_label(key) == target:
                return value

    # Try fallback keys
    for fallback in fallbacks:
        for key, value in mapping.items():
            if _canonical_sex_label(key) == fallback or key == fallback:
                return value

    # Return first value as last resort (maintains backward compatibility)
    return None

def sample_sex(sex_distribution: Dict[str, float]) -> str:
    """Sample sex from a weight dictionary; defaults to 'unspecified' if absent."""
    if not sex_distribution:
        return 'unspecified'
    normalized = {_canonical_sex_label(k): float(v) for k, v in sex_distribution.items()}
    weights = _normalize_weights(normalized)
    labels = list(weights.keys())
    probs = list(weights.values())
    return random.choices(population=labels, weights=probs, k=1)[0]

def get_stage_mix_for_sex(stage_mix_config: Optional[dict],
                          sex: Optional[str]) -> Optional[Dict[str, float]]:
    """Return the stage-mix weights to use for an individual of the provided sex."""
    if not stage_mix_config or not isinstance(stage_mix_config, dict):
        return None

    # Legacy support: already a mapping of stage -> weight
    if all(isinstance(v, (int, float)) for v in stage_mix_config.values()):
        return stage_mix_config

    canonical_sex = _canonical_sex_label(sex)
    for key, mix in stage_mix_config.items():
        if _canonical_sex_label(key) == canonical_sex and isinstance(mix, dict):
            return mix

    for fallback in ('all', 'any', 'either', 'both', 'default'):
        for key, mix in stage_mix_config.items():
            if (_canonical_sex_label(key) == fallback or key == fallback) and isinstance(mix, dict):
                return mix

    # Final fallback: return the first nested dictionary if present
    for mix in stage_mix_config.values():
        if isinstance(mix, dict):
            return mix

    return None


def get_stage_mix_for_age_and_sex(stage_mix_by_age: Optional[dict],
                                  age: float,
                                  sex: Optional[str]) -> Optional[Dict[str, float]]:
    """Return an age-conditional stage mix; falls back to sex/default keys."""
    if not stage_mix_by_age or not isinstance(stage_mix_by_age, dict):
        return None

    bands: List[Tuple[int, Optional[int]]] = []
    for key in stage_mix_by_age.keys():
        if isinstance(key, tuple) and len(key) == 2:
            lower, upper = key
            if isinstance(lower, (int, float)) and (upper is None or isinstance(upper, (int, float))):
                bands.append((int(lower), None if upper is None else int(upper)))
    bands.sort(key=lambda b: b[0])

    candidate = None
    if bands:
        chosen_band = assign_age_to_reporting_band(age, bands)
        if chosen_band is not None:
            candidate = stage_mix_by_age.get(chosen_band)

    if candidate is None:
        for fallback in ('default', 'all', 'any', 'either', 'both'):
            if fallback in stage_mix_by_age:
                candidate = stage_mix_by_age[fallback]
                break

    if candidate is None:
        return None

    if isinstance(candidate, dict):
        return get_stage_mix_for_sex(candidate, sex)

    return None


def get_dementia_stage_weights_for_sex(stage_mix_config: Optional[dict],
                                       sex: Optional[str]) -> Optional[Dict[str, float]]:
    """Return normalized dementia-stage weights (excluding cognitively normal) for a given sex."""
    mix = get_stage_mix_for_sex(stage_mix_config, sex)
    if not mix or not isinstance(mix, dict):
        return None
    dementia_weights = {
        stage: weight for stage, weight in mix.items()
        if stage in DEMENTIA_DISEASE_STAGES
    }
    if not dementia_weights:
        return None
    try:
        return _normalize_weights(dementia_weights)
    except ValueError:
        return None


def get_dementia_prevalence_for_age_and_sex(prevalence_config: Optional[dict],
                                            age: float,
                                            sex: Optional[str]) -> Optional[float]:
    """Fetch dementia prevalence for the given age/sex from banded configuration."""
    if not prevalence_config or not isinstance(prevalence_config, dict):
        return None

    bands: List[Tuple[int, Optional[int]]] = []
    for key in prevalence_config.keys():
        if isinstance(key, tuple) and len(key) == 2:
            lower, upper = key
            if isinstance(lower, (int, float)) and (upper is None or isinstance(upper, (int, float))):
                bands.append((int(lower), None if upper is None else int(upper)))
    bands.sort(key=lambda b: b[0])

    value = None
    if bands:
        band = assign_age_to_reporting_band(age, bands)
        if band is not None:
            value = prevalence_config.get(band)

    if value is None:
        for fallback in ('default', 'all', 'any', 'either', 'both'):
            if fallback in prevalence_config:
                value = prevalence_config[fallback]
                break

    if isinstance(value, dict):
        sex_label = _canonical_sex_label(sex)
        for key, val in value.items():
            if _canonical_sex_label(key) == sex_label:
                try:
                    prevalence = float(val)
                except (TypeError, ValueError):
                    return None
                return max(0.0, min(1.0, prevalence))
        for fallback in ('default', 'all', 'any', 'either', 'both'):
            for key, val in value.items():
                if _canonical_sex_label(key) == fallback or key == fallback:
                    try:
                        prevalence = float(val)
                    except (TypeError, ValueError):
                        return None
                    return max(0.0, min(1.0, prevalence))
        return None

    if value is None:
        return None

    try:
        prevalence = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, prevalence))

def sample_stage_from_mix(stage_mix: Optional[Dict[str, float]],
                          default_stage: str = 'cognitively_normal') -> str:
    """Sample a dementia stage from weighted mix; fall back to default if weights invalid."""
    if not stage_mix:
        return default_stage
    try:
        weights = _normalize_weights(stage_mix)
    except ValueError:
        return default_stage

    stages = list(weights.keys())
    probs = list(weights.values())
    return random.choices(population=stages, weights=probs, k=1)[0]

def resolve_risk_value(value: Any,
                       age: Optional[int],
                       sex: Optional[str]) -> Any:
    """Recursively resolve nested risk metadata keyed by sex and/or age (scalars allowed)."""
    if isinstance(value, dict):
        sex_keys = [k for k in value.keys() if isinstance(k, str)]
        if sex_keys:
            if sex is not None:
                target = _canonical_sex_label(sex)
                for key in sex_keys:
                    if _canonical_sex_label(key) == target:
                        return resolve_risk_value(value[key], age, None)
            for fallback in ('all', 'any', 'either', 'both', 'default'):
                for key in sex_keys:
                    if _canonical_sex_label(key) == fallback or key == fallback:
                        return resolve_risk_value(value[key], age, None)

        band_items = [(key, nested) for key, nested in value.items()
                      if isinstance(key, tuple) and len(key) == 2]
        if band_items:
            if age is not None:
                for band, nested in band_items:
                    lo, hi = band
                    if lo <= age <= hi:
                        return resolve_risk_value(nested, age, None)
                # fallback: choose band with midpoint closest to age
                closest_nested = min(
                    band_items,
                    key=lambda item: abs((item[0][0] + item[0][1]) / 2.0 - age)
                )[1]
                return resolve_risk_value(closest_nested, age, None)
            return resolve_risk_value(band_items[0][1], age, None)

        numeric_items = [(key, nested) for key, nested in value.items()
                         if isinstance(key, (int, float))]
        if numeric_items:
            numeric_items.sort(key=lambda item: item[0])
            if age is not None:
                chosen = numeric_items[0][1]
                for threshold, nested in numeric_items:
                    if age >= threshold:
                        chosen = nested
                    else:
                        break
                return resolve_risk_value(chosen, age, None)
            return resolve_risk_value(numeric_items[0][1], age, None)

        if 'default' in value:
            return resolve_risk_value(value['default'], age, None)
        if 'all' in value:
            return resolve_risk_value(value['all'], age, None)
    return value

def get_prevalence_for_person(risk_meta: dict, age: int, sex: str) -> float:
    """Fetch prevalence for a given age/sex combination (age ignored if scalar); clamps to [0, 1]."""
    raw = resolve_risk_value(risk_meta.get('prevalence', 0.0), age, sex)
    try:
        prevalence = float(raw)
    except (TypeError, ValueError):
        prevalence = 0.0
    return max(0.0, min(1.0, prevalence))

def get_relative_risk_for_person(risk_meta: dict,
                                 transition: str,
                                 age: int,
                                 sex: str) -> float:
    """Return the relative risk for the given transition, age, and sex (age ignored if scalar)."""
    rr_spec = risk_meta.get('relative_risks', {})
    transition_spec = rr_spec.get(transition, rr_spec.get('default', 1.0))
    raw = resolve_risk_value(transition_spec, age, sex)
    try:
        rr = float(raw)
    except (TypeError, ValueError):
        rr = 1.0
    return max(rr, 0.0)

# Hazard helpers

def prob_to_hazard(p: float, dt: float = 1.0) -> float:
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return float('inf')
    return -math.log(1.0 - p) / dt

def hazard_to_prob(h: float, dt: float = 1.0) -> float:
    if h <= 0.0:
        return 0.0
    return 1.0 - math.exp(-h * dt)

def base_hazard_from_duration(duration_years: float) -> float:
    if duration_years is None or duration_years <= 0.0:
        return 0.0
    return 1.0 / duration_years

def hazards_from_survival(times: List[float],
                          survival_probs: List[float]) -> pd.DataFrame:
    """
    Infer piecewise-constant hazards from survival probabilities at multiple time points.

    Parameters
    ----------
    times:
        Sequence of time points (years). Does not need to start at zero but must be strictly increasing.
    survival_probs:
        Survival probabilities corresponding to ``times`` (same length, positive).

    Returns
    -------
    DataFrame with one row per interval containing the start/end time, survival levels, interval length,
    implied hazard, and the deviation from the mean hazard across all intervals.
    """
    if len(times) != len(survival_probs):
        raise ValueError("times and survival_probs must have the same length.")
    if len(times) < 2:
        raise ValueError("At least two time points are required to infer hazards.")

    t = np.asarray(times, dtype=float)
    s = np.asarray(survival_probs, dtype=float)

    order = np.argsort(t)
    t = t[order]
    s = s[order]

    if np.any(~np.isfinite(t)) or np.any(~np.isfinite(s)):
        raise ValueError("times and survival_probs must contain finite values only.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing.")
    if (s <= 0).any() or (s > 1).any():
        raise ValueError("survival probabilities must be in the interval (0, 1].")

    interval_rows: List[Dict[str, float]] = []
    prev_t = t[0]
    prev_s = s[0]
    for curr_t, curr_s in zip(t[1:], s[1:]):
        dt = curr_t - prev_t
        if dt <= 0:
            raise ValueError("Encountered non-positive interval length when computing hazards.")
        if curr_s <= 0 or prev_s <= 0:
            raise ValueError("Survival probabilities must stay strictly positive.")

        # interval hazard implied by exponential survival over the interval
        hazard = -math.log(curr_s / prev_s) / dt
        interval_rows.append({
            'time_start': float(prev_t),
            'time_end': float(curr_t),
            'interval_length': float(dt),
            'survival_start': float(prev_s),
            'survival_end': float(curr_s),
            'interval_hazard': float(hazard),
        })
        prev_t = curr_t
        prev_s = curr_s

    interval_df = pd.DataFrame(interval_rows)
    if interval_df.empty:
        return interval_df

    mean_hazard = interval_df['interval_hazard'].mean()
    if mean_hazard <= 0:
        interval_df['relative_deviation'] = np.nan
    else:
        interval_df['relative_deviation'] = (
            interval_df['interval_hazard'] - mean_hazard
        ) / mean_hazard
    interval_df['mean_hazard'] = mean_hazard
    return interval_df

def check_constant_hazard(times: List[float],
                          survival_probs: List[float],
                          tolerance: float = 0.05) -> Dict[str, Union[bool, float, pd.DataFrame]]:
    """
    Assess the constant-hazard assumption using survival probabilities across time.

    The function computes piecewise hazards using :func:`hazards_from_survival` and compares each
    interval's implied hazard against the mean hazard. Deviations larger than ``tolerance`` (relative)
    indicate that the exponential/constant-hazard assumption may not hold over the provided horizon.

    Parameters
    ----------
    times:
        Sequence of time points (years) for the survival probabilities.
    survival_probs:
        Survival probabilities observed at ``times``.
    tolerance:
        Maximum allowed relative deviation (default 5%). Set to a higher value if measurement noise is large.

    Returns
    -------
    Dictionary with the inferred mean hazard, the maximum absolute relative deviation, a boolean flag signalling
    whether the assumption holds within tolerance, and the interval DataFrame for further inspection.
    """
    interval_df = hazards_from_survival(times, survival_probs)
    if interval_df.empty:
        return {
            'mean_hazard': float('nan'),
            'max_relative_deviation': float('nan'),
            'within_tolerance': False,
            'intervals': interval_df,
        }

    rel_dev = interval_df['relative_deviation'].abs().replace([np.inf, -np.inf], np.nan).dropna()
    max_dev = rel_dev.max() if not rel_dev.empty else 0.0
    mean_hazard = float(interval_df['mean_hazard'].iloc[0])
    within = bool(max_dev <= tolerance) if np.isfinite(max_dev) else False
    return {
        'mean_hazard': mean_hazard,
        'max_relative_deviation': float(max_dev) if np.isfinite(max_dev) else float('nan'),
        'within_tolerance': within,
        'intervals': interval_df,
    }

def check_constant_hazard_from_model(model_results: dict,
                                     tolerance: float = 0.05,
                                     cohort: str = 'baseline',
                                     use_calendar_year: bool = False) -> Dict[str, Union[bool, float, pd.DataFrame]]:
    """
    Convenience wrapper that pulls survival data from ``model_results`` and runs ``check_constant_hazard``.

    Parameters
    ----------
    model_results:
        Output dictionary returned by :func:`run_model`.
    tolerance:
        Maximum allowed relative deviation (default 5%).
    cohort:
        Either ``'baseline'`` (default) to follow the initial cohort only, or ``'population'`` to include entrants.
    use_calendar_year:
        If ``True`` and calendar years are available, use them as the time axis instead of ``time_step``.
    """
    df = summaries_to_dataframe(model_results)
    if df.empty:
        raise ValueError("Model results do not contain summary history.")

    if cohort == 'population':
        series_name = 'population_alive'
        cohort_key = 'population'
    else:
        series_name = 'baseline_alive'
        cohort_key = 'baseline'

    if series_name not in df.columns:
        raise ValueError(f"Summary dataframe does not include '{series_name}'.")

    alive = df[series_name].fillna(0.0).to_numpy(dtype=float)
    if alive.size == 0 or alive[0] <= 0.0:
        raise ValueError(f"No positive counts found for '{series_name}'.")

    survival = alive / alive[0]
    valid_mask = survival > 0
    if valid_mask.sum() < 2:
        raise ValueError("Need at least two strictly positive survival points to assess hazards.")

    time_key = 'calendar_year' if use_calendar_year and 'calendar_year' in df.columns else 'time_step'
    times = df[time_key].to_numpy(dtype=float)

    times = times[valid_mask]
    survival = survival[valid_mask]

    check = check_constant_hazard(times.tolist(), survival.tolist(), tolerance=tolerance)
    check['cohort'] = cohort_key
    check['time_axis'] = time_key
    return check

def _value_from_age_table(age: float, table: Dict[Union[int, float], float]) -> float:
    thresholds = sorted(table)
    eligible = [a for a in thresholds if a <= age]
    key = eligible[-1] if eligible else thresholds[0]
    return float(table[key])

def get_age_specific_utility(age: float,
                             utility_norms: Union[Dict[Union[int, float], float], Dict[str, Any]],
                             sex: Optional[str] = None) -> float:
    """
    Return age-specific utility. Supports either a flat age->utility map or nested dict keyed by sex.
    Falls back to the first available mapping if sex-specific entry is missing.
    """
    if not utility_norms:
        return 0.0

    if isinstance(utility_norms, dict):
        # Flat age table (legacy)
        if all(isinstance(k, (int, float)) for k in utility_norms.keys()):
            return _value_from_age_table(age, utility_norms)

        sex_key = (sex or '').strip().lower()
        if sex_key and sex_key in utility_norms:
            sex_table = utility_norms[sex_key]
            if isinstance(sex_table, dict) and sex_table:
                return _value_from_age_table(age, sex_table)

        # Try generic 'all' entry
        if 'all' in utility_norms:
            sex_table = utility_norms['all']
            if isinstance(sex_table, dict) and sex_table:
                return _value_from_age_table(age, sex_table)

        # Fallback to first dict-like value
        for value in utility_norms.values():
            if isinstance(value, dict) and value:
                return _value_from_age_table(age, value)

        # Fallback to scalar
        try:
            return float(next(iter(utility_norms.values())))
        except (TypeError, StopIteration, ValueError):
            return 0.0

    # Scalar fallback
    try:
        return float(utility_norms)
    except (TypeError, ValueError):
        return 0.0

def get_stage_age_qaly(subject: str,
                       stage: str,
                       age: float,
                       setting: Optional[str],
                       stage_age_config: Optional[Dict[str, Any]]) -> Optional[float]:
    """
    Return a direct utility weight for the given subject/stage/age/setting if specified.
    Accepts either a direct age-threshold map (e.g. {65: 0.7}) or nested maps keyed by setting
    with optional 'default' fallback. Returns None if no override is configured.
    """
    if not stage_age_config:
        return None
    subject_data = stage_age_config.get(subject)
    if not isinstance(subject_data, dict):
        return None
    stage_data = subject_data.get(stage)
    if not isinstance(stage_data, dict) or not stage_data:
        return None

    def _as_age_map(candidate: Any) -> Optional[Dict[Union[int, float], float]]:
        if not isinstance(candidate, dict) or not candidate:
            return None
        if all(isinstance(k, (int, float)) for k in candidate.keys()):
            return candidate  # direct age map
        return None

    # direct age map at stage level
    age_map = _as_age_map(stage_data)
    if age_map is None:
        selected: Optional[Dict[Union[int, float], float]] = None
        if setting and setting in stage_data:
            selected = _as_age_map(stage_data[setting])
        if selected is None and 'default' in stage_data:
            selected = _as_age_map(stage_data['default'])
        if selected is None:
            # if only one nested dict exists, assume it is the intended age map
            nested_maps = [
                _as_age_map(v) for v in stage_data.values()
                if isinstance(v, dict)
            ]
            nested_maps = [m for m in nested_maps if m is not None]
            if len(nested_maps) == 1:
                selected = nested_maps[0]
        age_map = selected

    if not age_map:
        return None
    return get_age_specific_utility(age, age_map)

def get_qaly_by_age_and_sex(age: float,
                            sex: Optional[str],
                            config: dict) -> float:
    """Return healthy QALY weight using utility_norms_by_age (per sex if provided)."""
    utility_norms = config.get('utility_norms_by_age')
    value = get_age_specific_utility(age, utility_norms, sex)
    if value:
        return value

    fallback = config.get('qalys_by_age')
    if isinstance(fallback, dict) and fallback:
        return get_age_specific_utility(age, fallback)
    return 0.0

def get_dementia_stage_qaly(stage: str,
                            sex: Optional[str],
                            config: dict) -> Optional[float]:
    """Return dementia-stage QALY weight, supporting optional sex-specific entries."""
    stage_map = config.get('dementia_stage_qalys')
    if not isinstance(stage_map, dict) or not stage_map:
        return None

    # direct stage match
    if stage in stage_map and not isinstance(stage_map[stage], dict):
        try:
            return float(stage_map[stage])
        except (TypeError, ValueError):
            return None

    sex_key = _canonical_sex_label(sex)
    if stage in stage_map and isinstance(stage_map[stage], dict):
        entry = stage_map[stage]
        if sex_key and sex_key in entry:
            try:
                return float(entry[sex_key])
            except (TypeError, ValueError):
                return None
        for fallback_key in ('all', 'default'):
            if fallback_key in entry:
                try:
                    return float(entry[fallback_key])
                except (TypeError, ValueError):
                    return None
        for value in entry.values():
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    # sex-first structure: {'female': {'mild': 0.7, ...}}
    if sex_key and sex_key in stage_map:
        sex_entry = stage_map[sex_key]
        if isinstance(sex_entry, dict) and stage in sex_entry:
            try:
                return float(sex_entry[stage])
            except (TypeError, ValueError):
                return None
        default_val = sex_entry.get('default') if isinstance(sex_entry, dict) else None
        if default_val is not None:
            try:
                return float(default_val)
            except (TypeError, ValueError):
                return None

    # generic fallback
    for fallback_key in ('all', 'default'):
        if fallback_key in stage_map:
            entry = stage_map[fallback_key]
            if isinstance(entry, dict) and stage in entry:
                try:
                    return float(entry[stage])
                except (TypeError, ValueError):
                    return None
            try:
                return float(entry)
            except (TypeError, ValueError):
                continue
    return None

def get_caregiver_qaly(age: float,
                       sex: Optional[str],
                       config: dict) -> Optional[float]:
    """Return caregiver QALY weight if caregiver tables are provided."""
    sex_key = _canonical_sex_label(sex)
    tables_by_sex = config.get('caregiver_qalys_by_age_and_sex')
    if isinstance(tables_by_sex, dict) and tables_by_sex:
        sex_table = tables_by_sex.get(sex_key) or tables_by_sex.get('all')
        if isinstance(sex_table, dict) and sex_table:
            return get_age_specific_utility(age, sex_table)

    caregiver_table = config.get('caregiver_qalys_by_age')
    if isinstance(caregiver_table, dict) and caregiver_table:
        return get_age_specific_utility(age, caregiver_table)
    return None

def get_age_multiplier(age: int, age_risk_multipliers: Dict[str, Dict[int, float]], transition: str) -> float:
    transition_multipliers = age_risk_multipliers.get(transition, {})
    if not transition_multipliers:
        return 1.0
    thresholds = sorted(transition_multipliers)
    eligible = [a for a in thresholds if a <= age]
    key = eligible[-1] if eligible else thresholds[0]
    return transition_multipliers[key]

# NEW: unified age HR getter (parametric Cox-style or banded fallback)
def get_age_hr_for_transition(age: int, config: dict, transition: str) -> float:
    """Return age hazard ratio using exp(beta*(age-ref_age)) (optionally two-piece) or banded HRs."""
    param = config.get('age_hr_parametric', {})
    if param.get('use', False):
        ref_age = param.get('ref_age', 65)
        betas = param.get('betas', {}) or {}
        base_beta = betas.get(transition, 0.0)  # default: no age effect if missing
        piecewise = (param.get('two_piece') or {}).get(transition)
        if piecewise and 'break_age' in piecewise:
            break_age = int(piecewise['break_age'])
            beta_before = piecewise.get('beta_before', base_beta)
            beta_after = piecewise.get('beta_after', beta_before)
            if age <= break_age:
                return math.exp(beta_before * (age - ref_age))
            # ensure continuity at the breakpoint so the curve is capped smoothly
            ratio_at_break = math.exp(beta_before * (break_age - ref_age))
            return ratio_at_break * math.exp(beta_after * (age - break_age))
        return math.exp(base_beta * (age - ref_age))
    return get_age_multiplier(age, config.get('age_risk_multipliers', {}), transition)

# ---- NEW helper: per-transition exponent for each risk's RR ----
def _get_rr_weight(config: dict, risk_name: str, transition_key: str) -> float:
    """Return the exponent to apply to a risk factor RR for a given transition."""
    tw = config.get('transition_rr_weights', {}) or {}
    per_risk = tw.get(risk_name, tw.get('default', {})) or {}
    return float(per_risk.get(transition_key, 1.0))

# ---- UPDATED: apply_hazard_ratios now raises RR to a weight and takes config ----
def apply_hazard_ratios(h_base: float,
                        risk_factors: Dict[str, bool],
                        risk_defs: Dict[str, dict],
                        transition_key: str,
                        age_hr: float,
                        age: int,
                        sex: str,
                        config: dict) -> float:
    h = h_base * age_hr
    for factor, active in risk_factors.items():
        if not active:
            continue
        rr = get_relative_risk_for_person(risk_defs.get(factor, {}), transition_key, age, sex)
        w = _get_rr_weight(config, factor, transition_key)
        h *= (rr ** w)
    return h

def transition_hazard_from_config(config: dict, person: dict, transition_key: str) -> float:
    """Return adjusted hazard for a named transition (duration -> h0, then multiply HRs)."""
    duration = config['stage_transition_durations'].get(transition_key)
    h0 = base_hazard_from_duration(duration)
    age_hr = get_age_hr_for_transition(person['age'], config, transition_key)  # CHANGED: parametric or banded
    return apply_hazard_ratios(
        h0,
        person['risk_factors'],
        config['risk_factors'],
        transition_key,
        age_hr,
        person['age'],
        person.get('sex', 'unspecified'),
        config,  # NEW
    )

def transition_prob_from_config(config: dict, person: dict, transition_key: str) -> float:
    """Convenience: hazard -> per-cycle probability."""
    h = transition_hazard_from_config(config, person, transition_key)
    return min(1.0, hazard_to_prob(h, dt=config['time_step_years']))

# Background mortality helpers

def get_background_mortality_hazard(age: int, hazard_table: Dict[int, float]) -> float:
    """Pick the hazard for the closest band <= age (falls back to smallest band if below)."""
    if not hazard_table:
        return 0.0
    thresholds = sorted(hazard_table)
    eligible = [a for a in thresholds if a <= age]
    key = eligible[-1] if eligible else thresholds[0]
    return hazard_table[key]

def get_dementia_mortality_multiplier(stage: str, mults: Dict[str, float]) -> float:
    """Multiply background hazard by a stage-specific factor (optional)."""
    if not mults:
        return 1.0
    return mults.get(stage, 1.0)

# Model storage helpers

def initialize_model_dictionary() -> Dict[int, dict]:
    """Container for per-timestep summary statistics."""
    return {}

def create_time_step_dictionary(model_dictionary: Dict[int, dict], time_step: int = 0, summary: Optional[dict] = None) -> None:
    """Store a per-timestep summary snapshot."""
    model_dictionary[time_step] = summary or {}

# Population init

def sample_age(config: dict) -> int:
    """
    Choose an age according to config:
      - If 'initial_age_weights' is provided: sample exact age by weight.
      - Else if 'initial_age_band_weights' is provided: sample within a band by weight.
      - Else fall back to uniform 'initial_age_range'.
    """
    if 'initial_age_weights' in config and config['initial_age_weights']:
        return sample_age_from_weighted_ages(config['initial_age_weights'])

    if 'initial_age_band_weights' in config and config['initial_age_band_weights']:
        return sample_age_from_band_weights(config['initial_age_band_weights'])

    # fallback: uniform range
    lo, hi = config['initial_age_range']
    return random.randint(lo, hi)

def assign_risk_factors(risk_factors: Dict[str, dict], age: int, sex: str) -> Dict[str, bool]:
    assigned = {}
    for rf, meta in risk_factors.items():
        prevalence = get_prevalence_for_person(meta, age, sex)
        assigned[rf] = random.random() < prevalence
    return assigned

def initialize_population(population: int,
                          config: dict) -> Tuple[Dict[int, dict], Counter]:
    base_year = int(config.get('base_year', 2023))
    stage_mix_config = config.get('initial_stage_mix', None)

    population_state: Dict[int, dict] = {}
    age_counter: Counter = Counter()

    for individual in range(population):
        age = sample_age(config)
        sex = sample_sex(config.get('sex_distribution', {}))

        stage0: Optional[str] = None

        prevalence = get_dementia_prevalence_for_age_and_sex(
            config.get('initial_dementia_prevalence_by_age_band'),
            age,
            sex
        )
        if prevalence is not None:
            if random.random() < prevalence:
                dementia_stage_weights = get_dementia_stage_weights_for_sex(stage_mix_config, sex)
                if dementia_stage_weights:
                    stage0 = sample_stage_from_mix(dementia_stage_weights, default_stage='mild')
                else:
                    stage0 = 'mild'
            else:
                stage0 = 'cognitively_normal'

        if stage0 is None:
            stage_weights = get_stage_mix_for_age_and_sex(
                config.get('initial_stage_mix_by_age_band'),
                age,
                sex
            )
            if stage_weights is None:
                stage_weights = get_stage_mix_for_sex(stage_mix_config, sex)
            stage0 = sample_stage_from_mix(stage_weights)

        population_state[individual] = {
            'ID': individual,
            'age': age,
            'sex': sex,
            'risk_factors': assign_risk_factors(config['risk_factors'], age, sex),
            'dementia_stage': stage0,
            'time_in_stage': 0,
            'living_setting': 'home',
            'alive': True,
            'cumulative_qalys_patient': 0.0,
            'cumulative_qalys_caregiver': 0.0,
            'cumulative_costs_nhs': 0.0,
            'cumulative_costs_informal': 0.0,
            'calendar_year': base_year,
            'baseline_stage': stage0,
            'entry_age': age,
            'entry_time_step': 0,
            'time_since_entry': 0.0,
            'ever_dementia': stage0 in DEMENTIA_DISEASE_STAGES,
            'age_at_onset': age if stage0 in DEMENTIA_DISEASE_STAGES else None,
        }
        age_counter[age] += 1

    return population_state, age_counter

def advance_population_state(population_state: Dict[int, dict],
                             config: dict,
                             calendar_year: int) -> None:
    """Increment age/time_in_stage for alive individuals and roll calendar year."""
    dt = config['time_step_years']
    for person in population_state.values():
        person['calendar_year'] = calendar_year
        if person['alive']:
            person['age'] += dt
            person['time_in_stage'] += dt
            person['time_since_entry'] = person.get('time_since_entry', 0.0) + dt

# Accumulation (QALYs/costs)

def apply_stage_accumulations(individual_data: dict, config: dict, time_step: int) -> None:
    """
    Update cumulative QALYs and costs for the current cycle given stage and living setting.
    Applies NICE discounting at rate config['discount_rate_annual'] to this cycle's flows.
    Discounting is end-of-cycle: factor = 1 / (1 + r) ** (time_step * dt)
    """
    if not individual_data['alive']:
        return

    stage = individual_data['dementia_stage']
    setting = individual_data['living_setting']
    utility_norm = get_age_specific_utility(
        individual_data['age'],
        config['utility_norms_by_age'],
        individual_data.get('sex')
    )

    stage_age_qalys = config.get('stage_age_qalys')
    patient_weight = get_stage_age_qaly(
        'patient', stage, individual_data['age'], setting, stage_age_qalys
    )

    caregiver_weight: float
    if setting == 'home':
        caregiver_override = get_stage_age_qaly(
            'caregiver', stage, individual_data['age'], setting, stage_age_qalys
        )
        if caregiver_override is None:
            caregiver_mult = config['utility_multipliers']['caregiver'].get(stage, {}).get(setting, 0.0)
            caregiver_weight = utility_norm * caregiver_mult
        else:
            caregiver_weight = caregiver_override
    else:
        caregiver_weight = 0.0

    if patient_weight is None:
        patient_mult = config['utility_multipliers']['patient'].get(stage, {}).get(setting, 0.0)
        patient_weight = utility_norm * patient_mult

    costs = config['costs'].get(stage, {}).get(setting, {'nhs': 0.0, 'informal': 0.0})

    dt = config['time_step_years']
    r = float(config.get('discount_rate_annual', 0.0))
    # End-of-cycle discount factor for this period
    disc_factor = 1.0 / ((1.0 + r) ** (time_step * dt))

    # This cycle's (undiscounted) flows
    q_patient = patient_weight * dt
    q_caregiver = caregiver_weight * dt
    c_nhs = costs['nhs'] * dt
    c_informal = costs['informal'] * dt

    # Add discounted flows
    individual_data['cumulative_qalys_patient'] += q_patient * disc_factor
    individual_data['cumulative_qalys_caregiver'] += q_caregiver * disc_factor
    individual_data['cumulative_costs_nhs'] += c_nhs * disc_factor
    individual_data['cumulative_costs_informal'] += c_informal * disc_factor

# Progression with mortality

def update_dementia_progression(population_state: Dict[int, dict],
                                config: dict,
                                time_step: int,
                                death_age_counter: Counter,
                                onset_tracker: Optional[Dict[str, Dict[str, int]]] = None,
                                age_band_exposure: Optional[Dict[Tuple[int, Optional[int]], float]] = None,
                                age_band_onsets: Optional[Dict[Tuple[int, Optional[int]], int]] = None,
                                age_band_exposure_by_sex: Optional[Dict[str, Dict[Tuple[int, Optional[int]], float]]] = None,
                                age_band_onsets_by_sex: Optional[Dict[str, Dict[Tuple[int, Optional[int]], int]]] = None
                                ) -> Tuple[int, int, Dict[Tuple[str, str], int], Dict[str, int]]:
    """Advance dementia stages and apply background/dementia mortality using hazards."""
    dt = config['time_step_years']
    base_year = int(config.get('base_year', 2023))
    calendar_year = base_year + time_step
    growth_cfg = config.get('incidence_growth') or {}
    if growth_cfg.get('use'):
        rate = float(growth_cfg.get('annual_rate', 0.0))
        ref_year = int(growth_cfg.get('reference_year', base_year))
        years_since_ref = max(0, calendar_year - ref_year)
        incidence_growth_multiplier = (1.0 + rate) ** years_since_ref
    else:
        incidence_growth_multiplier = 1.0
    deaths_this_step = 0
    onsets_this_step = 0
    transition_counter: Counter = Counter()
    stage_start_counts: Counter = Counter()
    for person in population_state.values():
        if not person['alive']:
            continue

        stage = person['dementia_stage']
        stage_start_counts[stage] += 1
        incidence_band: Optional[Tuple[int, Optional[int]]] = None
        sex_bucket = person.get('sex', 'unspecified')
        if stage == 'cognitively_normal':
            incidence_band = assign_age_to_reporting_band(person['age'], INCIDENCE_AGE_BANDS)
            if incidence_band is not None:
                if age_band_exposure is not None:
                    age_band_exposure[incidence_band] = age_band_exposure.get(incidence_band, 0.0) + dt
                if age_band_exposure_by_sex is not None:
                    exposure_by_band = age_band_exposure_by_sex.setdefault(sex_bucket, {})
                    exposure_by_band[incidence_band] = exposure_by_band.get(incidence_band, 0.0) + dt

        # --- Mortality step (competing risks if severe) ---
        bg_table_all = config.get('background_mortality_hazards', {})
        if isinstance(bg_table_all, dict) and any(isinstance(v, dict) for v in bg_table_all.values()):
            # nested by sex: choose table by person's sex, or fall back to 'all' or first available
            table = bg_table_all.get(sex_bucket) or bg_table_all.get('all')
            if table is None:
                # fall back to first nested dict if sex not found
                for v in bg_table_all.values():
                    if isinstance(v, dict):
                        table = v
                        break
            h_bg = get_background_mortality_hazard(person['age'], table or {})
        else:
            # flat table (backwards compatible)
            h_bg = get_background_mortality_hazard(person['age'], bg_table_all)

        # stage multiplier
        h_bg *= get_dementia_mortality_multiplier(stage, config.get('dementia_mortality_multipliers', {}))

        if stage == 'severe':
            # Add disease-specific severe->death hazard for competing risks
            h_dem = transition_hazard_from_config(config, person, 'severe_to_death')
            h_total = h_bg + h_dem
        else:
            h_total = h_bg

        p_death = hazard_to_prob(h_total, dt=dt)
        if random.random() < p_death:
            person['dementia_stage'] = 'death'
            person['alive'] = False
            person['living_setting'] = None
            death_age_counter[int(round(person['age']))] += 1
            deaths_this_step += 1
            transition_counter[(stage, 'death')] += 1
            continue  # skip progression if death occurs

        # --- If still alive, apply stage progression (non-death transitions) ---
        if stage == 'cognitively_normal':
            # Onset: either duration-driven (if provided) or base probability converted to hazard
            if 'normal_to_mild' in config['stage_transition_durations']:
                p = transition_prob_from_config(config, person, 'normal_to_mild')
            else:
                h0 = prob_to_hazard(config.get('base_onset_probability', 0.0), dt=dt)
                h0 *= incidence_growth_multiplier
                age_hr = get_age_hr_for_transition(person['age'], config, 'onset')  # CHANGED
                h = apply_hazard_ratios(
                    h0,
                    person['risk_factors'],
                    config['risk_factors'],
                    'onset',
                    age_hr,
                    person['age'],
                    person.get('sex', 'unspecified'),
                    config,  # NEW
                )
                p = hazard_to_prob(h, dt=dt)
            onset_triggered = False
            if random.random() < p:
                person['dementia_stage'] = 'mild'
                person['time_in_stage'] = 0
                if not person.get('ever_dementia', False):
                    person['ever_dementia'] = True
                    person['age_at_onset'] = float(person.get('age', 0.0))
                elif person.get('age_at_onset') is None:
                    person['age_at_onset'] = float(person.get('age', 0.0))
                onsets_this_step += 1
                onset_triggered = True
                if onset_tracker is not None:
                    risk_flags = person.get('risk_factors', {})
                    for risk_name, counts in onset_tracker.items():
                        exposed = bool(risk_flags.get(risk_name, False))
                        bucket = 'with' if exposed else 'without'
                        counts[bucket] = counts.get(bucket, 0) + 1
                if onset_triggered and age_band_onsets is not None and incidence_band is not None:
                    age_band_onsets[incidence_band] = age_band_onsets.get(incidence_band, 0) + 1
                if onset_triggered and age_band_onsets_by_sex is not None and incidence_band is not None:
                    onset_by_band = age_band_onsets_by_sex.setdefault(sex_bucket, {})
                    onset_by_band[incidence_band] = onset_by_band.get(incidence_band, 0) + 1

        elif stage == 'mild':
            p = transition_prob_from_config(config, person, 'mild_to_moderate')
            if random.random() < p:
                person['dementia_stage'] = 'moderate'
                person['time_in_stage'] = 0

        elif stage == 'moderate':
            p = transition_prob_from_config(config, person, 'moderate_to_severe')
            if random.random() < p:
                person['dementia_stage'] = 'severe'
                person['time_in_stage'] = 0

        elif stage == 'severe':
            # death handled above; no non-death transition
            pass

        else:
            continue
        end_stage = person['dementia_stage']
        transition_counter[(stage, end_stage)] += 1

    return deaths_this_step, onsets_this_step, dict(transition_counter), dict(stage_start_counts)

# Living setting transitions

def _select_living_setting_transition(config: Dict, stage: str, age: float) -> Dict[str, float]:
    """Return living setting transition probabilities for a stage/age, keeping backward compatibility."""
    stage_table = config.get('living_setting_transition_probabilities', {}).get(stage, {})
    if not stage_table:
        return {}
    # Legacy structure: flat dict per stage
    if isinstance(stage_table, dict) and (
        'to_institution' in stage_table or 'to_home' in stage_table
    ):
        return stage_table

    for band, probs in stage_table.items():
        if not isinstance(probs, dict):
            continue
        lower: Optional[float]
        upper: Optional[float]
        if isinstance(band, tuple) and len(band) == 2:
            lower, upper = band
        else:
            lower, upper = None, None
        lower = float(lower) if lower is not None else float('-inf')
        upper = float(upper) if upper is not None else float('inf')
        if lower <= age <= upper:
            return probs

    # Fallback in case no band matched (e.g. ages outside configured range)
    first = next(iter(stage_table.values()), {})
    return first if isinstance(first, dict) else {}


def update_living_setting(individual_data: dict, config: dict) -> None:
    if not individual_data['alive']:
        return
    stage = individual_data['dementia_stage']
    if stage in DEMENTIA_DISEASE_STAGES:
        probs = _select_living_setting_transition(config, stage, float(individual_data.get('age', 0.0)))
        current = individual_data['living_setting']
        if current == 'home' and random.random() < probs.get('to_institution', 0.0):
            individual_data['living_setting'] = 'institution'
        elif current == 'institution' and random.random() < probs.get('to_home', 0.0):
            individual_data['living_setting'] = 'home'
    elif stage == 'cognitively_normal':
        individual_data['living_setting'] = 'home'

# Per-cycle accumulation wrapper

def update_stage_accumulations(population_state: Dict[int, dict],
                               time_step: int,
                               config: dict) -> None:
    """Apply living transitions, then add (discounted) QALYs/costs for the cycle."""
    for person in population_state.values():
        update_living_setting(person, config)
        apply_stage_accumulations(person, config, time_step)

# Reporting utils

def summarize_population_state(population_state: Dict[int, dict],
                               time_step: int,
                               base_year: int,
                               entrants: int = 0,
                               deaths: int = 0,
                               new_onsets: int = 0) -> dict:
    """Aggregate key metrics for the current time step."""
    stage_counter = Counter()
    living_counter = Counter()
    age_band_dementia_counter = Counter()

    alive_count = 0
    age_alive_sum = 0.0
    baseline_alive_count = 0
    dementia_age_sum = 0.0
    dementia_count = 0
    total_qalys_patient = 0.0
    total_qalys_caregiver = 0.0
    total_costs_nhs = 0.0
    total_costs_informal = 0.0

    for person in population_state.values():
        stage = person['dementia_stage']
        stage_counter[stage] += 1
        total_qalys_patient += person['cumulative_qalys_patient']
        total_qalys_caregiver += person['cumulative_qalys_caregiver']
        total_costs_nhs += person['cumulative_costs_nhs']
        total_costs_informal += person['cumulative_costs_informal']

        if person['alive']:
            alive_count += 1
            age_alive_sum += person['age']
            living_counter[person.get('living_setting', 'unknown')] += 1
            if stage in DEMENTIA_DISEASE_STAGES:
                dementia_age_sum += person['age']
                dementia_count += 1
                band = assign_age_to_reporting_band(person['age'])
                if band is not None:
                    age_band_dementia_counter[band] += 1
            if person.get('entry_time_step', 0) == 0:
                baseline_alive_count += 1

    mean_age_alive = age_alive_sum / alive_count if alive_count else 0.0
    mean_age_dementia = dementia_age_sum / dementia_count if dementia_count else 0.0

    summary = {
        'time_step': time_step,
        'calendar_year': base_year + time_step,
        'population_total': len(population_state),
        'population_alive': alive_count,
        'baseline_alive': baseline_alive_count,
        'entrants': entrants,
        'deaths': deaths,
        'incident_onsets': new_onsets,
        'incidence_per_1000_alive': (new_onsets / alive_count * 1000.0) if alive_count else 0.0,
        'total_qalys_patient': total_qalys_patient,
        'total_qalys_caregiver': total_qalys_caregiver,
        'total_costs_nhs': total_costs_nhs,
        'total_costs_informal': total_costs_informal,
        'mean_age_alive': mean_age_alive,
        'mean_age_dementia': mean_age_dementia,
    }

    for stage in DEMENTIA_STAGES:
        summary[f'stage_{stage}'] = stage_counter.get(stage, 0)

    for setting in LIVING_SETTINGS:
        summary[f'living_{setting}'] = living_counter.get(setting, 0)
    summary['living_unknown'] = living_counter.get('unknown', 0)

    for band in REPORTING_AGE_BANDS:
        summary[f'ad_cases_age_{age_band_key(band)}'] = age_band_dementia_counter.get(band, 0)

    return summary

def generate_output(summary_history: Dict[int, dict], time_step: int) -> None:
    summary = summary_history.get(time_step)
    if summary is None:
        print(f'Time step {time_step}: no summary available')
        return

    print(f"Time step {time_step} (calendar year {summary.get('calendar_year', 'n/a')})")
    print('Stage counts:')
    for stage in DEMENTIA_STAGES:
        key = f'stage_{stage}'
        if key in summary:
            print(f"  {stage}: {summary[key]}")
    print(f"Alive count: {summary.get('population_alive', 0)}")
    print(f"Entrants this step: {summary.get('entrants', 0)}")
    print(f"Deaths this step: {summary.get('deaths', 0)}")
    if 'mean_age_alive' in summary:
        print(f"Mean age (alive): {summary['mean_age_alive']:.2f}")
    if 'mean_age_dementia' in summary:
        print(f"Mean age (dementia stages): {summary['mean_age_dementia']:.2f}")
    if 'incident_onsets' in summary:
        print(f"New onsets this step: {summary.get('incident_onsets', 0)}")

# Main run

def compute_lifetime_risk_by_entry_age(population_state: Dict[int, dict],
                                       restrict_to_cognitively_normal: bool = True) -> List[dict]:
    """
    Aggregate the lifetime dementia risk by entry age.

    Parameters
    ----------
    population_state: mapping of person ID to their final record.
    restrict_to_cognitively_normal: include only individuals who entered as cognitively normal.

    Returns
    -------
    A list of records sorted by entry age with keys:
        entry_age, population, dementia_cases, lifetime_risk.
    """
    total_by_age: Counter = Counter()
    dementia_by_age: Counter = Counter()

    for person in population_state.values():
        entry_age = person.get('entry_age')
        if entry_age is None:
            entry_age = person.get('age')
        if entry_age is None:
            continue
        try:
            entry_age_int = int(round(float(entry_age)))
        except (TypeError, ValueError):
            continue

        baseline_stage = person.get('baseline_stage', 'cognitively_normal')
        if restrict_to_cognitively_normal and baseline_stage != 'cognitively_normal':
            continue

        total_by_age[entry_age_int] += 1

        ever_dementia = bool(person.get('ever_dementia', False))
        if baseline_stage in DEMENTIA_DISEASE_STAGES:
            ever_dementia = True
        if ever_dementia:
            dementia_by_age[entry_age_int] += 1

    records: List[dict] = []
    for age in sorted(total_by_age.keys()):
        population = total_by_age[age]
        if population <= 0:
            continue
        dementia_cases = dementia_by_age.get(age, 0)
        risk = dementia_cases / population if population else 0.0
        records.append({
            'entry_age': age,
            'population': population,
            'dementia_cases': dementia_cases,
            'lifetime_risk': risk,
        })

    return records

def collect_individual_survival(population_state: Dict[int, dict]) -> List[dict]:
    records: List[dict] = []
    for person in population_state.values():
        baseline_stage = person.get('baseline_stage', person.get('dementia_stage', 'unknown'))
        record = {
            'ID': person.get('ID'),
            'baseline_stage': baseline_stage,
            'time': float(person.get('time_since_entry', 0.0)),
            'event': 0 if person.get('alive', False) else 1,
            'entry_time_step': int(person.get('entry_time_step', 0)),
        }
        records.append(record)
    return records


def summaries_to_dataframe(model_results: dict) -> pd.DataFrame:
    """Convert stored summaries into a tidy dataframe."""
    summaries = model_results.get('summaries', {}) if isinstance(model_results, dict) else {}
    rows = []
    for time_step in sorted(summaries.keys()):
        summary = summaries[time_step].copy()
        rows.append(summary)
    return pd.DataFrame(rows)

def run_model(config: dict, seed: Optional[int] = None) -> dict:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Validate required configuration parameters
    required_keys = ['number_of_timesteps', 'population', 'time_step_years', 'stage_transition_durations']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")

    if config['number_of_timesteps'] <= 0:
        raise ValueError(f"number_of_timesteps must be positive, got {config['number_of_timesteps']}")
    if config['population'] <= 0:
        raise ValueError(f"population must be positive, got {config['population']}")

    number_of_timesteps = config['number_of_timesteps'] + 1
    population = config['population']
    base_year = int(config.get('base_year', 2023))

    summary_history = initialize_model_dictionary()
    population_state, initial_age_counter = initialize_population(population, config)
    death_age_counter: Counter = Counter()
    transition_history: Dict[int, dict] = {}
    risk_onset_tracker: Dict[str, Dict[str, int]] = {
        name: {'with': 0, 'without': 0} for name in config.get('risk_factors', {})
    }
    incidence_age_exposure: Dict[Tuple[int, Optional[int]], float] = defaultdict(float)
    incidence_age_onsets: Dict[Tuple[int, Optional[int]], int] = defaultdict(int)

    baseline_summary = summarize_population_state(population_state, 0, base_year, entrants=0, deaths=0)
    baseline_overrides = config.get('initial_summary_overrides') or {}
    if baseline_overrides:
        baseline_summary.update(baseline_overrides)
        if 'incident_onsets' in baseline_overrides:
            alive = baseline_summary.get('population_alive', 0) or 0
            baseline_summary['incident_onsets'] = baseline_overrides['incident_onsets']
            baseline_summary['incidence_per_1000_alive'] = (
                (baseline_summary['incident_onsets'] / alive) * 1000.0 if alive else 0.0
            )
        if 'deaths' in baseline_overrides:
            baseline_summary['deaths'] = baseline_overrides['deaths']
        if 'entrants' in baseline_overrides:
            baseline_summary['entrants'] = baseline_overrides['entrants']
        if 'calendar_year' in baseline_overrides:
            baseline_summary['calendar_year'] = baseline_overrides['calendar_year']
        if 'time_step' in baseline_overrides:
            baseline_summary['time_step'] = baseline_overrides['time_step']

    create_time_step_dictionary(summary_history, 0, baseline_summary)
    generate_output(summary_history, 0)

    next_id = len(population_state)
    yearly_incidence_records: List[dict] = []

    for time_step in range(1, number_of_timesteps):
        calendar_year = base_year + time_step
        advance_population_state(population_state, config, calendar_year)

        next_id, entrants_this_step = add_new_entrants(population_state, config, next_id, calendar_year)
        per_sex_exposure: Dict[str, Dict[Tuple[int, Optional[int]], float]] = {}
        per_sex_onsets: Dict[str, Dict[Tuple[int, Optional[int]], int]] = {}
        deaths_this_step, onsets_this_step, transition_counts, stage_start_counts = update_dementia_progression(
            population_state,
            config,
            time_step,
            death_age_counter,
            risk_onset_tracker,
            incidence_age_exposure,
            incidence_age_onsets,
            per_sex_exposure,
            per_sex_onsets
        )
        update_stage_accumulations(population_state, time_step, config)

        alive_counts_by_sex_band: Dict[str, Dict[Tuple[int, Optional[int]], int]] = {}
        prevalent_counts_by_sex_band: Dict[str, Dict[Tuple[int, Optional[int]], int]] = {}
        for person in population_state.values():
            if not person.get('alive', False):
                continue
            band = assign_age_to_reporting_band(float(person.get('age', 0.0)), INCIDENCE_AGE_BANDS)
            if band is None:
                continue
            sex_key = person.get('sex', 'unspecified')
            alive_bucket = alive_counts_by_sex_band.setdefault(sex_key, {})
            alive_bucket[band] = alive_bucket.get(band, 0) + 1
            if person.get('dementia_stage') in DEMENTIA_DISEASE_STAGES:
                prevalent_bucket = prevalent_counts_by_sex_band.setdefault(sex_key, {})
                prevalent_bucket[band] = prevalent_bucket.get(band, 0) + 1

        sexes_present = (
            set(per_sex_exposure.keys())
            | set(per_sex_onsets.keys())
            | set(alive_counts_by_sex_band.keys())
            | set(prevalent_counts_by_sex_band.keys())
            | {'female', 'male'}
        )
        sexes_present.discard('all')

        totals_per_band = defaultdict(lambda: {
            'person_years_at_risk': 0.0,
            'incident_onsets_at_risk': 0,
            'population_alive_in_band': 0,
            'prevalent_dementia_cases_in_band': 0,
        })

        for sex in sorted(sexes_present):
            exposure_by_band = per_sex_exposure.get(sex, {})
            onsets_by_band = per_sex_onsets.get(sex, {})
            alive_by_band = alive_counts_by_sex_band.get(sex, {})
            prevalence_by_band = prevalent_counts_by_sex_band.get(sex, {})
            for band in INCIDENCE_AGE_BANDS:
                lower, upper = band
                band_label = age_band_label(band)
                person_years = float(exposure_by_band.get(band, 0.0))
                incident_onsets = int(onsets_by_band.get(band, 0))
                alive_count = int(alive_by_band.get(band, 0))
                prevalent_count = int(prevalence_by_band.get(band, 0))
                record = {
                    'time_step': time_step,
                    'calendar_year': calendar_year,
                    'sex': sex,
                    'age_band': band_label,
                    'age_lower': lower,
                    'age_upper': upper,
                    'person_years_at_risk': person_years,
                    'incident_onsets_at_risk': incident_onsets,
                    'population_alive_in_band': alive_count,
                    'prevalent_dementia_cases_in_band': prevalent_count,
                }
                yearly_incidence_records.append(record)
                totals_metrics = totals_per_band[band]
                totals_metrics['person_years_at_risk'] += person_years
                totals_metrics['incident_onsets_at_risk'] += incident_onsets
                totals_metrics['population_alive_in_band'] += alive_count
                totals_metrics['prevalent_dementia_cases_in_band'] += prevalent_count

        for band in INCIDENCE_AGE_BANDS:
            lower, upper = band
            band_label = age_band_label(band)
            totals_metrics = totals_per_band[band]
            record_all = {
                'time_step': time_step,
                'calendar_year': calendar_year,
                'sex': 'all',
                'age_band': band_label,
                'age_lower': lower,
                'age_upper': upper,
                'person_years_at_risk': float(totals_metrics['person_years_at_risk']),
                'incident_onsets_at_risk': int(totals_metrics['incident_onsets_at_risk']),
                'population_alive_in_band': int(totals_metrics['population_alive_in_band']),
                'prevalent_dementia_cases_in_band': int(totals_metrics['prevalent_dementia_cases_in_band']),
            }
            yearly_incidence_records.append(record_all)

        transition_history[time_step] = {
            'transition_counts': transition_counts,
            'stage_start_counts': stage_start_counts,
        }

        summary = summarize_population_state(population_state,
                                             time_step,
                                             base_year,
                                             entrants=entrants_this_step,
                                             deaths=deaths_this_step,
                                             new_onsets=onsets_this_step)
        create_time_step_dictionary(summary_history, time_step, summary)
        generate_output(summary_history, time_step)

    lifetime_risk_normal = compute_lifetime_risk_by_entry_age(population_state, restrict_to_cognitively_normal=True)
    lifetime_risk_all = compute_lifetime_risk_by_entry_age(population_state, restrict_to_cognitively_normal=False)
    if config.get('store_individual_survival', True):
        survival_records = collect_individual_survival(population_state)
    else:
        survival_records = []

    # Capture final alive/dementia counts by incidence age band
    age_band_alive_counts: Dict[Tuple[int, Optional[int]], int] = defaultdict(int)
    age_band_dementia_counts: Dict[Tuple[int, Optional[int]], int] = defaultdict(int)
    for person in population_state.values():
        if not person.get('alive', False):
            continue
        band = assign_age_to_reporting_band(float(person.get('age', 0.0)), INCIDENCE_AGE_BANDS)
        if band is None:
            continue
        age_band_alive_counts[band] += 1
        if person.get('dementia_stage') in DEMENTIA_DISEASE_STAGES:
            age_band_dementia_counts[band] += 1

    incidence_age_records: List[dict] = []
    for band in INCIDENCE_AGE_BANDS:
        exposure = float(incidence_age_exposure.get(band, 0.0))
        events = int(incidence_age_onsets.get(band, 0))
        hazard_per_year = (events / exposure) if exposure > 0 else 0.0
        probability_per_year = hazard_to_prob(hazard_per_year, dt=1.0)
        alive_count = int(age_band_alive_counts.get(band, 0))
        dementia_count = int(age_band_dementia_counts.get(band, 0))
        prevalence_value = (dementia_count / alive_count) if alive_count > 0 else 0.0
        lower, upper = band
        incidence_age_records.append({
            'age_band': age_band_label(band),
            'age_lower': lower,
            'age_upper': upper,
            'age_mid': age_band_midpoint(band),
            'person_years_at_risk': exposure,
            'incident_onsets': events,
             'cases_all': dementia_count,
             'population_all': alive_count,
             'prevalence': prevalence_value,
            'incidence_hazard_per_year': hazard_per_year,
            'incidence_probability_per_year': probability_per_year,
        })

    incidence_age_df = pd.DataFrame(incidence_age_records)
    if not incidence_age_df.empty:
        incidence_age_df['prevalence (smoothed)'] = smooth_series(
            incidence_age_df['prevalence'].tolist()
        )
        ref_hazard_series = incidence_age_df.loc[
            incidence_age_df['incidence_hazard_per_year'] > 0,
            'incidence_hazard_per_year'
        ]
        ref_hazard = float(ref_hazard_series.iloc[0]) if not ref_hazard_series.empty else float('nan')
        if ref_hazard > 0 and math.isfinite(ref_hazard):
            incidence_age_df['h/h_ref'] = incidence_age_df['incidence_hazard_per_year'] / ref_hazard
        else:
            incidence_age_df['h/h_ref'] = np.nan
        incidence_age_df['log(h/h_ref)'] = np.log(incidence_age_df['h/h_ref'].replace({0: np.nan}))
        incidence_age_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Reorder and rename columns for downstream consumers
        column_map = {
            'age_band': 'age band',
            'age_mid': 'age mid',
            'cases_all': 'cases (all)',
            'population_all': 'population (all)',
            'prevalence': 'prevalence',
            'incidence_hazard_per_year': 'Incidence hazard/yr'
        }
        incidence_age_df.rename(columns=column_map, inplace=True)
        desired_columns = [
            'age band',
            'age mid',
            'cases (all)',
            'population (all)',
            'prevalence',
            'prevalence (smoothed)',
            'Incidence hazard/yr',
            'h/h_ref',
            'log(h/h_ref)',
            'person_years_at_risk',
            'incident_onsets',
            'age_lower',
            'age_upper',
            'incidence_probability_per_year',
        ]
        # Keep consistent ordering for later selection
        existing_cols = [c for c in desired_columns if c in incidence_age_df.columns]
        remaining_cols = [c for c in incidence_age_df.columns if c not in existing_cols]
        incidence_age_df = incidence_age_df[existing_cols + remaining_cols]

    incidence_by_year_sex_df = pd.DataFrame(yearly_incidence_records)
    if not incidence_by_year_sex_df.empty:
        incidence_by_year_sex_df.sort_values(['calendar_year', 'sex', 'age_lower'], inplace=True)
        incidence_by_year_sex_df.reset_index(drop=True, inplace=True)

    onset_age_counter: Counter = Counter()
    for person in population_state.values():
        age_at_onset = person.get('age_at_onset')
        if age_at_onset is None:
            continue
        try:
            age_int = int(round(float(age_at_onset)))
        except (TypeError, ValueError):
            continue
        onset_age_counter[age_int] += 1

    return {
        'summaries': summary_history,
        'initial_age_distribution': dict(initial_age_counter),
        'age_at_death_distribution': dict(death_age_counter),
        'individual_survival': survival_records,
        'transition_history': transition_history,
        'incident_onsets_by_risk_factor': risk_onset_tracker,
        'lifetime_risk_by_entry_age': lifetime_risk_normal,
        'lifetime_risk_by_entry_age_all': lifetime_risk_all,
        'incidence_by_age_band': incidence_age_records,
        'incidence_by_age_band_df': incidence_age_df,
        'incidence_by_year_sex': yearly_incidence_records,
        'incidence_by_year_sex_df': incidence_by_year_sex_df,
        'age_at_onset_distribution': dict(onset_age_counter),
        'age_band_alive_counts': {age_band_label(b): int(count) for b, count in age_band_alive_counts.items()},
        'age_band_dementia_counts': {age_band_label(b): int(count) for b, count in age_band_dementia_counts.items()},
        'age_band_incidence_summary': incidence_age_df[['age band',
                                                        'age mid',
                                                        'cases (all)',
                                                        'population (all)',
                                                        'prevalence',
                                                        'prevalence (smoothed)',
                                                        'Incidence hazard/yr',
                                                        'h/h_ref',
                                                        'log(h/h_ref)']].copy() if not incidence_age_df.empty else pd.DataFrame(),
    }


def _total_incident_onsets(model_results: dict) -> int:
    """Sum dementia onsets across all simulated time steps."""
    summaries = model_results.get('summaries', {}) if isinstance(model_results, dict) else {}
    total = 0
    for summary in summaries.values():
        total += int(summary.get('incident_onsets', 0) or 0)
    return total


def _replace_nested_values(structure: Any, value: float) -> Any:
    """Recursively replace all numeric leaves in a nested mapping with the provided value."""
    if isinstance(structure, dict):
        return {key: _replace_nested_values(nested, value) for key, nested in structure.items()}
    return float(value)


def _counterfactual_config_without_risk(config: dict, risk_factor: str) -> dict:
    """
    Create a deep copy of the configuration where the chosen risk factor has zero prevalence
    and neutral (1.0) hazard ratios across all transitions.
    """
    cfg = copy.deepcopy(config)
    risk_defs = cfg.get('risk_factors', {})
    meta = risk_defs.get(risk_factor)
    if not isinstance(meta, dict):
        return cfg

    meta['prevalence'] = 0.0
    rr_spec = meta.get('relative_risks', {})
    if isinstance(rr_spec, dict) and rr_spec:
        meta['relative_risks'] = _replace_nested_values(rr_spec, 1.0)
    else:
        meta['relative_risks'] = {
            'onset': 1.0,
            'mild_to_moderate': 1.0,
            'moderate_to_severe': 1.0,
            'severe_to_death': 1.0,
        }
    return cfg


def compute_population_attributable_fraction(config: dict,
                                             risk_factor: str,
                                             baseline_results: Optional[dict] = None,
                                             seed: Optional[int] = None) -> Optional[dict]:
    """
    Estimate the population attributable fraction (PAF) for a named risk factor by
    comparing baseline simulation results with a counterfactual scenario in which
    the risk factor is removed (zero prevalence and neutral hazard ratios).
    """
    baseline_results_local = baseline_results
    if baseline_results_local is None:
        baseline_results_local = run_model(copy.deepcopy(config), seed=seed)

    counterfactual_results = run_model(
        _counterfactual_config_without_risk(config, risk_factor),
        seed=seed,
    )

    baseline_onsets = _total_incident_onsets(baseline_results_local)
    counterfactual_onsets = _total_incident_onsets(counterfactual_results)
    if baseline_onsets <= 0:
        return None

    paf = (baseline_onsets - counterfactual_onsets) / baseline_onsets
    paf = max(0.0, min(1.0, paf))

    baseline_breakdown = (
        baseline_results_local
        .get('incident_onsets_by_risk_factor', {})
        .get(risk_factor, {})
        if isinstance(baseline_results_local, dict) else {}
    )

    return {
        'risk_factor': risk_factor,
        'baseline_onsets': baseline_onsets,
        'counterfactual_onsets': counterfactual_onsets,
        'paf': paf,
        'baseline_with_risk_onsets': baseline_breakdown.get('with', 0),
        'baseline_without_risk_onsets': baseline_breakdown.get('without', 0),
    }


# -------- Probabilistic sensitivity analysis (PSA) utilities --------

def apply_psa_draw(base_config: dict,
                   psa_cfg: Optional[dict] = None,
                   rng: Optional[np.random.Generator] = None) -> dict:
    """
    Return a deep-copied config with PSA sampling applied to costs, utilities,
    probabilities, and risk-factor parameters.
    """
    if rng is None:
        rng = np.random.default_rng()
    psa_meta = psa_cfg or base_config.get('psa') or {}
    rel_beta = float(psa_meta.get('relative_sd_beta', PSA_DEFAULT_RELATIVE_SD))
    rel_gamma = float(psa_meta.get('relative_sd_gamma', PSA_DEFAULT_RELATIVE_SD))

    cfg = copy.deepcopy(base_config)

    base_prob = cfg.get('base_onset_probability')
    if base_prob is not None:
        cfg['base_onset_probability'] = _sample_probability_value(base_prob, rel_beta, rng)

    costs_cfg = cfg.get('costs')
    if isinstance(costs_cfg, dict):
        _sample_cost_structure(costs_cfg, rel_gamma, rng)

    for key in ('utility_norms_by_age', 'utility_multipliers', 'stage_age_qalys', 'dementia_stage_qalys'):
        mapping = cfg.get(key)
        if isinstance(mapping, dict):
            _apply_beta_to_mapping(mapping, rel_beta, rng)

    risk_defs = cfg.get('risk_factors')
    if isinstance(risk_defs, dict):
        _sample_risk_factor_prevalence(risk_defs, rel_beta, rng)
        _sample_risk_factor_relative_risks(risk_defs, rng)

    return cfg


def extract_psa_metrics(model_results: dict) -> dict:
    """
    Pull decision metrics (costs, QALYs, incidence, severity) from a single model run.
    """
    summaries = model_results.get('summaries', {}) if isinstance(model_results, dict) else {}
    if not summaries:
        return {}
    final_step = max(summaries)
    final_summary = summaries[final_step]

    total_incidence = 0
    for summary in summaries.values():
        total_incidence += int(summary.get('incident_onsets', 0) or 0)

    metrics = {
        'total_costs_nhs': float(final_summary.get('total_costs_nhs', 0.0) or 0.0),
        'total_costs_informal': float(final_summary.get('total_costs_informal', 0.0) or 0.0),
        'total_qalys_patient': float(final_summary.get('total_qalys_patient', 0.0) or 0.0),
        'total_qalys_caregiver': float(final_summary.get('total_qalys_caregiver', 0.0) or 0.0),
        'incident_onsets_total': float(total_incidence),
    }
    metrics['total_costs_all'] = metrics['total_costs_nhs'] + metrics['total_costs_informal']
    metrics['total_qalys_combined'] = metrics['total_qalys_patient'] + metrics['total_qalys_caregiver']

    for stage in DEMENTIA_DISEASE_STAGES:
        key = f'stage_{stage}'
        metrics[key] = float(final_summary.get(key, 0) or 0)

    return metrics


def summarize_psa_results(metrics_df: pd.DataFrame) -> Dict[str, dict]:
    """
    Compute mean and 95% intervals for each numeric PSA metric.
    """
    if metrics_df is None or metrics_df.empty:
        return {}
    summary: Dict[str, dict] = {}
    for column in metrics_df.columns:
        if column == 'iteration':
            continue
        if not pd.api.types.is_numeric_dtype(metrics_df[column]):
            continue
        series = metrics_df[column].dropna()
        if series.empty:
            continue
        summary[column] = {
            'mean': float(series.mean()),
            'lower_95': float(series.quantile(0.025)),
            'upper_95': float(series.quantile(0.975)),
        }
    return summary


def run_probabilistic_sensitivity_analysis(base_config: dict,
                                           psa_cfg: Optional[dict] = None,
                                           *,
                                           collect_draw_level: bool = False,
                                           seed: Optional[int] = None) -> dict:
    """
    Execute a Monte Carlo PSA using the provided configuration.
    Returns summary 95% intervals plus optional draw-level metrics.
    """
    psa_meta = copy.deepcopy(psa_cfg or base_config.get('psa') or {})
    if not psa_meta.get('use', False):
        raise ValueError("PSA is disabled; set config['psa']['use'] = True to run the analysis.")

    iterations = int(psa_meta.get('iterations', 1000))
    if iterations <= 0:
        raise ValueError("PSA iterations must be a positive integer.")

    base_seed = seed if seed is not None else psa_meta.get('seed')
    rng = np.random.default_rng(base_seed)

    draw_metrics: List[dict] = []
    for draw_idx in range(iterations):
        draw_config = apply_psa_draw(base_config, psa_meta, rng)
        draw_seed = int(rng.integers(0, 2**32 - 1))
        draw_results = run_model(draw_config, seed=draw_seed)
        metrics = extract_psa_metrics(draw_results)
        metrics['iteration'] = draw_idx + 1
        draw_metrics.append(metrics)

    metrics_df = pd.DataFrame(draw_metrics)
    summary = summarize_psa_results(metrics_df)

    payload = {
        'summary': summary,
        'iterations': iterations,
    }
    if collect_draw_level:
        payload['draws'] = metrics_df
    return payload

# Visuals

def save_or_show(save_path, show=False, label="plot"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"Saved {label} to {save_path.resolve()}")


def plot_ad_prevalence(model_results, save_path="plots/ad_prevalence.png", show=False):
    df = summaries_to_dataframe(model_results)
    if df.empty:
        print("No summary data available; skipping prevalence plot.")
        return

    ad_cols = [f'stage_{stage}' for stage in DEMENTIA_DISEASE_STAGES]
    for col in ad_cols:
        if col not in df.columns:
            df[col] = 0

    df['ad_cases'] = df[ad_cols].sum(axis=1)
    # Use alive population each cycle as denominator; fall back to total if alive not tracked
    denom_series = df['population_alive'] if 'population_alive' in df.columns else df['population_total']
    denom_series = denom_series.replace(0, np.nan)
    df['prevalence_pct'] = 100.0 * df['ad_cases'] / denom_series
    df['prevalence_pct'] = df['prevalence_pct'].fillna(0.0)
    x = df['calendar_year'] if 'calendar_year' in df.columns else df['time_step']

    max_prev = float(df['prevalence_pct'].max()) if not df.empty else 0.0
    upper_limit = 5.0 if max_prev <= 0 else max(5.0, max_prev * 1.1)

    plt.figure()
    plt.plot(x, df['prevalence_pct'], marker='o')
    plt.ylabel("Alzheimer's prevalence (%)")
    plt.xlabel('Year' if 'calendar_year' in df.columns else 'Time step (years)')
    plt.ylim(0, upper_limit)
    plt.title("Alzheimer's prevalence over time")
    save_or_show(save_path, show, label="prevalence plot")


def plot_ad_incidence(model_results,
                      save_path="plots/ad_incidence.png",
                      show=False):
    df = summaries_to_dataframe(model_results)
    if df.empty:
        print("No summary data available; skipping incidence plot.")
        return
    if 'incident_onsets' not in df.columns:
        print("Incident onset counts not available; skipping incidence plot.")
        return

    df = df.copy()
    axis_col: Optional[str]
    if 'calendar_year' in df.columns:
        axis_col = 'calendar_year'
    elif 'time_step' in df.columns:
        axis_col = 'time_step'
    else:
        print("No time axis available; skipping incidence plot.")
        return

    df = df.sort_values(axis_col).reset_index(drop=True)

    baseline_value = df[axis_col].iloc[0]
    post_baseline_df = df.loc[df[axis_col] != baseline_value].copy()
    if post_baseline_df.empty:
        print("Only baseline data available; skipping incidence plot.")
        return
    df = post_baseline_df

    df['incident_onsets'] = df['incident_onsets'].fillna(0)
    if 'incidence_per_1000_alive' not in df.columns:
        alive = df.get('population_alive', pd.Series(dtype=float)).replace(0, np.nan)
        df['incidence_per_1000_alive'] = (df['incident_onsets'] / alive) * 1000.0
        df['incidence_per_1000_alive'] = df['incidence_per_1000_alive'].fillna(0.0)
    else:
        df['incidence_per_1000_alive'] = df['incidence_per_1000_alive'].fillna(0.0)

    x = df[axis_col]
    counts = df['incident_onsets']
    rates = df['incidence_per_1000_alive']

    fig, ax_count = plt.subplots()
    ax_count.bar(x, counts, width=0.6, alpha=0.4, label="New onsets (count)")
    if len(df) >= 2:
        x_numeric = pd.to_numeric(x, errors='coerce')
        if x_numeric.isna().any():
            x_numeric = pd.Series(np.arange(len(df), dtype=float), index=df.index)
        x_numeric = x_numeric.astype(float)
        coeffs = np.polyfit(x_numeric.to_numpy(dtype=float), counts.to_numpy(dtype=float), 1)
        trend_counts = np.polyval(coeffs, x_numeric.to_numpy(dtype=float))
        ax_count.plot(x, trend_counts, color="tab:green", linestyle="--", linewidth=2, label="New onsets trend")
    ax_count.set_ylabel("New onsets (count)")

    ax_rate = ax_count.twinx()
    ax_rate.plot(x, rates, color="tab:red", marker='o', label="Incidence per 1,000 alive")
    ax_rate.set_ylabel("Incidence per 1,000 alive")

    max_count = float(counts.max()) if not counts.empty else 0.0
    max_rate = float(rates.max()) if not rates.empty else 0.0
    count_upper = max_count * 1.1 if max_count > 0 else 1.0
    rate_upper = max_rate * 1.1 if max_rate > 0 else 1.0
    ax_count.set_ylim(0.0, count_upper)
    ax_rate.set_ylim(0.0, rate_upper)

    ax_count.set_xlabel('Year' if 'calendar_year' in df.columns else 'Time step (years)')
    ax_count.set_title("Alzheimer's incidence over time")

    handles1, labels1 = ax_count.get_legend_handles_labels()
    handles2, labels2 = ax_rate.get_legend_handles_labels()
    ax_count.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    save_or_show(save_path, show, label="incidence plot")


def plot_age_specific_ad_cases(model_results,
                               age_bands: Optional[List[Tuple[int, Optional[int]]]] = None,
                               save_path: str = "plots/ad_age_specific_cases.png",
                               show: bool = False) -> None:
    """Stacked bar chart of dementia cases by age band for each simulated year."""
    df = summaries_to_dataframe(model_results)
    if df.empty:
        print("No summary data available; skipping age-specific dementia plot.")
        return

    bands = age_bands if age_bands is not None else REPORTING_AGE_BANDS
    if not bands:
        print("No age bands configured for reporting; skipping age-specific dementia plot.")
        return

    index_field = 'calendar_year' if 'calendar_year' in df.columns else 'time_step'
    x_labels = df[index_field].tolist()
    x_positions = np.arange(len(x_labels))

    plt.figure()
    stacked_bottom = np.zeros(len(x_labels), dtype=float)

    for band in bands:
        column_name = f"ad_cases_age_{age_band_key(band)}"
        if column_name not in df.columns:
            df[column_name] = 0
        counts = df[column_name].fillna(0.0).to_numpy(dtype=float)
        plt.bar(x_positions,
                counts,
                bottom=stacked_bottom,
                width=0.6,
                label=age_band_label(band))
        stacked_bottom += counts

    plt.xticks(x_positions, x_labels, rotation=90, ha='center')
    plt.ylabel("Estimated dementia cases (count)")
    plt.xlabel('Year' if index_field == 'calendar_year' else 'Time step (years)')
    plt.title("Age-specific dementia cases over time")
    plt.legend(title="Age band")

    save_or_show(save_path, show, label="age-specific dementia cases plot")


def plot_dementia_prevalence_by_stage(model_results,
                                      stages: Optional[List[str]] = None,
                                      save_path: str = "plots/ad_stage_prevalence.png",
                                      show: bool = False) -> None:
    """Stacked bar chart showing dementia prevalence by stage for each simulated year."""
    df = summaries_to_dataframe(model_results)
    if df.empty:
        print("No summary data available; skipping stage-specific prevalence plot.")
        return

    tracked_stages = stages if stages is not None else list(DEMENTIA_DISEASE_STAGES)
    if not tracked_stages:
        print("No stages provided for prevalence plotting; skipping stage-specific prevalence plot.")
        return

    index_field = 'calendar_year' if 'calendar_year' in df.columns else 'time_step'
    x_labels = df[index_field].tolist()
    x_positions = np.arange(len(x_labels))

    denom = df['population_alive'] if 'population_alive' in df.columns else df['population_total']
    denom = denom.replace(0, np.nan)

    plt.figure()
    stacked_bottom = np.zeros(len(x_labels), dtype=float)

    for stage in tracked_stages:
        column_name = f'stage_{stage}'
        if column_name not in df.columns:
            df[column_name] = 0
        prevalence_series = (df[column_name].fillna(0.0) / denom).replace([np.inf, -np.inf], np.nan)
        prevalence_pct = prevalence_series.fillna(0.0) * 100.0
        values = prevalence_pct.to_numpy(dtype=float)

        plt.bar(
            x_positions,
            values,
            bottom=stacked_bottom,
            width=0.6,
            label=stage.replace('_', ' ').title()
        )
        stacked_bottom += values

    plt.xticks(x_positions, x_labels, rotation=90, ha='center')
    plt.ylabel("Share of alive population (%)")
    plt.xlabel('Year' if index_field == 'calendar_year' else 'Time step (years)')
    plt.title("Dementia prevalence by stage over time")
    y_max = stacked_bottom.max() if stacked_bottom.size else 0.0
    y_upper = max(5.0, y_max * 1.05) if y_max > 0 else 5.0
    plt.ylim(0, min(100.0, y_upper))
    plt.legend(title="Stage")

    save_or_show(save_path, show, label="stage-specific prevalence plot")


def plot_survival_curve(model_results, save_path='plots/survival_curve.png', show=False):
    df = summaries_to_dataframe(model_results)
    if df.empty:
        print("No summary data available; skipping survival plot.")
        return

    series_name = 'baseline_alive' if 'baseline_alive' in df.columns else 'population_alive'
    if series_name not in df.columns or df[series_name].fillna(0).iloc[0] == 0:
        print("No individuals at baseline; cannot plot survival.")
        return

    alive = df[series_name].fillna(0)
    baseline_alive = alive.iloc[0]
    survival = alive / baseline_alive if baseline_alive else alive
    x = df['calendar_year'] if 'calendar_year' in df.columns else df['time_step']

    plt.figure()
    plt.step(x, survival, where='post')
    plt.ylim(0, 1.01)
    plt.xlabel("Year" if 'calendar_year' in df.columns else "Time step (years)")
    plt.ylabel("Survival proportion")
    title = "Survival curve"
    if series_name == 'baseline_alive':
        title += " (baseline cohort)"
    plt.title(title)
    save_or_show(save_path, show, label="survival curve")


def _kaplan_meier_curve(times: np.ndarray, events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if times.size == 0:
        return np.array([]), np.array([])

    order = np.argsort(times)
    times = times[order]
    events = events[order]

    event_mask = events == 1
    unique_event_times = np.unique(times[event_mask])

    timeline = [0.0]
    survival = [1.0]
    current = 1.0

    for t in unique_event_times:
        at_risk = np.sum(times >= t)
        events_at_t = np.sum((times == t) & event_mask)
        if at_risk == 0:
            continue
        current *= (1.0 - events_at_t / at_risk)
        timeline.append(float(t))
        survival.append(current)

    max_time = float(times.max())
    if timeline[-1] < max_time:
        timeline.append(max_time)
        survival.append(current)

    return np.array(timeline), np.array(survival)


def plot_survival_by_baseline_stage(model_results,
                                    save_path='plots/survival_by_stage.png',
                                    show=False):
    records = model_results.get('individual_survival', []) if isinstance(model_results, dict) else []
    if not records:
        print("No individual-level survival records available; skipping Kaplan-Meier plot.")
        return

    survival_df = pd.DataFrame(records)
    if survival_df.empty or 'baseline_stage' not in survival_df or 'time' not in survival_df:
        print("Incomplete survival records; skipping Kaplan-Meier plot.")
        return

    plt.figure()
    unique_stages = [stage for stage in DEMENTIA_STAGES if stage in survival_df['baseline_stage'].unique()]
    for stage in unique_stages:
        stage_df = survival_df[survival_df['baseline_stage'] == stage]
        if stage_df.empty:
            continue
        times = stage_df['time'].to_numpy(dtype=float)
        events = stage_df['event'].to_numpy(dtype=int)
        t_points, surv = _kaplan_meier_curve(times, events)
        if t_points.size == 0:
            continue
        label = stage.replace('_', ' ').title()
        plt.step(t_points, surv, where='post', label=label)

    if not plt.gca().has_data():
        print("No valid Kaplan-Meier curves to plot.")
        plt.close()
        return

    plt.ylim(0, 1.01)
    plt.xlabel("Time since entry (years)")
    plt.ylabel("Survival proportion")
    plt.title("Kaplan-Meier survival by baseline dementia stage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_or_show(save_path, show, label="Kaplan-Meier survival by stage")


def plot_baseline_age_hist(model_results, bins=None, save_path="plots/baseline_age_hist.png", show=False):
    age_dist = model_results.get('initial_age_distribution', {}) if isinstance(model_results, dict) else {}
    if not age_dist:
        print("No baseline age distribution available; skipping histogram.")
        return

    ages = np.array(sorted(age_dist.keys()))
    counts = np.array([age_dist[a] for a in ages], dtype=float)

    plt.figure()
    if isinstance(bins, int) and bins > 0:
        hist, bin_edges = np.histogram(ages, bins=bins, weights=counts)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        widths = np.diff(bin_edges)
        plt.bar(centers, hist, width=widths, align='center')
    else:
        plt.bar(ages, counts, width=0.9, align='center')

    plt.xlabel("Age at baseline (years)")
    plt.ylabel("Count")
    plt.title("Baseline age distribution")
    save_or_show(save_path, show, label="baseline age histogram")


def plot_age_at_death_hist(model_results, bins=20,
                           save_path='plots/age_at_death_hist.png', show=False):
    age_dist = model_results.get('age_at_death_distribution', {}) if isinstance(model_results, dict) else {}
    if not age_dist:
        print("No deaths to plot (yet).")
        return

    ages = np.array(sorted(age_dist.keys()))
    counts = np.array([age_dist[a] for a in ages], dtype=float)

    plt.figure()
    if isinstance(bins, int) and bins > 0:
        hist, bin_edges = np.histogram(ages, bins=bins, weights=counts)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        widths = np.diff(bin_edges)
        plt.bar(centers, hist, width=widths, align='center')
    else:
        plt.bar(ages, counts, width=0.9, align='center')

    plt.xlabel("Age at death (years)")
    plt.ylabel("Count")
    plt.title("Distribution of age at death")
    save_or_show(save_path, show, label="age-at-death histogram")


def _compute_cost_qaly_metrics(model_results):
    df = summaries_to_dataframe(model_results)
    if df.empty:
        return df

    df = df.copy()
    denom = df['population_total'].replace(0, np.nan)
    df['cumulative_costs_nhs_mean'] = df['total_costs_nhs'] / denom
    df['cumulative_costs_informal_mean'] = df['total_costs_informal'] / denom
    df['cumulative_qalys_patient_mean'] = df['total_qalys_patient'] / denom
    df['cumulative_qalys_caregiver_mean'] = df['total_qalys_caregiver'] / denom

    df['total_costs_mean'] = df['cumulative_costs_nhs_mean'] + df['cumulative_costs_informal_mean']
    df['total_qalys_mean'] = df['cumulative_qalys_patient_mean'] + df['cumulative_qalys_caregiver_mean']
    df['total_costs_total'] = df['total_costs_nhs'] + df['total_costs_informal']
    df['total_qalys_total'] = df['total_qalys_patient'] + df['total_qalys_caregiver']
    df.fillna(0, inplace=True)
    return df


def plot_costs_per_person_over_time(model_results,
                                    save_path="plots/costs_per_person_over_time.png",
                                    show=False):
    """Mean cumulative costs per person (NHS, Informal, Total) over time."""
    agg = _compute_cost_qaly_metrics(model_results)
    if agg.empty:
        print("No summary data available; skipping cost plot.")
        return

    x = agg['calendar_year'] if 'calendar_year' in agg.columns else agg['time_step']

    plt.figure()
    plt.plot(x, agg['cumulative_costs_nhs_mean'], marker="o", linestyle="-", label="NHS (GBP/person)")
    plt.plot(x, agg['cumulative_costs_informal_mean'], marker="o", linestyle="--", label="Informal (GBP/person)")
    plt.plot(x, agg['total_costs_mean'], marker="o", linestyle="-.", label="Total (GBP/person)")
    plt.xlabel('Year' if 'calendar_year' in agg.columns else 'Time step (years)')
    plt.ylabel('Mean cumulative cost per person (GBP)')
    plt.title('Mean cumulative costs per person over time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_or_show(save_path, show, label="costs per person")


def plot_qalys_per_person_over_time(model_results,
                                    save_path="plots/qalys_per_person_over_time.png",
                                    show=False):
    """Mean cumulative QALYs per person (Patient, Caregiver, Total) over time."""
    agg = _compute_cost_qaly_metrics(model_results)
    if agg.empty:
        print("No summary data available; skipping QALY plot.")
        return

    x = agg['calendar_year'] if 'calendar_year' in agg.columns else agg['time_step']

    plt.figure()
    plt.plot(x, agg['cumulative_qalys_patient_mean'], marker="o", linestyle="-", label="Patient QALYs/person")
    plt.plot(x, agg['cumulative_qalys_caregiver_mean'], marker="o", linestyle="--", label="Caregiver QALYs/person")
    plt.plot(x, agg['total_qalys_mean'], marker="o", linestyle="-.", label="Total QALYs/person")
    plt.xlabel('Year' if 'calendar_year' in agg.columns else 'Time step (years)')
    plt.ylabel('Mean cumulative QALYs per person')
    plt.title('Mean cumulative QALYs per person over time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_or_show(save_path, show, label="QALYs per person")

# ---------- Hazard-ratio visualisers (use base_onset_probability path) ----------

def _mock_person(age: int, risk_flags: dict, sex: str = "female") -> dict:
    return {
        "age": age,
        "sex": _canonical_sex_label(sex),
        "risk_factors": risk_flags,
        "dementia_stage": "cognitively_normal",
        "time_in_stage": 0,
        "living_setting": "home",
        "alive": True,
    }

def _onset_hazard_from_base_prob(config: dict,
                                 age: int,
                                 risk_flags: dict,
                                 sex: str = "female") -> float:
    """
    Use base_onset_probability (per-cycle) -> hazard, then apply age & risk-factor HRs.
    This path is used when 'normal_to_mild' is NOT defined in stage_transition_durations.
    """
    dt = config["time_step_years"]
    p0 = config.get("base_onset_probability", 0.0)
    h0 = prob_to_hazard(p0, dt=dt)
    age_hr = get_age_hr_for_transition(age, config, "onset")  # CHANGED: parametric/banded
    person = _mock_person(age, risk_flags, sex=sex)
    h = apply_hazard_ratios(
        h0,
        person["risk_factors"],
        config["risk_factors"],
        "onset",
        age_hr,
        person["age"],
        person.get("sex", "unspecified"),
        config,  # NEW
    )
    return h

def plot_onset_hazard_vs_age_from_base_prob(config: dict,
                                            risk_profiles: dict = None,
                                            ages: np.ndarray | list = None,
                                            save_path: str = "plots/onset_hazard_vs_age.png",
                                            show: bool = False):
    """Plot the adjusted ONSET hazard vs age using base_onset_probability (no duration needed)."""
    if risk_profiles is None:
        risk_profiles = {
            "None": {},
            "Periodontal": {"periodontal_disease": True},
            "Smoking": {"smoking": True},
            "Smoking + Periodontal": {"smoking": True, "periodontal_disease": True},
        }
    if ages is None:
        ages = np.arange(50, 91, 5)

    plt.figure()
    for label, flags in risk_profiles.items():
        if isinstance(flags, dict):
            profile_sex = _canonical_sex_label(flags.get('sex', 'female'))
            risk_flags = {k: v for k, v in flags.items() if k != 'sex'}
        else:
            profile_sex = 'female'
            risk_flags = {}
        hazards = [
            _onset_hazard_from_base_prob(config, age, risk_flags, sex=profile_sex)
            for age in ages
        ]
        plt.plot(ages, hazards, marker="o", label=label)

    plt.xlabel("Age (years)")
    plt.ylabel("Onset hazard (per year)")
    plt.title("Onset hazard vs age (from base_onset_probability)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_or_show(save_path, show, label="onset hazard vs age")

def plot_onset_probability_vs_age_from_base_prob(config: dict,
                                                 risk_profiles: dict = None,
                                                 ages: np.ndarray | list = None,
                                                 save_path: str = "plots/onset_probability_vs_age.png",
                                                 show: bool = False):
    """Same as above, but converted to per-cycle probabilities for your Δt."""
    if risk_profiles is None:
        risk_profiles = {
            "None": {},
            "Periodontal": {"periodontal_disease": True},
            "Smoking": {"smoking": True},
            "Smoking + Periodontal": {"smoking": True, "periodontal_disease": True},
        }
    if ages is None:
        ages = np.arange(50, 91, 5)

    dt = config["time_step_years"]
    plt.figure()
    for label, flags in risk_profiles.items():
        if isinstance(flags, dict):
            profile_sex = _canonical_sex_label(flags.get('sex', 'female'))
            risk_flags = {k: v for k, v in flags.items() if k != 'sex'}
        else:
            profile_sex = 'female'
            risk_flags = {}
        probs = [
            hazard_to_prob(
                _onset_hazard_from_base_prob(config, age, risk_flags, sex=profile_sex),
                dt=dt,
            )
            for age in ages
        ]
        plt.plot(ages, probs, marker="o", label=label)

    plt.xlabel("Age (years)")
    plt.ylabel(f"Per-cycle probability (Δt={dt} y)")
    plt.title("Onset probability vs age (from base_onset_probability)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_or_show(save_path, show, label="onset probability vs age")

# ===== Constant-hazard diagnostics (add below your imports) =====

def check_constant_hazard_from_transitions(model_results: dict,
                                           from_stage: str,
                                           to_stage: str,
                                           *,
                                           dt: float = 1.0,
                                           tolerance: float = 0.05) -> dict:
    """
    Uses transition_history to test if the per-step hazard for a specific transition is ~constant over time.
    Returns: dict with mean_hazard, max_relative_deviation, within_tolerance, and a DataFrame of interval hazards.
    """
    th = model_results.get('transition_history', {}) or {}
    rows = []
    for t in sorted(th.keys()):
        payload = th[t] or {}
        starts = payload.get('stage_start_counts', {}) or {}
        trans  = payload.get('transition_counts', {}) or {}
        start_n = float(starts.get(from_stage, 0.0))
        if start_n <= 0:
            continue
        trans_n  = float(trans.get((from_stage, to_stage), 0.0))
        # per-interval probability for that step
        p = trans_n / start_n
        p = max(0.0, min(1.0, p))
        # convert probability -> hazard under exponential assumption for interval length dt
        h = -math.log(max(1e-12, 1.0 - p)) / dt
        rows.append({
            'time_step': t,
            'start_count': start_n,
            'transition_count': trans_n,
            'probability': p,
            'interval_hazard': h,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return {
            'mean_hazard': float('nan'),
            'max_relative_deviation': float('nan'),
            'within_tolerance': False,
            'intervals': df,
        }

    mean_h = float(df['interval_hazard'].mean())
    if mean_h > 0:
        df['relative_deviation'] = (df['interval_hazard'] - mean_h) / mean_h
        max_dev = float(df['relative_deviation'].abs().max())
        within  = bool(max_dev <= tolerance)
    else:
        df['relative_deviation'] = float('nan')
        max_dev = float('nan')
        within  = False

    return {
        'mean_hazard': mean_h,
        'max_relative_deviation': max_dev,
        'within_tolerance': within,
        'intervals': df,
    }


def plot_transition_interval_hazards(intervals_df: pd.DataFrame,
                                     *,
                                     title: str,
                                     save_path: str = "plots/transition_interval_hazards.png",
                                     show: bool = False) -> None:
    """
    Simple line plot of per-step interval hazards with a horizontal line at the mean.
    Saves to file using your existing save_or_show() helper.
    """
    if intervals_df is None or intervals_df.empty:
        print(f"No intervals to plot for: {title}")
        return

    # Choose x from calendar_year if available via summaries; otherwise use time_step
    x = intervals_df['time_step']
    y = intervals_df['interval_hazard']
    mean_y = float(y.mean())

    plt.figure()
    plt.plot(x, y, marker='o', label='Interval hazard')
    plt.axhline(mean_y, linestyle='--', linewidth=1.5, label=f"Mean = {mean_y:.4f}")
    plt.xlabel("Time step (years)")
    plt.ylabel("Interval hazard (/year)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_or_show(save_path, show, label=title)


def run_constant_hazard_diagnostics(model_results: dict,
                                    config: dict,
                                    *,
                                    tolerance: float = 0.05,
                                    show_plots: bool = False) -> None:
    """
    Runs:
      (1) Whole-cohort survival constant-hazard check (your existing helper)
      (2) Per-transition constant-hazard checks for key transitions
    Prints a concise summary and saves plots under plots/.
    """
    dt = float(config.get('time_step_years', 1.0))

    # --- (1) Whole-cohort survival constant-hazard check ---
    try:
        cohort_check = check_constant_hazard_from_model(
            model_results,
            tolerance=tolerance,
            cohort='baseline',          # change to 'population' if you prefer
            use_calendar_year=True
        )
        print("\n[Whole-cohort survival] Constant-hazard check")
        print(f"  Time axis: {cohort_check.get('time_axis')}")
        print(f"  Mean hazard: {cohort_check['mean_hazard']:.6f} /year")
        print(f"  Max relative deviation: {cohort_check['max_relative_deviation']:.2%}")
        print(f"  Within tolerance (±{tolerance:.0%}): {cohort_check['within_tolerance']}")
        # Optional quick plot of piecewise hazards reconstructed from survival:
        intervals = cohort_check.get('intervals')
        if intervals is not None and not intervals.empty:
            # reuse the generic plotting helper by renaming columns to match expectation
            tmp = intervals.rename(columns={'time_start': 'time_step',
                                            'interval_hazard': 'interval_hazard'}).copy()
            # use time_end index as proxy for step index to avoid duplicate x
            tmp['time_step'] = np.arange(len(tmp), dtype=float)
            plot_transition_interval_hazards(
                tmp[['time_step', 'interval_hazard']].copy(),
                title="Whole-cohort survival: piecewise hazards",
                save_path="plots/whole_cohort_piecewise_hazards.png",
                show=show_plots
            )
    except ValueError as exc:
        print(f"\n[Whole-cohort survival] Check unavailable: {exc}")

    # --- (2) Per-transition constant-hazard checks ---
    transitions_to_test = [
        ('mild', 'moderate'),
        ('moderate', 'severe'),
        ('severe', 'death'),
    ]

    for (frm, to) in transitions_to_test:
        res = check_constant_hazard_from_transitions(
            model_results, frm, to, dt=dt, tolerance=tolerance
        )
        print(f"\n[{frm} → {to}] Constant-hazard check")
        print(f"  Mean hazard: {res['mean_hazard']:.6f} /year")
        print(f"  Max relative deviation: {res['max_relative_deviation']:.2%}")
        print(f"  Within tolerance (±{tolerance:.0%}): {res['within_tolerance']}")

        # Plot per-step interval hazards
        plot_transition_interval_hazards(
            res['intervals'],
            title=f"{frm.title()} → {to.title()} interval hazards",
            save_path=f"plots/{frm}_to_{to}_interval_hazards.png",
            show=show_plots
        )

# Lifetime risk visuals

def plot_lifetime_risk_by_entry_age(model_results,
                                    save_path: str = "plots/lifetime_risk_by_age.png",
                                    show: bool = False,
                                    *,
                                    include_all: bool = False,
                                    min_population: int = 25) -> None:
    """Line chart of lifetime dementia risk (%) by entry age."""
    data_key = 'lifetime_risk_by_entry_age_all' if include_all else 'lifetime_risk_by_entry_age'
    records = model_results.get(data_key, []) if isinstance(model_results, dict) else []
    if not records:
        target = "all entrants" if include_all else "cognitively normal entrants"
        print(f"No lifetime risk data for {target}; skipping lifetime risk plot.")
        return

    df = pd.DataFrame(records)
    if df.empty:
        print("Lifetime risk data frame is empty; skipping lifetime risk plot.")
        return

    if min_population > 1:
        df = df[df['population'] >= min_population]
        if df.empty:
            print(f"No ages meet the minimum population threshold ({min_population}); skipping lifetime risk plot.")
            return

    df = df.sort_values('entry_age').reset_index(drop=True)
    df['lifetime_risk_pct'] = df['lifetime_risk'] * 100.0

    plt.figure()
    plt.plot(df['entry_age'], df['lifetime_risk_pct'], marker='o')
    plt.xlabel("Age at entry (years)")
    plt.ylabel("Lifetime dementia risk (%)")
    title_suffix = " (all entrants)" if include_all else " (baseline cognitively normal)"
    plt.title(f"Lifetime dementia risk by entry age{title_suffix}")
    max_risk_pct = float(df['lifetime_risk_pct'].max()) if not df.empty else 0.0
    upper_limit = max(5.0, max_risk_pct * 1.1) if max_risk_pct > 0 else 5.0
    plt.ylim(0, upper_limit)
    plt.grid(True, alpha=0.3)

    save_or_show(save_path, show, label="lifetime dementia risk plot")

# Export

def export_results_to_excel(model_results, path="PD_AD_PD50.xlsx"):
    summaries = summaries_to_dataframe(model_results)
    if summaries.empty:
        print("No summary data available; nothing exported.")
        return

    calendar_lookup: Dict[int, Any] = {}
    if 'time_step' in summaries.columns:
        if 'calendar_year' in summaries.columns:
            calendar_lookup = dict(zip(summaries['time_step'], summaries['calendar_year']))
        else:
            calendar_lookup = dict(zip(summaries['time_step'], summaries['time_step']))

    with pd.ExcelWriter(path) as writer:
        summaries.to_excel(writer, sheet_name="Summary", index=False)

        target_years = [2025, 2030, 2035, 2040]
        severity_columns = [col for col in ('stage_mild', 'stage_moderate', 'stage_severe') if col in summaries.columns]
        if severity_columns and 'calendar_year' in summaries.columns:
            severity_df = (
                summaries[summaries['calendar_year'].isin(target_years)]
                [['calendar_year', *severity_columns]]
                .copy()
            )
            rename_map = {
                'stage_mild': 'mild_cases',
                'stage_moderate': 'moderate_cases',
                'stage_severe': 'severe_cases',
            }
            severity_df.rename(columns=rename_map, inplace=True)
            case_columns = [rename_map.get(col, col) for col in severity_columns]
            severity_df = pd.DataFrame({'calendar_year': target_years}).merge(
                severity_df, on='calendar_year', how='left'
            )
            for col in case_columns:
                if col not in severity_df.columns:
                    severity_df[col] = np.nan
            severity_df['total_dementia_cases'] = severity_df[case_columns].sum(axis=1, min_count=1)
            ordered_columns = ['calendar_year', *case_columns, 'total_dementia_cases']
            severity_df = severity_df[ordered_columns]
            severity_df.to_excel(writer, sheet_name="SeverityPrevalence", index=False)

        # Add high-level incidence metrics for quick reference in the workbook.
        incidence_cols = {'incident_onsets', 'population_alive'}
        if incidence_cols.issubset(summaries.columns):
            cases_cols = [col for col in ['time_step', 'calendar_year', 'incident_onsets', 'population_alive'] if col in summaries.columns]
            cases_df = summaries[cases_cols].copy().sort_values('time_step' if 'time_step' in cases_cols else cases_cols[0])
            denom = cases_df['population_alive'].replace(0, np.nan)
            cases_df['cases_per_1k_population'] = (cases_df['incident_onsets'] / denom) * 1_000.0
            cases_df['cases_per_1k_population'] = cases_df['cases_per_1k_population'].fillna(0.0)
            cases_df.to_excel(writer, sheet_name="CasesPer1k", index=False)

        incidence_by_year_sex_df = model_results.get('incidence_by_year_sex_df')
        if isinstance(incidence_by_year_sex_df, pd.DataFrame) and not incidence_by_year_sex_df.empty:
            incidence_by_year_sex_df.to_excel(writer, sheet_name="IncidenceByYearSex", index=False)

        mean_age_cols = ['time_step', 'calendar_year', 'mean_age_alive', 'mean_age_dementia']
        if set(mean_age_cols).issubset(summaries.columns):
            mean_age_df = summaries[mean_age_cols].sort_values('time_step')
            mean_age_df.to_excel(writer, sheet_name="MeanAgeByTimeStep", index=False)

        age_dist = model_results.get('initial_age_distribution', {}) if isinstance(model_results, dict) else {}
        if age_dist:
            age_df = pd.DataFrame(
                sorted(age_dist.items()), columns=["age", "count"]
            )
            age_df.to_excel(writer, sheet_name="BaselineAgeDist", index=False)

        death_dist = model_results.get('age_at_death_distribution', {}) if isinstance(model_results, dict) else {}
        if death_dist:
            death_df = pd.DataFrame(
                sorted(death_dist.items()), columns=["age", "count"]
            )
            death_df.to_excel(writer, sheet_name="AgeAtDeathDist", index=False)

        onset_dist = model_results.get('age_at_onset_distribution', {}) if isinstance(model_results, dict) else {}
        if onset_dist:
            onset_df = pd.DataFrame(
                sorted(onset_dist.items()), columns=["age_at_onset", "count"]
            )
            onset_df.to_excel(writer, sheet_name="AgeAtOnsetDist", index=False)

        risk_onsets = model_results.get('incident_onsets_by_risk_factor', {}) if isinstance(model_results, dict) else {}
        if risk_onsets:
            rows = []
            for risk_name, counts in risk_onsets.items():
                with_risk = int(counts.get('with', 0) or 0)
                without_risk = int(counts.get('without', 0) or 0)
                total = with_risk + without_risk
                rows.append({
                    "risk_factor": risk_name,
                    "onsets_with_risk": with_risk,
                    "onsets_without_risk": without_risk,
                    "total_onsets": total,
                    "with_fraction": (with_risk / total) if total else 0.0,
                })
            risk_df = pd.DataFrame(rows)
            risk_df.to_excel(writer, sheet_name="RiskFactorOnsets", index=False)

        paf_summary = model_results.get('paf_summary') if isinstance(model_results, dict) else None
        if paf_summary:
            paf_df = pd.DataFrame([paf_summary])
            paf_df.to_excel(writer, sheet_name="PAFSummary", index=False)

        lifetime_risk_normal = model_results.get('lifetime_risk_by_entry_age', []) if isinstance(model_results, dict) else []
        if lifetime_risk_normal:
            lifetime_df = pd.DataFrame(lifetime_risk_normal)
            if not lifetime_df.empty:
                lifetime_df = lifetime_df.sort_values('entry_age').reset_index(drop=True)
                lifetime_df['lifetime_risk_pct'] = lifetime_df['lifetime_risk'] * 100.0
                lifetime_df.to_excel(writer, sheet_name="LifetimeRiskNormal", index=False)

        lifetime_risk_all = model_results.get('lifetime_risk_by_entry_age_all', []) if isinstance(model_results, dict) else []
        if lifetime_risk_all:
            lifetime_all_df = pd.DataFrame(lifetime_risk_all)
            if not lifetime_all_df.empty:
                lifetime_all_df = lifetime_all_df.sort_values('entry_age').reset_index(drop=True)
                lifetime_all_df['lifetime_risk_pct'] = lifetime_all_df['lifetime_risk'] * 100.0
                lifetime_all_df.to_excel(writer, sheet_name="LifetimeRiskAll", index=False)

        transition_history = model_results.get('transition_history', {}) if isinstance(model_results, dict) else {}
        if transition_history:
            rows: List[dict] = []
            for time_step in sorted(transition_history.keys()):
                payload = transition_history.get(time_step, {}) or {}
                start_counts = payload.get('stage_start_counts', {}) or {}
                transition_counts = payload.get('transition_counts', {}) or {}
                calendar_year = calendar_lookup.get(time_step)
                for from_stage in DEMENTIA_STAGES:
                    start_total = float(start_counts.get(from_stage, 0))
                    if start_total <= 0:
                        continue
                    for to_stage in DEMENTIA_STAGES:
                        count = float(transition_counts.get((from_stage, to_stage), 0))
                        probability = count / start_total if start_total > 0 else 0.0
                        rows.append({
                            'time_step': time_step,
                            'calendar_year': calendar_year,
                            'from_stage': from_stage,
                            'to_stage': to_stage,
                            'start_count': start_total,
                            'transition_count': count,
                            'transition_probability': probability,
                        })
            transition_df = pd.DataFrame(rows)
            if not transition_df.empty:
                prob_matrix = transition_df.pivot(
                    index=['time_step', 'calendar_year', 'from_stage'],
                    columns='to_stage',
                    values='transition_probability'
                ).reset_index().fillna(0.0)
                prob_matrix.columns.name = None
                prob_matrix.to_excel(writer, sheet_name="TransitionProbabilities", index=False)

                avg_prob_matrix = (
                    transition_df.groupby(['from_stage', 'to_stage'])['transition_probability']
                    .mean()
                    .unstack(fill_value=0.0)
                    .reset_index()
                )
                avg_prob_matrix.columns.name = None
                avg_prob_matrix.to_excel(writer, sheet_name="TransitionProbabilitiesAverage", index=False)

                count_matrix = transition_df.pivot(
                    index=['time_step', 'calendar_year', 'from_stage'],
                    columns='to_stage',
                    values='transition_count'
                ).reset_index().fillna(0.0)
                count_matrix.columns.name = None
                count_matrix.to_excel(writer, sheet_name="TransitionCounts", index=False)

                start_counts_df = transition_df[['time_step', 'calendar_year', 'from_stage', 'start_count']].drop_duplicates()
                start_counts_df.to_excel(writer, sheet_name="TransitionStarts", index=False)

        age_band_summary = model_results.get('age_band_incidence_summary')
        if isinstance(age_band_summary, pd.DataFrame) and not age_band_summary.empty:
            age_band_summary.to_excel(writer, sheet_name="AgeHazardSummary", index=False)

    print(f"Saved aggregated model results to {path}")

# Run & output

if __name__ == "__main__":
    run_seed = 42
    model_results = run_model(general_config, seed=run_seed)

    try:
        baseline_check = check_constant_hazard_from_model(model_results, tolerance=0.05, cohort='baseline')
        mean_hazard = baseline_check['mean_hazard']
        max_dev = baseline_check['max_relative_deviation']
        within = baseline_check['within_tolerance']
        print("\nBaseline cohort constant-hazard check:")
        print(f"  Mean hazard: {mean_hazard:.6f} per year")
        print(f"  Max relative deviation: {max_dev:.2%}")
        print(f"  Within tolerance (+/- 5%): {within}")
    except ValueError as exc:
        print(f"\nBaseline cohort constant-hazard check unavailable: {exc}")

    if general_config.get('compute_paf_in_main', False):
        paf_summary = compute_population_attributable_fraction(
            general_config,
            risk_factor='periodontal_disease',
            baseline_results=model_results,
            seed=run_seed,
        )
        if paf_summary:
            model_results['paf_summary'] = paf_summary
            if general_config.get('report_paf_to_terminal', False):
                total_onsets = paf_summary['baseline_onsets']
                with_periodontal = paf_summary['baseline_with_risk_onsets']
                without_periodontal = paf_summary['baseline_without_risk_onsets']
                paf_value = paf_summary['paf']
                print("\nPeriodontal disease population attributable fraction (PAF) summary:")
                print(f"  Total dementia onsets (baseline): {total_onsets}")
                print(f"  Dementia onsets with periodontal disease: {with_periodontal}")
                print(f"  Dementia onsets without periodontal disease: {without_periodontal}")
                print(f"  PAF attributable to periodontal disease: {paf_value:.2%}")
                print(f"  PAF for those without periodontal disease: {(1.0 - paf_value):.2%}")
        elif general_config.get('report_paf_to_terminal', False):
            print("Unable to compute periodontal disease PAF (no dementia onsets in baseline scenario).")

    psa_cfg = general_config.get('psa', {})
    if psa_cfg.get('use', False):
        psa_results = run_probabilistic_sensitivity_analysis(
            general_config,
            psa_cfg,
            collect_draw_level=False,
            seed=run_seed,
        )
        summary = psa_results.get('summary', {})
        if summary:
            print("\nProbabilistic sensitivity analysis (95% CI):")
            focus_metrics = [
                'total_costs_all',
                'total_qalys_combined',
                'incident_onsets_total',
                'stage_mild',
                'stage_moderate',
                'stage_severe',
            ]
            for metric in focus_metrics:
                stats = summary.get(metric)
                if not stats:
                    continue
                mean_val = stats.get('mean')
                lo = stats.get('lower_95')
                hi = stats.get('upper_95')
                print(f"  {metric}: mean={mean_val:.2f}, 95% CI [{lo:.2f}, {hi:.2f}]")

    # After: model_results = run_model(general_config, seed=run_seed)

    # Run both diagnostics (set show_plots=True if you want them onscreen as well as saved)
    run_constant_hazard_diagnostics(
    model_results,
    general_config,
    tolerance=0.05,     # tighten/loosen as you like
    show_plots=False
)

    plot_ad_prevalence(model_results, show=True)
    plot_ad_incidence(model_results, show=True)
    plot_age_specific_ad_cases(model_results, show=True)
    plot_dementia_prevalence_by_stage(model_results, show=True)
    plot_survival_curve(model_results, show=True)
    plot_survival_by_baseline_stage(model_results, show=True)
    plot_baseline_age_hist(model_results, show=True)
    plot_age_at_death_hist(model_results, show=True)
    plot_costs_per_person_over_time(model_results, show=True)
    plot_qalys_per_person_over_time(model_results, show=True)
    plot_lifetime_risk_by_entry_age(model_results, show=True)
    plot_onset_hazard_vs_age_from_base_prob(general_config, show=True)
    plot_onset_probability_vs_age_from_base_prob(general_config, show=True)
    export_results_to_excel(model_results)

# -------- 1. Data (paste directly) --------
data = [
    [35, 49, "F", 0.0001, 0.000776677],
    [50, 64, "F", 0.0012, 0.003178285],
    [65, 79, "F", 0.0178, 0.024883946],
    [80, None, "F", 0.1244, 0.13821326],
    [35, 49, "M", 0.0001, 0.000971348],
    [50, 64, "M", 0.0013, 0.003344429],
    [65, 79, "M", 0.0168, 0.023354982],
    [80, None, "M", 0.0910, 0.10122974],
]
df = pd.DataFrame(data, columns=["age_lower","age_upper","sex","obs","pred"])
df["age_band"] = df.apply(
    lambda r: f"{int(r.age_lower)}+" if pd.isna(r.age_upper)
    else f"{int(r.age_lower)}-{int(r.age_upper)}", axis=1
)

# -------- 2. Output folder --------
# Use a relative path in the current working directory for portability
save_dir = Path("plots")
save_dir.mkdir(parents=True, exist_ok=True)

# -------- 3. Simple OLS regression (no statsmodels) --------
def ols_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.column_stack((np.ones_like(x), x))
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    y_hat = X @ beta
    resid = y - y_hat
    ss_tot = np.sum((y - y.mean())**2)
    ss_res = np.sum(resid**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"alpha": beta[0], "beta": beta[1], "r2": r2}

# -------- 4. Run regression + save plot --------
def run_and_save(sub, sex_label, save_dir):
    res = ols_fit(sub["pred"], sub["obs"])
    alpha, beta, R2 = res["alpha"], res["beta"], res["r2"]
    print(f"\nSex = {sex_label}")
    print(f"  α (intercept): {alpha:.6f}")
    print(f"  β (slope):     {beta:.6f}")
    print(f"  R²:            {R2:.3f}")

    plt.figure(figsize=(6,6))
    plt.scatter(sub["pred"], sub["obs"], s=90)
    xmax = max(sub["pred"].max(), sub["obs"].max()) * 1.05
    x = np.linspace(0, xmax, 100)
    plt.plot(x, x, "k--", label="1:1 line")
    plt.plot(x, alpha + beta*x, "r-", label=f"Fitted (β={beta:.2f})")
    for _, r in sub.iterrows():
        plt.annotate(r["age_band"], (r["pred"], r["obs"]), xytext=(4,4),
                     textcoords="offset points", fontsize=8)
    plt.xlabel("Predicted prevalence (2024)")
    plt.ylabel("Observed prevalence (2024)")
    plt.title(f"Calibration — Prevalence 2024 ({sex_label})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    # Save the plot
    file_path = os.path.join(save_dir, f"calibration_prevalence_{sex_label}.png")
    plt.savefig(file_path, dpi=300)
    plt.close()
    print(f"  → Plot saved to: {file_path}")

    return res

# -------- 5. Run for both sexes --------
res_f = run_and_save(df[df["sex"]=="F"], "Female", save_dir)
res_m = run_and_save(df[df["sex"]=="M"], "Male", save_dir)
