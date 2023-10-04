import os
import shutil
import unittest

import numpy as np
import pymc3 as pm
import pytest

from tess_atlas.data.tic_entry import TICEntry

DEPTH = "depth"
DURATION = "dur"
RADIUS_RATIO = "r"
TIME_START = "tmin"
TIME_END = "tmax"
ORBITAL_PERIOD = "p"
MEAN_FLUX = "f0"
LC_JITTER = "jitter"
GP_RHO = "rho"
GP_SIGMA = "sigma"
RHO_CIRC = "rho_circ"  # stellar density at e=0
LIMB_DARKENING_PARAM = "u"
IMPACT_PARAM = "b"

DATA_DIR = "test_data/toi_103_files"

CLEAR_AFTER_TEST = False


@pytest.fixture
def tic_entry():
    return TICEntry.load(toi=103)


def __build_model(tic_entry):
    with pm.Model() as my_planet_transit_model:
        ## define planet parameters

        # 1) d: transit duration (duration of eclipse)
        d_priors = pm.Bound(pm.Lognormal, lower=0, upper=2)(
            name=DURATION,
            mu=2,
            sigma=np.log(1.2),
            shape=2,
            testval=0,
        )

        # 2) r: radius ratio (planet radius / star radius)
        r_priors = pm.Lognormal(
            name=RADIUS_RATIO, mu=0.5 * np.log(1 * 1e-3), sd=1.0, shape=1
        )
        # 3) b: impact parameter
        b_priors = pm.Lognormal(
            name=IMPACT_PARAM, mu=0, sd=1, shape=1, testval=0
        )
        planet_priors = [r_priors, d_priors, b_priors]

        ## define orbit-timing parameters

        # 1) tmin: the time of the first transit in data (a reference time)
        tmin_norm = pm.Bound(pm.Normal, lower=1, upper=2)
        tmin_priors = tmin_norm(
            TIME_START, mu=1, sigma=0.5 * 2, shape=1, testval=1
        )

        # 2) period: the planets' orbital period
        p_params, p_priors_list, tmax_priors_list = [], [], []
        for n, planet in enumerate(tic_entry.candidates):
            # if only one transit in data we use the period
            if planet.has_data_only_for_single_transit:
                p_prior = pm.Pareto(
                    name=f"{ORBITAL_PERIOD}_{planet.index}",
                    m=planet.period_min,
                    alpha=2.0 / 3.0,
                    testval=planet.period,
                )
                p_param = p_prior
                tmax_prior = planet.tmin
            # if more than one transit in data we use a second time reference (tmax)
            else:
                tmax_norm = pm.Bound(
                    pm.Normal,
                    lower=planet.tmax - planet.duration_max,
                    upper=planet.tmax + planet.duration_max,
                )
                tmax_prior = tmax_norm(
                    name=f"{TIME_END}_{planet.index}",
                    mu=planet.tmax,
                    sigma=0.5 * planet.duration,
                    testval=planet.tmax,
                )
                p_prior = (tmax_prior - tmin_priors[n]) / planet.num_periods
                p_param = tmax_prior

            p_params.append(p_param)  # the param needed to calculate p
            p_priors_list.append(p_prior)
            tmax_priors_list.append(tmax_prior)

        # p_priors = pm.Deterministic(ORBITAL_PERIOD, tt.stack(p_priors_list))
        # tmax_priors = pm.Deterministic(TIME_END, tt.stack(tmax_priors_list))

        ## define stellar parameters

        # 1) f0: the mean flux from the star
        f0_prior = pm.Normal(name=MEAN_FLUX, mu=0.0, sd=10.0)

        # 2) u1, u2: limb darkening parameters
        u0_prior = pm.Normal("u[0]", mu=0.3, sd=0.2, shape=2)
        u1_prior = pm.Normal("u[1]", mu=0.3, sd=0.2, shape=2)
        stellar_priors = [f0_prior, u1_prior, u0_prior]

        ## define k(t, t1; parameters)
        jitter_prior = pm.InverseGamma(name=LC_JITTER, alpha=3.0, beta=2 * 3.0)
        sigma_prior = pm.InverseGamma(name=GP_SIGMA, alpha=3.0, beta=2 * 3.0)
        rho_prior = pm.InverseGamma(name=GP_RHO, alpha=3.0, beta=2 * 3.0)
        lightcurve_models = tic_entry.lightcurve.flux
        my_planet_transit_model.lightcurve_models = (
            lightcurve_models + f0_prior
        )

    return my_planet_transit_model


@pytest.mark.skip(
    reason="Fails at initial_lightcurves=generate_model_lightcurve"
)
def test_static_phase_plot(tic_entry):
    from tess_atlas.plotting import plot_phase

    model = __build_model(tic_entry)
    plot_phase(
        tic_entry=tic_entry,
        model=model,
        initial_params={
            "b": [0.5],
            "dur": [0.1],
            "f0": 1,
            "jitter": 0.1,
            "r": [0.6],
            "rho": 0.4,
            "sigma": 0.5,
            "tmax_1": 1,
            "tmin": 0,
            "u[0]": 0.1,
            "u[1]": 0.2,
        },
    )


def test_histogram_plot():
    from tess_atlas.plotting.histogram_plotter import __plot_histograms

    trues = dict(a=0, b=5, c=2, d=-1)
    n = 5000
    samples = dict(
        a=np.random.uniform(low=trues["a"] - 1, high=trues["a"] + 1, size=n),
        b=np.random.normal(loc=trues["b"], scale=10, size=n),
        c=np.random.pareto(a=trues["c"], size=n),
        d=np.random.standard_cauchy(size=n),
    )
    samples_table = {
        "Type 1": dict(a=samples["a"]),
        "Type 2": dict(b=samples["b"], c=samples["c"]),
        "Type 3": dict(d=samples["d"]),
    }
    __plot_histograms(
        samples_table=samples_table,
        trues=trues,
    )

    latex_label = dict(a=r"$\alpha$", b=r"$\beta$", c=r"$c$", d="d")
    __plot_histograms(
        samples_table=samples_table,
        latex_label=latex_label,
    )
