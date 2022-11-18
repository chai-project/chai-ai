# pylint: disable=line-too-long, missing-module-docstring, too-few-public-methods, missing-class-docstring
import os
import sys

import click
import numpy as np
import tomli
import tomli_w
from scipy import stats

from chai_ai.db_definitions import Profile, SetpointChange

PCT_TO_STD = [
    0.0, 0.01, 0.02, 0.04, 0.05, 0.06, 0.07, 0.09, 0.1, 0.11, 0.12, 0.14, 0.15, 0.16, 0.17, 0.19, 0.2, 0.21, 0.23,
    0.24, 0.25, 0.26, 0.28, 0.29, 0.3, 0.32, 0.33, 0.34, 0.36, 0.37, 0.38, 0.4, 0.41, 0.42, 0.44, 0.45, 0.47, 0.48,
    0.49, 0.51, 0.52, 0.54, 0.55, 0.57, 0.58, 0.6, 0.61, 0.63, 0.64, 0.66, 0.67, 0.69, 0.7, 0.72, 0.74, 0.75, 0.77,
    0.79, 0.8, 0.82, 0.84, 0.86, 0.87, 0.89, 0.91, 0.93, 0.95, 0.97, 1, 1.01, 1.03, 1.05, 1.08, 1.1, 1.12, 1.14, 1.17,
    1.19, 1.22, 1.24, 1.27, 1.3, 1.32, 1.36, 1.39, 1.43, 1.46, 1.5, 1.54, 1.59, 1.63, 1.67, 1.73, 1.78, 1.85, 1.96,
    2.01, 2.21, 2.58, 2.81, 3.0
]


def model_update(profile: Profile, setpoint_change: SetpointChange) -> Profile:
    """
    Update a profile's model when the setpoint is changed at a given price.

    https://maxhalford.github.io/blog/bayesian-linear-regression/

    :param profile: The current profile, including model parameters.
    :param setpoint_change: The new setpoint input by the user, including price for when the input occurred.
    :return: The updated profile, comprised of updated model parameters.
    """
    mean = np.array([profile.mean1, profile.mean2])
    covariance_matrix = np.array([[profile.variance1, profile.correlation1], [profile.correlation2, profile.variance2]])
    inverse_covariance_matrix = np.linalg.inv(covariance_matrix)

    x = np.array([1.0, setpoint_change.price_at_change])
    y = setpoint_change.temperature

    # Update the inverse covariance matrix (Bishop eq. 3.51)
    updated_inverse_covariance_matrix = inverse_covariance_matrix + profile.noise_precision * np.outer(x, x)

    # Update the mean vector (Bishop eq. 3.50)
    updated_covariance_matrix = np.linalg.inv(updated_inverse_covariance_matrix)
    updated_mean = updated_covariance_matrix @ (inverse_covariance_matrix @ mean + profile.noise_precision * y * x)

    return Profile(
        profile_id=profile.profile_id,
        mean1=updated_mean[0],
        mean2=updated_mean[1],
        variance1=updated_covariance_matrix[0, 0],
        variance2=updated_covariance_matrix[1, 1],
        correlation1=updated_covariance_matrix[0, 1],
        correlation2=updated_covariance_matrix[1, 0],
        noise_precision=profile.noise_precision,
        confidence_region=list(confidence_region(updated_covariance_matrix)),
        prediction_banded=[list(prediction) for prediction in predict(updated_mean, updated_covariance_matrix, profile.noise_precision)],
        home_id=profile.home_id,
        setpoint_id=setpoint_change.id
    )


def confidence_region(covariance_matrix, percentage: int = 95) -> tuple:
    """
    Calculate the width, height, and angle of the confidence region ellipse for a 2D covariance matrix.

    https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py

    :param covariance_matrix: The 2-dimensional covariance matrix.
    :param percentage: The percentage of the desired confidence interval, in [0, 100].
    :return: The model's mean along with the width, height, and angle of its confidence ellipse.
    """
    
    assert 0 <= percentage <= 100
    
    # converting between a percentage and a number of standard deviations is not easy because it relies
    # on a two-sided Gaussian distribution - instead use a table lookup for all values
    num_std: float = PCT_TO_STD[percentage]

    def eigen_sorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigen_sorted(covariance_matrix)
    
    # the angle returned by this function is the "default" angle which starts at the horizontal at 0°
    # and then moves counter-clockwise with each 90° representing a quarter
    
    # the Graph.js library instead sees 0° as the clockwise noon and moves clockwise
    # correct for this using the conversion 90 - angle
    angle = 90 - np.degrees(np.arctan2(*vecs[:, 0][::-1]))  # Angle in degrees, not radians
    width, height = 2 * num_std * np.sqrt(vals)  # Width and height are "full" widths, not radii

    return int(round(angle, 0)), round(height, 2), round(width, 2)


def predict(mean: np.array, covariance_matrix: np.array, noise_precision: float, prices: iter = range(36), confidence: float = 0.99) -> iter:
    """
    Apply a model to predict setpoint distributions for an iterator of prices, with each predicted distribution
    summarised by its mean and confidence interval.

    https://maxhalford.github.io/blog/bayesian-linear-regression/

    :param mean: The model's mean.
    :param covariance_matrix: The model's covariance matrix.
    :param noise_precision: The model's noise precision.
    :param prices: Input prices for the predictions (default: [0, 1, ..., 35]).
    :param confidence: The level of the confidence intervals (default: 99% confidence intervals).
    :return: Yields means and confidence intervals.
    """
    for price in prices:
        x = np.array([1.0, price])

        # Obtain the predictive mean (Bishop eq. 3.58)
        predictive_mean = x @ mean

        # Obtain the predictive variance (Bishop eq. 3.59)
        predictive_variance = 1 / noise_precision + x @ covariance_matrix @ x.T

        predictive_confidence_lower, predictive_confidence_upper = stats.norm(loc=predictive_mean, scale=np.sqrt(predictive_variance)).interval(confidence)

        yield round(predictive_confidence_lower, 2), round(predictive_mean, 2), round(predictive_confidence_upper, 2)


def config_generator(base_config, num_profiles=5):
    """
    Generate a full TOML profile configuration file from a base configuration file containing the minimal set of
    parameters (i.e. mean, covariance matrix, and noise precision). The full configuration file consists of
    n=num_profiles identical profiles. An example base configuration file is as follows:

    [profile]
    mean1             = 23
    mean2             = -0.05
    variance1         = 1
    variance2         = 0.01
    noiseprecision    = 0.1
    correlation1      = 0
    correlation2      = 0

    :param base_config: The location of the base profile configuration file.
    :param num_profiles: The number of profiles to include in the full configuration file.
    :return: The TOML string for the full configuration file.
    """
    full_config = dict()

    if base_config and not os.path.isfile(base_config):
        click.echo("The configuration file is not found. Please provide a valid file path.")
        sys.exit(0)

    if base_config:
        with open(base_config, "rb") as file:
            try:
                base_toml = tomli.load(file)

                mean1 = base_toml["profile"]["mean1"]
                mean2 = base_toml["profile"]["mean2"]
                variance1 = base_toml["profile"]["variance1"]
                variance2 = base_toml["profile"]["variance2"]
                noise_precision = base_toml["profile"]["noiseprecision"]
                correlation1 = base_toml["profile"]["correlation1"]
                correlation2 = base_toml["profile"]["correlation2"]

                mean = np.array([mean1, mean2])
                covariance_matrix = np.array([[variance1, correlation1], [correlation2, variance2]])

                region_angle, region_width, region_height = confidence_region(covariance_matrix)
                prediction_banded = [list(prediction) for prediction in predict(mean, covariance_matrix, noise_precision)]

                profile_configs = {
                    str(i): {  # tomli_w does not support int keys
                        "mean1": mean1,
                        "mean2": mean2,
                        "variance1": variance1,
                        "variance2": variance2,
                        "noiseprecision": noise_precision,
                        "correlation1": correlation1,
                        "correlation2": correlation2,
                        "region_angle": region_angle,
                        "region_width": region_width,
                        "region_height": region_height,
                        "prediction_banded": prediction_banded,
                    } for i in range(1, num_profiles + 1)
                }

                full_config = {
                    "profiles": {
                        "number": num_profiles,
                        **profile_configs
                    }
                }
            except tomli.TOMLDecodeError:
                click.echo("The configuration file is not valid and cannot be parsed.")
                sys.exit(0)

    return tomli_w.dumps(full_config)


def test_correctness():
    init_variance1 = 1
    init_variance2 = 0.01
    init_correlation = 0.001 * np.sqrt(init_variance1) * np.sqrt(init_variance2)

    init_profile = Profile(
        profile_id=None,  # noqa
        mean1=22,
        mean2=-0.1,
        variance1=init_variance1,
        variance2=init_variance2,
        correlation1=init_correlation,
        correlation2=init_correlation,
        noise_precision=0.3333333333333333,
        confidence_region=None,
        prediction_banded=None,
        home=None,  # noqa
        setpointChange=None  # noqa
    )

    init_setpoint_change = SetpointChange(
        changed_at=None,  # noqa
        expires_at=None,  # noqa
        checked=None,  # noqa
        duration=None,  # noqa
        mode=None,  # noqa
        temperature=20.0,
        price_at_change=17.663,
        home=None  # noqa
    )

    updated_profile = model_update(init_profile, init_setpoint_change)

    assert updated_profile.mean1 == 21.96713444650198
    assert updated_profile.mean2 == -0.10579808810668156
    assert updated_profile.variance1 == 0.8591202228277273
    assert updated_profile.variance2 == 0.005615335425357953
    assert updated_profile.correlation1 == -0.024753783781362747
    assert updated_profile.correlation2 == -0.024753783781362733
    assert updated_profile.noise_precision == init_profile.noise_precision
    assert updated_profile.confidence_region == [182, 0.28, 3.71]
    assert updated_profile.prediction_banded[:3] == [
        [16.91, 21.97, 27.03],
        [16.83, 21.86, 26.89],
        [16.75, 21.76, 26.77]
    ]


if __name__ == "__main__":
    # test_correctness()
    pass
