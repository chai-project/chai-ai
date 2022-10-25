# pylint: disable=line-too-long, missing-module-docstring, too-few-public-methods, missing-class-docstring
import os
import sys

import click
import numpy as np
import tomli
import tomli_w
from scipy import stats

from chai_ai.db_definitions import Profile, SetpointChange


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


def confidence_region(covariance_matrix, num_std: float = 2) -> tuple:
    """
    Calculate the width, height, and angle of the confidence region ellipse for a 2D covariance matrix.

    https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py

    :param covariance_matrix: The 2-dimensional covariance matrix.
    :param num_std: The number of standard deviations to use for the confidence region. Note: we want to use a
                    confidence level rather than a number of standard deviations.
    :return: The model's mean along with the width, height, and angle of its confidence ellipse.
    """

    def eigen_sorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigen_sorted(covariance_matrix)
    angle = 360 - np.degrees(np.arctan2(*vecs[:, 0][::-1]))  # Angle in degrees, not radians
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
