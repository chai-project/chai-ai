import numpy as np
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

    return angle, height, width


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

        yield predictive_confidence_lower, predictive_mean, predictive_confidence_upper


if __name__ == "__main__":
    init_variance1 = 1
    init_variance2 = 0.01
    init_correlation = 0.001 * np.sqrt(init_variance1) * np.sqrt(init_variance2)

    init_profile = Profile(
        profile_id=None,
        mean1=22,
        mean2=-0.1,
        variance1=init_variance1,
        variance2=init_variance2,
        correlation1=init_correlation,
        correlation2=init_correlation,
        noise_precision=0.3333333333333333,
        confidence_region=None,
        prediction_banded=None,
        home=None,
        setpointChange=None
    )

    init_setpoint_change = SetpointChange(
        changed_at=None,
        expires_at=None,
        checked=None,
        duration=None,
        mode=None,
        temperature=20.0,
        price_at_change=17.663,
        home=None
    )

    updated_profile = model_update(init_profile, init_setpoint_change)

    assert updated_profile.mean1 == 21.96713444650198
    assert updated_profile.mean2 == -0.10579808810668156
    assert updated_profile.variance1 == 0.8591202228277273
    assert updated_profile.variance2 == 0.005615335425357953
    assert updated_profile.correlation1 == -0.024753783781362747
    assert updated_profile.correlation2 == -0.024753783781362733
    assert updated_profile.noise_precision == init_profile.noise_precision
    assert updated_profile.confidence_region == [181.659861488596, 0.2799433209487951, 3.709097015326712]
    assert updated_profile.prediction_banded[:3] == [
        [16.907009581503893, 21.96713444650198, 27.027259311500067],
        [16.830069795435982, 21.8613363583953, 26.892602921354616],
        [16.74585393436608, 21.755538270288618, 26.765222606211154]
    ]
