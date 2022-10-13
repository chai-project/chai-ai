import numpy as np
from scipy import stats

from chai_ai.db_definitions import Profile, SetpointChange


def model_update(profile: Profile, setpoint_change: SetpointChange, price: float) -> Profile:
    """
    Update a profile's model when the setpoint is changed at a given price.

    https://maxhalford.github.io/blog/bayesian-linear-regression/

    :param profile: The current profile, including model parameters.
    :param setpoint_change: The new setpoint input by the user.
    :param price: The price when the user adjusted the setpoint. Note: this should be part of SetpointChange.
    :return: The updated profile, comprised of updated model parameters.
    """
    mean = np.array([profile.mean1, profile.mean2])
    covariance_matrix = np.array([[profile.variance1, profile.correlation1], [profile.correlation2, profile.variance2]])
    inverse_covariance_matrix = np.linalg.inv(covariance_matrix)

    x = np.array([1.0, price])
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
        home=profile.home,
        setpointChange=setpoint_change
    )


def model_summary(profile: Profile, num_std: float = 2) -> tuple:
    """
    Summarise a profile's model by its mean and confidence region, with the latter represented by the width, height,
    and angle of its ellipse.

    https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py

    :param profile: The current profile, including model parameters.
    :param num_std: The number of standard deviations to use for the confidence region. Note: we want to use a
                    confidence level rather than a number of standard deviations.
    :return: The model's mean along with the width, height, and angle of its confidence ellipse.
    """
    mean = profile.mean1, profile.mean2
    covariance_matrix = np.array([[profile.variance1, profile.correlation1], [profile.correlation2, profile.variance2]])

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(covariance_matrix)
    angle = 360 - np.degrees(np.arctan2(*vecs[:, 0][::-1]))  # Angle in degrees, not radians
    width, height = 2 * num_std * np.sqrt(vals)  # Width and height are "full" widths, not radii

    confidence_region = width, height, angle

    return mean, confidence_region


def model_predict(profile: Profile, prices: iter, confidence: float = 0.99) -> iter:
    """
    Apply a profile's model to predict setpoint distributions for an iterator of prices, with each predicted
    distribution summarised by its mean and confidence interval.

    https://maxhalford.github.io/blog/bayesian-linear-regression/

    :param profile: The current profile, including model parameters.
    :param prices: The input prices for predictions.
    :param confidence: The level of the confidence intervals (default: 99% confidence intervals).
    :return: Yields input prices and summarised predictions (means and confidence intervals).
    """
    mean = np.array([profile.mean1, profile.mean2])
    covariance_matrix = np.array([[profile.variance1, profile.correlation1], [profile.correlation2, profile.variance2]])

    for price in prices:
        x = np.array([1.0, price])

        # Obtain the predictive mean (Bishop eq. 3.58)
        predictive_mean = x @ mean

        # Obtain the predictive variance (Bishop eq. 3.59)
        predictive_variance = 1 / profile.noise_precision + x @ covariance_matrix @ x.T

        predictive_confidence_interval = stats.norm(loc=predictive_mean, scale=np.sqrt(predictive_variance)).interval(confidence)

        prediction = predictive_mean, predictive_confidence_interval

        yield price, prediction


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
        home=None
    )

    updated_profile = model_update(init_profile, init_setpoint_change, 17.663)

    assert updated_profile.mean1 == 21.96713444650198
    assert updated_profile.mean2 == -0.10579808810668156
    assert updated_profile.variance1 == 0.8591202228277273
    assert updated_profile.variance2 == 0.005615335425357953
    assert updated_profile.correlation1 == -0.024753783781362747
    assert updated_profile.correlation2 == -0.024753783781362733
    assert updated_profile.noise_precision == init_profile.noise_precision

    assert model_summary(updated_profile) == ((updated_profile.mean1, updated_profile.mean2), (3.709097015326712, 0.2799433209487951, 181.659861488596))

    assert list(model_predict(updated_profile, [0, 17.5, 35])) == [
        (0, (21.96713444650198, (16.907009581503893, 27.027259311500067))),
        (17.5, (20.115667904635053, (14.52402109139302, 25.707314717877086))),
        (35, (18.264201362768127, (10.534506601706743, 25.993896123829508)))
    ]
