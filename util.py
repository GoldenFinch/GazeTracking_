import numpy as np
import matplotlib.pyplot as plt


def angle_calculation_avg(vector1, vector2):
    absolute_vector1 = np.sqrt(np.sum(vector1 * vector1, axis=1))
    absolute_vector2 = np.sqrt(np.sum(vector2 * vector2, axis=1))
    cos_theta = np.sum(vector1 * vector2, axis=1) / (absolute_vector1 * absolute_vector2)
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.degrees(np.arccos(cos_theta))

    return np.sum(theta) / len(vector1)

