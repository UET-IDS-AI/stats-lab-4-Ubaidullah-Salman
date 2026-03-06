""" AI Stats Lab Random Variables and Distributions """

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad


# =========================================================
# QUESTION 1 — CDF Probabilities
# =========================================================

def cdf_probabilities():

    # Analytic values for Exp(1)
    analytic_gt5 = math.exp(-5)
    analytic_lt5 = 1 - math.exp(-5)
    analytic_interval = math.exp(-3) - math.exp(-7)

    # Simulation
    samples = np.random.exponential(scale=1, size=100000)

    simulated_gt5 = np.mean(samples > 5)

    print("Q1 Results")
    print("P(X > 5) analytic:", analytic_gt5)
    print("P(X < 5) analytic:", analytic_lt5)
    print("P(3 < X < 7) analytic:", analytic_interval)
    print("P(X > 5) simulated:", simulated_gt5)

    return analytic_gt5, analytic_lt5, analytic_interval, simulated_gt5


# =========================================================
# QUESTION 2 — PDF Validation and Plot
# =========================================================

def pdf_validation_plot():

    def f(x):
        return 2*x*np.exp(-x**2)

    # Integral from 0 to infinity
    integral_value, _ = quad(f, 0, np.inf)

    is_valid_pdf = abs(integral_value - 1) < 1e-6

    # Plot
    x = np.linspace(0, 3, 400)
    y = f(x)

    plt.figure()
    plt.plot(x, y)
    plt.title("PDF f(x) = 2x e^(-x^2)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()

    print("\nQ2 Results")
    print("Integral value:", integral_value)
    print("Is valid PDF:", is_valid_pdf)

    return integral_value, is_valid_pdf


# =========================================================
# QUESTION 3 — Exponential Distribution
# =========================================================

def exponential_probabilities():

    analytic_gt5 = math.exp(-5)
    analytic_interval = math.exp(-1) - math.exp(-3)

    samples = np.random.exponential(scale=1, size=100000)

    simulated_gt5 = np.mean(samples > 5)
    simulated_interval = np.mean((samples > 1) & (samples < 3))

    print("\nQ3 Results")
    print("P(X > 5) analytic:", analytic_gt5)
    print("P(1 < X < 3) analytic:", analytic_interval)
    print("P(X > 5) simulated:", simulated_gt5)
    print("P(1 < X < 3) simulated:", simulated_interval)

    return analytic_gt5, analytic_interval, simulated_gt5, simulated_interval


# =========================================================
# QUESTION 4 — Gaussian Distribution
# =========================================================

def gaussian_probabilities():

    # analytic
    analytic_le12 = norm.cdf(12, loc=10, scale=2)
    analytic_interval = norm.cdf(12, loc=10, scale=2) - norm.cdf(8, loc=10, scale=2)

    # simulation
    samples = np.random.normal(loc=10, scale=2, size=100000)

    simulated_le12 = np.mean(samples <= 12)
    simulated_interval = np.mean((samples > 8) & (samples < 12))

    print("\nQ4 Results")
    print("P(X ≤ 12) analytic:", analytic_le12)
    print("P(8 < X < 12) analytic:", analytic_interval)
    print("P(X ≤ 12) simulated:", simulated_le12)
    print("P(8 < X < 12) simulated:", simulated_interval)

    return analytic_le12, analytic_interval, simulated_le12, simulated_interval


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    cdf_probabilities()
    pdf_validation_plot()
    exponential_probabilities()
    gaussian_probabilities()
