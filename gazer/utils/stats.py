""" Miscellaneous statistical utility functions.
"""
from scipy.stats import uniform

def _uniform(a, b):
    return uniform(a, b-a)