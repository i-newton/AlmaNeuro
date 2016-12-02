import math


def simple(v):
    return v


def simple_classifier(v):
    if v >= 0:
        return 1
    else:
        return 0


def logistic_function(v):
    return 1/(1 + pow(math.e, -v))
