import random

from alma.common import exception


def vector_scalar_mult(vec1, vec2):
    if len(vec1) != len(vec2):
        raise exception.SizeMustBeEqual(size1=len(vec1),
                                        size2=len(vec2))
    m = 0
    for v1, v2 in zip(vec1, vec2):
        m += v1 * v2
    return m


def vector_sum(vec1, vec2):
    if len(vec1) != len(vec2):
        raise exception.SizeMustBeEqual(size1=len(vec1),
                                        size2=len(vec2))
    result = []
    for v1, v2 in zip(vec1, vec2):
        result.append(v1 + v2)


def vector_add(vec1, vec2):
    if len(vec1) != len(vec2):
        raise exception.SizeMustBeEqual(size1=len(vec1),
                                        size2=len(vec2))
    for i in range(len(vec1)):
        vec1[i] += vec2[i]


def vector_mult_number(vec, num):
    for i in range(len(vec)):
        vec[i] *= num


def matrix_add(m1, m2):
    if len(m1) != len(m2):
        raise exception.SizeMustBeEqual(size1=len(m1),
                                        size2=len(m2))
    for i in range(m1):
        for j in range(m1[0]):
            m1[i][j] += m2[i][j]


def matrix_sum(m1, m2):
    if len(m1) != len(m2):
        raise exception.SizeMustBeEqual(size1=len(m1),
                                        size2=len(m2))
    res = []
    for i in range(m1):
        res.append(vector_sum(m1[i], m2[i]))
    return res


def matrix_mult_number(m, num):
    for v in m:
        vector_mult_number(v, num)


def get_vector(size, randomize=False):
    if randomize:
        return [random.random() for _ in range(size)]
    else:
        return [0.0] * size


def get_matrix(rows, columns, randomize=False):
    return [get_vector(columns, randomize) for _ in range(rows)]
