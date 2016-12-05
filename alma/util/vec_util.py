import random


def multiply(vec1, vec2):
    m = 0
    for v1, v2 in zip(vec1, vec2):
        m += v1 * v2
    return m


def sum(vec1, vec2):
    for i in range(vec1):
        for j in range(vec1[0]):
            vec1[i][j] += vec2[i][j]


def get_random_vector_list(vector_dim, num_vectors):
    return [[random.random() for i in range(vector_dim)] for j in range(num_vectors)]


def get_empty_vector_list(vector_dim, num_vectors):
    return [[0.0 for i in range(vector_dim)] for j in range(num_vectors)]
