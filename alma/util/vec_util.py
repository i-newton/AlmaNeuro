import random


def multiply(vec1, vec2):
    m = 0
    for v1, v2 in zip(vec1, vec2):
        m += v1 * v2
    return m


def get_random_vector_list(vector_dim, num_vectors):
    return [[random.random() for i in range(vector_dim)] for j in range(num_vectors)]