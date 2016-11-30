def multiply(vec1, vec2):
    m = 0
    for v1, v2 in zip(vec1, vec2):
        m += v1 * v2
    return m
