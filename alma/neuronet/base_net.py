class BaseNet:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def learn(self, input_vecs, result_vecs, error_threshold,
              max_iterations=1000):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()

    def get_results(self, input_vec):
        raise NotImplementedError()
