from alma.neuronet import base_net


class MLP(base_net.BaseNet):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(MLP, self).__init__(input_dim, output_dim)
        self.hidden_dim = hidden_dim

    def learn(self, input_vecs, result_vecs, error_threshold,
              max_iterations=1000):
        pass

    def get_results(self, input_vec):
        pass

    def clear(self):
        pass
