from alma.neuronet import base_net
from alma.net_components import activation_function
from alma.net_components import layer


class MLP(base_net.BaseNet):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(MLP, self).__init__(input_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.hidden_layer = layer.Layer(input_dim, hidden_dim,
                                        activation_func=activation_function.logistic_function)
        self.output_layer = layer.Layer(hidden_dim, output_dim,
                                        activation_func=activation_function.logistic_function)

    def learn(self, input_vecs, result_vecs, error_threshold,
              max_iterations=1000):
        pass

    def get_results(self, input_vec):
        hidden_results = self.hidden_layer.get_result(input_vec)
        return self.output_layer.get_result(hidden_results)

    def clear(self):
        pass
