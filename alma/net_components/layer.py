from alma.net_components import activation_function
from alma.util import vec_util

"""Package for neuro net net_components."""


class Layer:
    def __init__(self, input_num, neuron_num, activation_func=activation_function.simple):
        self.neuron_num = neuron_num
        self.weight_dim = input_num + 1
        self.activation_func = activation_func
        self._weights = vec_util.get_random_vector_list(
            self.weight_dim, self.neuron_num
        )

    def get_result(self, input_vec):
        return [self.get_result_for_neuron(input_vec, i)
                for i in range(self.neuron_num)]

    def get_result_for_neuron(self, input_vec, neuron_no):
        return self.activation_func(
            vec_util.multiply([1] + input_vec, self._weights[neuron_no]))

    def get_weight(self, neuron, weight_no):
        return self._weights[neuron][weight_no]

    def set_weight(self, neuron, weight_no, value):
        self._weights[neuron][weight_no] = value

    def add(self, neuron, weight_no, value):
        self._weights[neuron][weight_no] += value

    def add_weigths(self, neuron, vec):
        for i in range(self.weight_dim):
            self._weights[neuron][i] += vec[i]

    def add_gradient(self, gradient):
        for i in range(self.neuron_num):
            for j in range(self.weight_dim):
                self._weights[i][j] += gradient[i][j]

    def get_inputs_for_neuron(self, neuron):
        return self._weights[neuron]

    def get_inputs_from_neuron(self, neuron):
        return [w[neuron] for w in self._weights]
