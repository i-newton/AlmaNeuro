from alma.net_components import activation_function
from alma.util import vec_util

"""Package for neuro net net_components."""


class Layer:
    def __init__(self, input_num, neuron_num, activation_func=activation_function.simple):
        self.neuron_num = neuron_num
        self.weight_num = input_num + 1
        self.activation_func = activation_func
        self._weights = vec_util.get_matrix(self.neuron_num, self.weight_num)

    def get_result(self, input_vec):
        return [self.get_result_for_neuron(input_vec, i)
                for i in range(self.neuron_num)]

    def get_result_for_neuron(self, input_vec, neuron_no):
        return self.activation_func(
            vec_util.vector_scalar_mult([1] + input_vec,
                                        self._weights[neuron_no]))

    def get_weight(self, neuron, weight_no):
        return self._weights[neuron][weight_no]

    def set_weight(self, neuron, weight_no, value):
        self._weights[neuron][weight_no] = value

    def add(self, neuron, weight_no, value):
        self._weights[neuron][weight_no] += value

    def add_weigth_vec(self, neuron, vec):
        vec_util.vector_add(self._weights[neuron], vec)

    def add_gradient(self, gradient):
        vec_util.matrix_add(self._weights, gradient)

    def get_weights_for_neuron(self, neuron_no):
        return self._weights[neuron_no]

    def get_weights_for_input(self, input_no):
        return [w[input_no] for w in self._weights]
