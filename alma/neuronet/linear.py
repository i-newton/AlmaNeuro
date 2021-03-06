from alma.net_components import activation_function

from alma.util import vec_util
from alma.net_components import layer
from alma.neuronet import base_net


class LinearPredictor(base_net.BaseNet):
    def __init__(self, input_dim, output_dim):
        super(LinearPredictor, self).__init__(input_dim, output_dim)
        self.layer = layer.Layer(input_num=input_dim, neuron_num=output_dim)

    LEARNING_RATE = 0.01

    def learn(self, input_vecs, result_vecs, error_threshold=0.01,
              max_iterations=10000):
        # for each output classifier we learn all examples
        for neuron_num in range(self.output_dim):
            for _ in range(max_iterations):
                # for each target vec we need to calculate errors over all
                # input samples
                satisfied = True
                # deltas is mean diff gradient for each weight
                grads = vec_util.get_vector(self.layer.weight_num)
                # compare each sample with desired output
                for sample_input, desired_output in zip(
                        input_vecs, result_vecs):
                    # calculate actual output for sample input
                    actual = self.layer.get_result_for_neuron(
                        sample_input, neuron_num)
                    # compare with desired output and calculate error
                    error = desired_output[neuron_num] - actual
                    if (error*error)/2 > error_threshold:
                        satisfied = False
                    # add error gradient for this actual output to
                    # overall gradient
                    for k, sample in enumerate([1] + sample_input):
                        grads[k] += sample * error
                if satisfied:
                    break
                else:
                    for weight_num in range(self.layer.weight_num):
                        self.layer.add(neuron_num, weight_num,
                                       (self.LEARNING_RATE * grads[weight_num]) /
                                       len(input_vecs))

    def get_results(self, input_vec):
        return self.layer.get_result(input_vec)

    def clear(self):
        pass


class LinearClassifier(LinearPredictor):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifier, self).__init__(input_dim, output_dim)
        self.layer.activation_func = activation_function.simple_classifier
