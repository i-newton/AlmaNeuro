import random

from samples.neuronets import base_net
from utils import vec_utils


class LinearClassifier(base_net.BaseNet):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifier, self).__init__(input_dim, output_dim)
        self.weights = [
            [random.random() for i in range(input_dim + 1)]
            for j in range(output_dim)
        ]

    STEP = 0.1

    def learn(self, input_vecs, result_vecs, error_threshold=0.01,
              max_iterations=1000):
        current_iter = 0
        # for each output classifier we learn all examples
        for output_num in range(self.output_dim):
            while current_iter < max_iterations:
                # for each target vec we need to calculate errors over all
                # input samples
                satisfied = True
                # deltas is mean diff gradient for each weight
                deltas = [0.0 for i in range(self.input_dim + 1)]
                # compare each sample with desired output
                for sample_input, desired_output in zip(
                        input_vecs, result_vecs):
                    # calculate actual output for sample input
                    actual = self.get_output(sample_input, output_num)
                    # compare with desired output and calculate error
                    error = actual - desired_output[output_num]
                    if (error**2)/2 > error_threshold:
                        satisfied = False
                    # add error gradient for this actual output to
                    # overall gradient
                    deltas[0] += error/len(input_vecs)
                    for k in range(1, self.input_dim + 1):
                        deltas[k] += sample_input[k] * error/len(input_vecs)
                if satisfied:
                    break
                else:
                    for weight_num in range(self.input_dim + 1):
                        self.weights[output_num][weight_num] += \
                            self.STEP * deltas[weight_num]

    def get_output(self, input_vec, vec_num):
        return self.weights[vec_num][0] + vec_utils.multiply(
            input_vec, self.weights[vec_num][1:])

    def get_results(self, input_vec):
        return [self.get_output(input_vec, i) for i in range(self.output_dim)]

    def clear(self):
        pass
