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
        for _ in range(max_iterations):
            error_rate_succeeded = True
            for input_vec, target_result in zip(input_vecs, result_vecs):
                # calculate b gradients
                actual_result = self.get_results(input_vec)
                p = []
                for i in range(len(actual_result)):
                    if (actual_result[i] - target_result[i])**2 > error_threshold:
                        error_rate_succeeded = False
                    z = actual_result[i]
                    p.append((z - target_result[i]) * z * (1 - z))

                hidden_result = self.hidden_layer.get_result(input_vec)
                b_grads = []
                # calculate input gradients
                for i in range(len(actual_result)):
                    i_grad = [p[i]]
                    for y in hidden_result:
                        i_grad.append(p[i]*y)
                    b_grads.append(i_grad)


                # if error exceeds some value then we shouldn't stop the algorithm
                # calculate gradient value for each of weight and add this to overall result
                # calculate batch gradient for each value and add this to weights
            if error_rate_succeeded:
                break

    def get_results(self, input_vec):
        hidden_results = self.hidden_layer.get_result(input_vec)
        return self.output_layer.get_result(hidden_results)

    def clear(self):
        pass
