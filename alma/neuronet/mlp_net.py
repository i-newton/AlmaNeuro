from alma.neuronet import base_net
from alma.net_components import activation_function
from alma.net_components import layer
from alma.util import vec_util


class MLP(base_net.BaseNet):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(MLP, self).__init__(input_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.hidden_layer = layer.Layer(input_dim, hidden_dim,
                                        activation_func=activation_function.logistic_function)
        self.output_layer = layer.Layer(hidden_dim, output_dim,
                                        activation_func=activation_function.logistic_function)

    LEARNING_RATE = 0.1

    def learn(self, input_vecs, result_vecs, error_threshold,
              max_iterations=1000):
        learning_rate = self.LEARNING_RATE
        for _ in range(max_iterations):
            error_rate_succeeded = True
            b_gradients = vec_util.get_matrix(self.output_layer.neuron_num,
                                              self.output_layer.weight_num)
            a_gradients = vec_util.get_matrix(self.hidden_layer.neuron_num,
                                              self.hidden_layer.weight_num)
            sum_error = 0.0
            for sample_input, sample_output in zip(input_vecs, result_vecs):
                hidden_output = self.hidden_layer.get_result(sample_input)
                net_output = self.output_layer.get_result(hidden_output)
                # calculate p
                p = []
                for i in range(len(net_output)):
                    z = net_output[i]
                    error = z - sample_output[i]
                    sum_error += (error**2)/2
                    if error**2 > error_threshold:
                        error_rate_succeeded = False
                    p.append(error * z * (1 - z))
                # calculate b gradients
                b_sample_grads = []
                for i in range(len(net_output)):
                    bi_grad = [p[i]]
                    for y in hidden_output:
                        bi_grad.append(p[i]*y)
                    b_sample_grads.append(bi_grad)

                # calculate q
                q = []
                for i in range(self.hidden_dim):
                    out = self.output_layer.get_weights_for_input(i + 1)
                    t = vec_util.vector_scalar_mult(p, out)
                    t = t*hidden_output[i]*(1 - hidden_output[i])
                    q.append(t)

                a_sample_grads = []
                # calculate a gradients
                for i in range(self.hidden_dim):
                    ai_grad = [q[i]]
                    for x in sample_input:
                        ai_grad.append(q[i]*x)
                    a_sample_grads.append(ai_grad)

                # add sample gradients to epoch
                vec_util.matrix_add(b_gradients, b_sample_grads)
                vec_util.matrix_add(a_gradients, a_sample_grads)

            print(str(_) + '=' + str(sum_error))

            if error_rate_succeeded:
                break
            else:
                vec_util.matrix_mult_number(a_gradients,
                                            -learning_rate/len(input_vecs))
                vec_util.matrix_mult_number(b_gradients,
                                            -learning_rate/len(input_vecs))
                self.hidden_layer.add_gradient(a_gradients)
                self.output_layer.add_gradient(b_gradients)

    def get_results(self, input_vec):
        hidden_results = self.hidden_layer.get_result(input_vec)
        return self.output_layer.get_result(hidden_results)

    def clear(self):
        pass
