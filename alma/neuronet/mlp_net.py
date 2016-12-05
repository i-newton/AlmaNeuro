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

    LEARNING_RATE = 0.01

    def learn(self, input_vecs, result_vecs, error_threshold,
              max_iterations=1000):
        for _ in range(max_iterations):
            error_rate_succeeded = True
            b_grad_res = []
            a_grad_res = []
            for input_vec, target_result in zip(input_vecs, result_vecs):
                # calculate p
                actual_result = self.get_results(input_vec)
                p = []
                for i in range(len(actual_result)):
                    if (actual_result[i] - target_result[i])**2 > error_threshold:
                        error_rate_succeeded = False
                    z = actual_result[i]
                    p.append((z - target_result[i]) * z * (1 - z))

                hidden_result = self.hidden_layer.get_result(input_vec)
                b_grads = []
                # calculate b gradients
                for i in range(len(actual_result)):
                    bi_grad = [p[i]]
                    for y in hidden_result:
                        bi_grad.append(p[i]*y)
                    b_grads.append(bi_grad)
                if b_grad_res:
                    vec_util.sum(b_grad_res, b_grads)
                else:
                    b_grad_res = b_grads
                # calculate q
                q = []
                for i in range(self.hidden_dim):
                    out = self.output_layer.get_inputs_from_neuron(i)
                    a = vec_util.multiply(q, out[1:])
                    a = a*hidden_result[i]*(1 - hidden_result[i])
                    q.append(a)

                a_grads = []
                # calculate a gradients
                for i in range(self.hidden_dim):
                    ai_grad = [q[i]]
                    for x in input_vec:
                        ai_grad.append(q[i]*x)
                    a_grads.append(ai_grad)

                # add gradients to result gradients in epoch
                if a_grad_res:
                    vec_util.sum(a_grad_res, a_grads)
                else:
                    a_grad_res = a_grads

            if error_rate_succeeded:
                break
            else:
                # increase weigths according to learning weight
                pass

    def get_results(self, input_vec):
        hidden_results = self.hidden_layer.get_result(input_vec)
        return self.output_layer.get_result(hidden_results)

    def clear(self):
        pass
