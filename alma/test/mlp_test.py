from alma.neuronet import mlp_net


if __name__ == "__main__":
    mlp = mlp_net.MLP(input_dim=1, output_dim=1, hidden_dim=1)

    sample_inputs = []
    sample_outputs = []
    for i in range(-1, 2):
        x = float(i)
        sample_inputs.append([x])
        sample_outputs.append([x])

    mlp.learn(sample_inputs, sample_outputs, 0.0001, max_iterations=1000)

    print(mlp.get_results([2.5]))
