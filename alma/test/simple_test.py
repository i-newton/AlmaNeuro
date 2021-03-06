from alma.neuronet import linear

if __name__ == '__main__':
    net = linear.LinearPredictor(2, 1)
    input_samples = [[1, 1], [0, 0], [2, 2], [3, 3]]
    output_samples = [[1], [0], [2], [3]]
    net.learn(input_samples, output_samples, error_threshold=0.00001)
    print(net.get_results([4, 4]))

    net2 = linear.LinearPredictor(1, 1)
    input_samples = []
    output_samples = []
    for i in range(10):
        input_samples.append([i])
        output_samples.append([i-1])
    net2.learn(input_samples, output_samples, error_threshold=0.00001)
    print(net2.get_results([-1]))
