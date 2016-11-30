from samples.neuronets import linear_classifier

if __name__ == '__main__':
    net = linear_classifier.LinearClassifier(2, 1)
    input_samples = [[1, 1], [0, 0], [2, 2], [3, 3]]
    output_samples = [[1], [0], [2], [3]]
    net.learn(input_samples, output_samples, error_threshold=0.1,
              max_iterations=50)
    print net.get_results([4, 4])

    net2 = linear_classifier.LinearClassifier(1, 1)
    input_samples = [[1], [0], [2], [3]]
    output_samples = [[1], [0], [2], [3]]
    net2.learn(input_samples, output_samples, error_threshold=0.1,
               max_iterations=50)
    print net2.get_results([4])
