from NeuralNetwork import StandardModel


def main():
    model = StandardModel('inception', (512, 512, 3))
    model.build_binary_model().summary()


if __name__ == '__main__':
    main()
