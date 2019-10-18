from NeuralNetwork import MyModel
from utilities import utils, Output


def main():
    model = MyModel('inception', (512, 512, 3))
    model.build_binary_model().summary()


if __name__ == '__main__':
    main()
