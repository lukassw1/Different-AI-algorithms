from network import Network, RESULT_LAYER_SIZE, IMAGE_SIZE
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt

TRAIN_SIZE = 0.7
TEST_SIZE = 0.66
VALID_SIZE = 0.66


def nest_arrays(data: np.array, rows: int, columns: int) -> np.array:
    data = [x.reshape(rows, columns) for x in data]
    return data


def make_matrix_by_label(data: np.array) -> np.ndarray:
    new_matrix = np.zeros((data.size, data.max() + 1))
    new_matrix[np.arange(data.size), data] = 1
    new_matrix = nest_arrays(new_matrix, RESULT_LAYER_SIZE, 1)
    return new_matrix


def prepare_my_data() -> tuple:
    my_data = load_digits(return_X_y=True)
    Y = my_data[1]
    X = np.array(my_data[0])
    x_train, x_rem, y_train, y_rem = train_test_split(X, Y, train_size=TRAIN_SIZE)
    x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size=VALID_SIZE)
    x_data = [x_train, x_valid, x_test]
    y_data = [y_train, y_valid, y_test]
    x_data = [nest_arrays(x, IMAGE_SIZE, 1) for x in x_data]
    y_data = [make_matrix_by_label(y) for y in y_data]
    ret = x_data + y_data
    return tuple(ret)


def findBestAlpha(neuron_list, x_train, y_train, x_valid, y_valid):
    Xdata = []
    Ydata = []
    rangeOfSearching = 21  # /100
    for i in range(1, rangeOfSearching):
        alpha = i/100
        print(f"checking alpha {alpha}/{rangeOfSearching/100}")
        Xdata.append(alpha)
        neural = Network(neuron_list, sigmoid_function, sigmoid_derivative)
        neural.train_network(200, x_train, y_train, alpha)
        result = neural.check_accuracy(x_valid, y_valid)
        Ydata.append(result)

    idxOfBestA = Ydata.index(max(Ydata))
    bestAlpha = Xdata[idxOfBestA]
    return bestAlpha


def findBestIter(neuron_list, x_train, y_train, x_valid, y_valid, bestAlpha):
    Xdata = []
    Ydata = []
    rangeOfEpoch = 31  # *10
    for i in range(6, rangeOfEpoch):
        epochNum = i*10
        print(f"checking iterations number {epochNum}/{rangeOfEpoch*10}")
        Xdata.append(epochNum)
        neural = Network(neuron_list, sigmoid_function, sigmoid_derivative)
        neural.train_network(epochNum, x_train, y_train, bestAlpha)
        result = neural.check_accuracy(x_valid, y_valid)
        Ydata.append(result)
    idxBestIter = Ydata.index(max(Ydata))
    bestIter = Xdata[idxBestIter]
    return bestIter


def main():
    x_train, x_valid, x_test, y_train, y_valid, y_test = prepare_my_data()
    neuron_list = [25, 10, 10]
    bestAlpha = findBestAlpha(neuron_list, x_train, y_train, x_valid, y_valid)
    bestEpochNum = findBestIter(neuron_list, x_train, y_train, x_valid, y_valid, bestAlpha)
    neural = Network(neuron_list, sigmoid_function, sigmoid_derivative)
    neural.train_network(bestEpochNum, x_train, y_train, bestAlpha)
    result = neural.check_accuracy(x_test, y_test)
    print(f"Best learning rate: {bestAlpha}")
    print(f"Best iterations number: {bestEpochNum}")
    print(f"Result: {result * 100}%")


def sigmoid_function(x: np.array) -> np.array:
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.array) -> np.array:
    return sigmoid_function(x) * (1 - sigmoid_function(x))


if __name__ == "__main__":
    main()
