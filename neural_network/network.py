import numpy as np

IMAGE_SIZE = 64
RESULT_LAYER_SIZE = 10


class Network:

    def __init__(self, sizes: list, activation_function: callable, activation_derivative: callable) -> None:
        self.layers = len(sizes) + 2
        self.sizes = [IMAGE_SIZE] + sizes + [RESULT_LAYER_SIZE]
        self.weights = self.init_weights()
        self.biases = self.init_biases()
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

    def init_weights(self) -> np.array:
        init_weights = []
        for i in range(len(self.sizes) - 1):
            weight = np.random.uniform(-1/np.sqrt(IMAGE_SIZE), 1/np.sqrt(IMAGE_SIZE), (self.sizes[i+1], self.sizes[i]))
            init_weights.append(weight)
        return init_weights

    def init_biases(self) -> np.array:
        init_biases = []
        for size in self.sizes[1:]:
            bias = np.random.uniform(-1/np.sqrt(IMAGE_SIZE), 1/np.sqrt(IMAGE_SIZE), (size, 1))
            init_biases.append(bias)
        return init_biases

    def forward_propagation(self, cur_pixels: np.array) -> tuple:
        not_activated_zs = []
        activated_zs = []
        activated_zs.append(cur_pixels)
        for bias, weight in zip(self.biases, self.weights):
            not_activated_z = np.dot(weight, activated_zs[-1]) + bias
            not_activated_zs.append(not_activated_z)
            activated_z = self.activation_function(not_activated_z)
            activated_zs.append(activated_z)
        return not_activated_zs, activated_zs

    def back_propagation(self, label: np.array, not_activated_zs: np.array, activated_zs: np.array) -> tuple:
        weight_improv = [np.zeros(weights.shape) for weights in self.weights]
        bias_improv = [np.zeros(biases.shape) for biases in self.biases]
        error = (activated_zs[-1] - label) * self.activation_derivative(not_activated_zs[-1])
        bias_improv[-1] = error
        weight_improv[-1] = np.dot(error, activated_zs[-2].T)
        for i in range(2, self.layers):
            z = not_activated_zs[-i]
            derivative = self.activation_derivative(z)
            error = np.dot(self.weights[-i + 1].T, error) * derivative
            bias_improv[-i] = error
            weight_improv[-i] = np.dot(error, activated_zs[-i-1].T)
        return weight_improv, bias_improv

    def update_parameters(self, improve_weights: np.array, improve_biases: np.array, alpha: float) -> None:
        for index, (imp_w, imp_b) in enumerate(zip(improve_weights, improve_biases)):
            self.weights[index] = self.weights[index] - alpha * imp_w
            self.biases[index] = self.biases[index] - alpha * imp_b

    def train_network(self, epochNum: int, data_x: np.array, data_y: np.array, alpha: float) -> None:
        for i in range(epochNum):
            for image, label in zip(data_x, data_y):
                not_activated_zs, activated_zs = self.forward_propagation(image)
                weight_improv, bias_improv = self.back_propagation(label, not_activated_zs, activated_zs)
                self.update_parameters(weight_improv, bias_improv, alpha)

    def check_accuracy(self, x: np.array, y: np.array) -> float:
        predicted_labels = []
        for image in x:
            not_activated_zs, activated_zs = self.forward_propagation(image)
            last_z = activated_zs[-1]
            predicted_labels.append(last_z)
        predicted_labels = np.argmax(predicted_labels, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.sum(predicted_labels == true_labels)/len(x)
        return accuracy
