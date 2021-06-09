import numpy as np

class ActivationFunctions:
    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def softmax(x):
        y = np.exp(x - x.max())
        return y / y.sum()


class MLP:
    def __init__(self, inputs_count=2, hidden_layers_counts_with_act_funcs=[
        (2, ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative)],
                 outputs_count_with_act_funcs=[
                     (1, ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative)]):
        self.inputs_count = inputs_count
        self.hidden_layers_counts_with_act_funcs = hidden_layers_counts_with_act_funcs
        self.outputs_counts_with_act_funcs = outputs_count_with_act_funcs

        self.layers_counts_with_funcs = [(self.inputs_count, None, None)] + self.hidden_layers_counts_with_act_funcs \
                                        + self.outputs_counts_with_act_funcs

        self.weights = []
        for i in range(len(self.layers_counts_with_funcs) - 1):
            self.weights.append(np.random.rand(self.layers_counts_with_funcs[i][0],
                                               self.layers_counts_with_funcs[i + 1][0]))

        self.derivatives = []
        for i in range(len(self.layers_counts_with_funcs) - 1):
            self.derivatives.append(np.zeros((self.layers_counts_with_funcs[i][0],
                                              self.layers_counts_with_funcs[i + 1][0])))

        self.act_parameters = []
        for i in range(len(self.layers_counts_with_funcs)):
            self.act_parameters.append(np.zeros(self.layers_counts_with_funcs[i][0]))

    def predict(self, inputs):
        currents_layers_act_parameters = inputs
        self.act_parameters[0] = inputs

        for index, matrix in enumerate(self.weights):
            current_layer_inputs = np.dot(currents_layers_act_parameters, matrix)
            currents_layers_act_parameters = self.layers_counts_with_funcs[index + 1][1](current_layer_inputs)
            self.act_parameters[index + 1] = currents_layers_act_parameters

        return currents_layers_act_parameters

    def fit(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            for j, x in enumerate(X):
                y_obj = y[j]
                output = self.predict(x)
                error = y_obj - output

                for k in reversed(range(len(self.derivatives))):
                    delta = error * self.layers_counts_with_funcs[k + 1][2](self.act_parameters[k + 1])
                    delta_transposed = delta.reshape(delta.shape[0], -1).T
                    current_act_parameters = self.act_parameters[k]
                    current_act_parameters = current_act_parameters.reshape(current_act_parameters.shape[0], -1)
                    self.derivatives[k] = np.dot(current_act_parameters, delta_transposed)

                    error = np.dot(delta, self.weights[k].T)

                for k in range(len(self.weights)):
                    matrix = self.weights[k]
                    derivatives = self.derivatives[k]
                    matrix += derivatives * learning_rate

        return self

