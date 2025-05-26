from lib import matrix as Matrix
from lib import activation as Activation
from lib import loss as Loss
from lib import color as Color
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import dill as pickle
import os

class FFNN:
    def __init__(self, layer_neurons, X_train, y_train, X_val, y_val, learning_rate, l1_lambda = 0, l2_lambda = 0, activation_functions=None):
        self.layer_neurons = layer_neurons
        self.input = X_train
        self.target = y_train
        self.learning_rate = learning_rate
        self.activations = activation_functions
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.loss_function = Loss.mse

        if activation_functions != None and len(activation_functions) != len(layer_neurons):
            raise ValueError("Number of activation functions must match number of layers")

        # For weight initialization
        self.sizes = [(self.layer_neurons[i] + 1, self.layer_neurons[i + 1]) for i in range(len(self.layer_neurons) - 1)]

        # For validation
        self.X_val = X_val
        self.y_val = y_val
    
    def setLearningRate(self, learning_rate):
        self.learning_rate = learning_rate

    def setLossFunction(self, loss_function):
        self.loss_function = loss_function

    def setActivationUniform(self, activation_function):
        self.activations = [activation_function for i in range(len(self.layer_neurons))]

    def setOutputActivation(self, activation_function):
        self.activations[-1] = activation_function

    def setWeights(self, weights):
        self.weights = weights

    def initializeWeightZeros(self):
        self.weights = [np.zeros(size) for size in self.sizes]

    def initializeWeightRandomUniform(self, lower_bound, upper_bound, seed=None):
        if seed != None:
            np.random.seed(seed)
        self.weights = [np.random.uniform(lower_bound, upper_bound, size) for size in self.sizes]

    def initializeWeightRandomNormal(self, mean, variance, seed=None):
        if seed != None:
            np.random.seed(seed)
        self.weights = [np.random.normal(mean, variance, size) for size in self.sizes]

    def FFNNForwardPropagation(self, current_input):
        self.layer_results = [current_input]
        self.layer_results_before_activation = [current_input]
        input = Matrix.addBiasColumn(current_input)
        i = 1
        for layer in self.weights:
            # Get layer result
            initial_result = np.matmul(input, layer)
            self.layer_results_before_activation.append(initial_result)

            # Apply activation function to result
            activation = self.activations[i]
            result = activation(initial_result)
            self.layer_results.append(result)

            # Change input to result
            biased_result = Matrix.addBiasColumn(result)
            input = biased_result
            i += 1

            # Return if at output layer
            if i > len(self.weights):
                return result

    def FFNNBackPropagation(self, current_target):
        deltas = []
        delta_weights = []
        n = len(self.weights)

        for i in range(n, 0, -1):
            # Calculate delta matrix
            output = self.layer_results[i]                                  # Output (Oj) matrix

            if i == n:
                # Output layer
                if self.activations[i] == Activation.softmax:
                    delta = current_target - output
                else:
                    delta = Loss.getErrorDerivativeMatrix(self.loss_function, current_target, output) * Activation.getDerivativeMatrix(self.activations[i], output)
            else:
                # Hidden layer
                weight_ds = Matrix.removeBiasRow(self.weights[i])           # Downstream weight (Wkj) matrix
                delta_ds = deltas[0]                                        # Downstream delta (delta_k) matrix

                delta = Activation.getDerivativeMatrix(self.activations[i], output) * (np.matmul(delta_ds, np.transpose(weight_ds)))

            deltas = [delta] + deltas

            # Calculate new weights
            layer_input = Matrix.addBiasColumn(self.layer_results[i - 1])   # Input (Xji) matrix
            weight_change = self.learning_rate * (np.matmul(np.transpose(layer_input), delta))      # delta_w (n * delta_j * xji)

            if self.l2_lambda > 0:
                weight_change -= self.learning_rate * self.l2_lambda * self.weights[i - 1]
            if self.l1_lambda > 0:
                weight_change -= self.learning_rate * self.l1_lambda * np.sign(self.weights[i - 1])

            delta_weights = [weight_change] + delta_weights

        # Set current epoch's gradient array 
        self.deltas = deltas

        # Update old weights after backpropagation has finished
        for i, weight_change in enumerate(delta_weights):
            self.weights[i] += weight_change

    def train(self, batch_size, learning_rate, epochs, verbose=True, printResults=False):
        self.setLearningRate(learning_rate)
        self.training_loss_list = []
        self.validation_loss_list = []

        for epoch in range(epochs):
            training_loss = 0
            validation_loss = 0
            
            # Iterate through batches
            for i in range(0, len(self.input), batch_size):
                batch_end = (i + batch_size) if (i + batch_size) < len(self.input) else len(self.input)
                X_batch = self.input[i:batch_end]
                y_batch = self.target[i:batch_end]

                self.FFNNForwardPropagation(X_batch)
                self.FFNNBackPropagation(y_batch)

                training_loss += Loss.mse(y_batch, self.layer_results[-1])

            # Calculate epoch loss
            training_loss /= len(self.input)
            self.training_loss_list.append(training_loss)

            self.FFNNForwardPropagation(self.X_val)
            validation_loss = Loss.mse(self.y_val, self.layer_results[-1])
            self.validation_loss_list.append(validation_loss)

            # Print epoch results
            if verbose:
                progress_bar = Color.progress_bar(epoch + 1, epochs)
                print(Color.YELLOW + f" [Epoch {epoch + 1}]:" + Color.GREEN + f"\tTraining Loss: {training_loss}" + Color.BLUE + f"\tValidation Loss: {validation_loss}" + Color.YELLOW + f'\tProgress: [{progress_bar}]' + Color.RESET)
                if printResults:
                    print(f"{Color.CYAN}     Prediction:\t{self.layer_results[-1]}")
                    print(f"{Color.MAGENTA}     Target:\t\t{self.target}{Color.RESET}")

        return self.training_loss_list, self.validation_loss_list

    def printGraph(self):
        G = nx.DiGraph()
        positions = {}
        node_colors = {}
        node_labels = {}

        layer_spacing = 3
        neuron_spacing = 1.5

        num_layers = len(self.layer_neurons)
        max_neurons = max(self.layer_neurons)

        fig_width = layer_spacing * num_layers * 1.2
        fig_height = max_neurons * neuron_spacing * 1.5

        for layer_idx, num_neurons in enumerate(self.layer_neurons):
            y_start = -(num_neurons - 1) * neuron_spacing / 2

            for i in range(num_neurons):
                if layer_idx == 0:
                    node_name = f"I{i+1}"
                    color = "lightgreen"
                elif layer_idx == num_layers - 1:
                    node_name = f"O{i+1}"
                    color = "yellow"
                else:
                    node_name = f"H{layer_idx}_{i+1}"
                    color = "lightblue"

                G.add_node(node_name)
                positions[node_name] = (layer_idx * layer_spacing, y_start + i * neuron_spacing)
                node_colors[node_name] = color
                node_labels[node_name] = f"{node_name}"

            if layer_idx < num_layers - 1:
                bias_node = f"B{layer_idx+1}"
                G.add_node(bias_node)
                positions[bias_node] = (layer_idx * layer_spacing, y_start - neuron_spacing)
                node_colors[bias_node] = "lightgray"
                node_labels[bias_node] = f"{bias_node}"

        edge_labels = {}
        for layer_idx, weight_matrix in enumerate(self.weights):
            for src in range(len(weight_matrix) - 1):
                for dest in range(weight_matrix.shape[1]):
                    src_node = (
                        f"I{src+1}" if layer_idx == 0 else f"H{layer_idx}_{src+1}"
                    )
                    dest_node = (
                        f"H{layer_idx+1}_{dest+1}" if layer_idx + 1 < num_layers - 1 else f"O{dest+1}"
                    )
                    weight = weight_matrix[src + 1, dest]
                    G.add_edge(src_node, dest_node, weight=weight)
                    edge_labels[(src_node, dest_node)] = f"{weight:.2f}"

            for dest in range(weight_matrix.shape[1]):
                bias_node = f"B{layer_idx+1}"
                dest_node = (
                    f"H{layer_idx+1}_{dest+1}" if layer_idx + 1 < num_layers - 1 else f"O{dest+1}"
                )
                weight = weight_matrix[0, dest]
                G.add_edge(bias_node, dest_node, weight=weight)
                edge_labels[(bias_node, dest_node)] = f"{weight:.2f}"

        plt.figure(figsize=(fig_width, fig_height))
        nx.draw(
            G,
            pos=positions,
            with_labels=True,
            labels=node_labels,
            node_color=[node_colors[n] for n in G.nodes()],
            edge_color="black",
            node_size=2000,
            font_size=10,
            font_weight="bold",
            arrowsize=15,
        )
        
        for node_name, (x, y) in positions.items():
            if node_name.startswith("I") or node_name.startswith("B"):  
                continue 

            if node_name.startswith("H"):
                layer_idx = int(node_name.split("_")[0][1:]) - 1 
            elif node_name.startswith("O"):
                layer_idx = len(self.layer_neurons) - 2

            neuron_idx = int(node_name.split("_")[1]) - 1 if "_" in node_name else int(node_name[1]) - 1

            if layer_idx < len(self.deltas) and neuron_idx < self.deltas[layer_idx].shape[1]:  
                delta_value = self.deltas[layer_idx][0, neuron_idx]  
                plt.text(x, y + 0.25, f"Δ={delta_value:.5f}", fontsize=9, ha="center", color="green") 

        nx.draw_networkx_edge_labels(G, pos=positions, edge_labels=edge_labels, font_size=8, label_pos=0.75)

        plt.title("Feedforward Neural Network Graph", fontsize=14)
        plt.show()


    def plot_weight_distribution(self, layers):

        if not hasattr(self, 'weights'):
            raise ValueError("Bobot belum diinisialisasi")

        for layer in layers:
            if layer < 0 or layer >= len(self.weights):
                raise ValueError(f"Layer {layer} di luar jangkauan. Indeks layer harus antara 0 dan {len(self.weights) - 1}.")
            
            weights = self.weights[layer].flatten()
            plt.figure(figsize=(8, 5))
            plt.hist(weights, bins=30, alpha=0.7, color='b', edgecolor='black')
            plt.xlabel("Nilai Bobot")
            plt.ylabel("Frekuensi")
            plt.title(f"Distribusi Bobot - Layer {layer}")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()

    def plot_gradient_distribution(self, layers):

        if not hasattr(self, 'deltas'):
            raise ValueError("Gradien belum tersedia")

        for layer in layers:
            if layer < 0 or layer >= len(self.deltas):
                raise ValueError(f"Layer {layer} di luar jangkauan. Indeks layer harus antara 0 dan {len(self.deltas) - 1}.")
            
            gradients = self.deltas[layer].flatten()
            plt.figure(figsize=(8, 5))
            plt.hist(gradients, bins=30, alpha=0.7, color='r', edgecolor='black')
            plt.xlabel("Nilai Gradien Bobot")
            plt.ylabel("Frekuensi")
            plt.title(f"Distribusi Gradien Bobot - Layer {layer}")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()

    def save(self, filename):
        try:
            with open(filename, "wb") as f:
                pickle.dump(self, f)
            print(f" File Successfuly Saved to: '{filename}'")
        except Exception as e:
            print(f"Error: {e}")

    @classmethod
    def load(cls, filename):
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' doesn't exist.")
            return None
        
        try:
            with open(filename, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, cls):
                print(f"Successfuly loaded '{filename}'")
                return obj
            else:
                print(f"Error: Loaded object is not an instance of {cls.__name__}")
                return None
        except Exception as e:
            print(f"Error: {e}")
            return None
