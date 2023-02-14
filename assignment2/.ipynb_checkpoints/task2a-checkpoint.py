import numpy as np
import cupy as cp
import utils
import typing
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    #(X-mean)/standard deviation
    X = X / 255
    mean=np.mean(X)
    std=np.std(X)
    X=(X-mean)/std
    ones= np.ones((X.shape[0], 1))
    X= np.append(X, ones, axis=1)
    return X


def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    # TODO: Implement this function (copy from last assignment)
    """
    cost=-(targets*np.log(outputs)+(1-targets)*np.log(1-outputs))
    cel = np.average(cost)
    return cel
    """
    return -(targets * np.log(outputs)).sum(axis=1).mean().item()



def sigmoid(Z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-Z))

def sigmoid_dot(Z: np.ndarray) -> np.ndarray:
    return np.exp(Z) / (np.exp(Z) + 1) ** 2

def softmax(Z: np.ndarray) -> np.ndarray:
    return np.exp(Z) / np.sum(np.exp(Z), axis=1, keepdims=True)

def sigmoid_improved(Z: np.ndarray) -> np.ndarray:
    return (1.7159 * np.tanh(2/3 * Z))

def sigmoid_improved_dot(Z: np.ndarray) -> np.ndarray:
    return (1.7259 * 2/3) * (1 - (np.tanh(2/3*Z))**2)


class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3b hyperparameter
                 use_improved_weight_init: bool,  # Task 3a hyperparameter
                 use_relu: bool  # Task 4 hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid
        self.use_relu = use_relu
        self.use_improved_weight_init = use_improved_weight_init

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            w = np.zeros(w_shape)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]
        #Z
        self.hidden_layer_input =[np.zeros(size, dtype=float) for size in self.neurons_per_layer[:-1]] 
        #a
        self.hidden_layer_output = [np.zeros(size, dtype=float) for size in self.neurons_per_layer[:-1]] 
        
        
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...
        #use [:-1] since we are not using the bias here
        for layer, w in enumerate(self.ws[:-1]): #layer number adn w for that layer
            #calculate and save Z
            self.hidden_layer_input[layer]=(Z:=np.dot(X,w)) 
            #calcualte and save a
            self.hidden_layer_output[layer]=(X:=sigmoid(Z))
        return softmax(X.dot(self.ws[-1]))

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad
    
        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
       
        """
        delta_k=(outputs-targets)
        #Grad for softmax and output node weight
        self.grads[1]=self.hidden_layer_output[0].T.dot(delta_k)
        #Grad for sigma hidden layer 
        delta_j=delta_k.dot(self.ws[1].T)*sigmoid_dot(self.hidden_layer_input[0])
        self.grads[0]=X.T.dot(delta_j)
        """
        
        delta = (outputs - targets)
        for layer in range(len(self.ws) - 1, 0, -1):
            self.grads[layer] = np.dot(self.hidden_layer_output[layer - 1].T, delta) / outputs.shape[0]
            delta = np.dot(delta, self.ws[layer].T) * sigmoid_dot(self.hidden_layer_input[layer - 1])

        self.grads[0] = np.dot(X.T, delta) / outputs.shape[0]

        
        #print(self.grads[1].shape)
        #print(self.grads[0].shape)
        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."
        
        
        
    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]

def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # TODO: Implement this function (copy from last assignment)
    
    one_hot_Y=[]
    for value in range(Y.shape[0]):
        Y_value= np.zeros(num_classes)
        Y_value[Y[value][0]]=1
        one_hot_Y.append(Y_value)
    one_hot_Y=np.asarray(one_hot_Y)
    return one_hot_Y
    #raise NotImplementedError


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**1,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


def main():
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_relu = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)


if __name__ == "__main__":
    main()
