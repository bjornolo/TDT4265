import numpy as np
import utils
import typing
np.random.seed(1)

def mean_X(X: np.ndarray):
    return np.mean(X)

def std_X(X: np.ndarray):
    return np.std(X)

def sigmoid(Z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-Z))

def sigmoid_dot(Z: np.ndarray) -> np.ndarray:
    ez = np.exp(Z)
    return ez / (ez + 1) ** 2

def sigmoid_improved(Z: np.ndarray) -> np.ndarray:
    return 1.7159 * np.tanh(2/3 * Z)

def sigmoid_improved_dot(Z: np.ndarray) -> np.ndarray:
    return 1.7259 * 2/3 * (1 - (np.tanh(2/3*Z))**2)

def softmax(Z: np.ndarray) -> np.ndarray:
    ez = np.exp(Z)
    return ez / np.sum(ez, axis=1, keepdims=True)

def weight(w_shape, use_improved_weight_init):
    
    if use_improved_weight_init:   
        w_std = 1 / (np.sqrt( w_shape[0]))
        return np.random.normal(scale=w_std, size=w_shape)
    return np.random.uniform(-1, 1, size=w_shape) 

def ReLU(Z: np.ndarray) -> np.ndarray:
    return np.maximum(0,Z)

def ReLU_dot(Z: np.ndarray) -> np.ndarray:
    Z[Z<=0] = 0
    Z[Z>0] = 1
    return Z

def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """

    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    # Image Normalization
    mean=mean_X(X)
    std=std_X(X)
    X=(X-mean)/std
    #Bias Trick
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
    # Cross entropy loss
    cost = targets * np.log(outputs)
    return np.mean(-cost.sum(axis=1))


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
            w =weight(w_shape, self.use_improved_weight_init)
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)

        self.layer_Z=[] #Z
        self.layer_a=[] #a
        Z=np.copy(X)
        a=np.copy(X)
        self.layer_a.append(X) #a0 is X
        self.layer_Z.append(None) #z0 is none
        for layer in range(len(self.ws)):
            w=self.ws[layer]
            Z=np.dot(a,w)
            if layer +1 == len(self.ws):
                a=softmax(Z)
            else:
                if self.use_improved_sigmoid:
                    a=sigmoid_improved(Z)
                elif self.use_relu:
                    a=ReLU(Z)
                else:
                    a=sigmoid(Z)
            self.layer_Z.append(Z)    
            self.layer_a.append(a)
        return a

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
        #self.grads = []
        delta=-(targets-outputs) 
        for layer in reversed(range(len(self.ws))):
            dw=np.zeros_like(self.ws[layer])
            dw=np.dot(self.layer_a[layer].T,delta)/X.shape[0]
            self.grads[layer]=dw
            if layer>0:
                deltadotw=np.dot(delta,self.ws[layer].T)
                if self.use_improved_sigmoid:
                    delta=sigmoid_improved_dot(self.layer_Z[layer])*deltadotw
                elif self.use_relu:
                    delta=ReLU_dot(self.layer_Z[layer])*deltadotw
                else:
                    delta=sigmoid_dot(self.layer_Z[layer])*deltadotw
            #self.grads[layer]=dw
        
        
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


def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 3e-3
    for layer_idx, w in enumerate(model.ws):
        print(layer_idx)
        for i in range(w.shape[0]):
            #print(layer_idx,i)
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

    #gradient_approximation_test(model, X_train, Y_train)


if __name__ == "__main__":
    main()
