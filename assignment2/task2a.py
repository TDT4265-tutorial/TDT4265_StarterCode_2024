import numpy as np
import utils
import typing
import time
np.random.seed(1)


def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    assert X.shape[1] == 784, f"X.shape[1]: {X.shape[1]}, should be 784"
    # TODO implement this function (Task 2a)
    mean = np.mean(X)
    std = np.std(X)
    printStats = True
    if printStats:
        print("Mean: ", mean)
        print("\nStd: ", std, "\n")
    X = (X-mean)/std
    X = np.c_[X, np.ones(X.shape[0])]
    print(X.shape)
    return X



def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray) -> float:
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, 1]
        outputs: outputs of model of shape: [batch size, 1]
    Returns:
        Cross entropy error (float)
    """
    # TODO implement this function (Task 3a)
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"

    loss = -np.sum(targets * np.log(outputs)) / targets.shape[0]
    return loss

def sigmoid(X: np.ndarray):
    return 1/(1+np.exp(-X))


class SoftmaxModel:

    def __init__(
        self,
        # Number of neurons per layer
        neurons_per_layer: typing.List[int],
        use_improved_sigmoid: bool,  # Task 3b hyperparameter
        use_improved_weight_init: bool,  # Task 3a hyperparameter
        use_relu: bool,  # Task 3c hyperparameter
    ):
        np.random.seed(
            1
        )  # Always reset random seed before weight init to get comparable results.
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
        self.hidden_layer_z = [None for i in range(len(neurons_per_layer) + 1)]
        self.hidden_layer_a = [None for i in range(len(neurons_per_layer) + 1)]
        self.ws = []
        prev = self.I
        print(self.neurons_per_layer)
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)
            if self.use_improved_weight_init:
                w = np.random.normal(0, 1/np.sqrt(prev), w_shape)
            else:
                w = np.random.uniform(-1, 1, w_shape) 
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
        # HINT: For performing the backward pass, you can save intermediate activations in variables in the forward pass.
        # such as self.hidden_layer_output = ...
        #print("Input: ", X.shape)
        
        self.hidden_layer_a[0] = X
        for i, w in enumerate(self.ws):
            self.hidden_layer_z[i+1] = self.hidden_layer_a[i].dot(w)
            if i + 1 == len(self.ws):
                output_e = np.exp(self.hidden_layer_z[i+1])
                sum = np.sum(output_e, axis=1)
                self.hidden_layer_a[i+1] = output_e/sum[:, None]
            else:
                if self.use_improved_sigmoid:
                    self.hidden_layer_a[i+1] = 1.7159 * np.tanh((2/3) * self.hidden_layer_z[i+1])
                else:
                    self.hidden_layer_a[i+1] = sigmoid(self.hidden_layer_z[i+1])

        #print("Output: ", output_a.shape)

        return self.hidden_layer_a[-1]

    def backward(self, X: np.ndarray, outputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert (
            targets.shape == outputs.shape
        ), f"Output shape: {outputs.shape}, targets: {targets.shape}"

        '''
        delta_y = targets - outputs

        wd = delta_y.dot(self.ws[1].transpose())
        if self.use_improved_sigmoid:
            f_derivative = 1.7159 * (2/3) * (1 - np.tanh((2/3) * self.hidden_layer_z)**2)
        else:
            f_derivative = sigmoid(self.hidden_layer_z)*(1-sigmoid(self.hidden_layer_z))
        delta_1 = f_derivative * wd
        #print("SHape: ", wd.shape)
        
        grad_1 = -X.transpose().dot(delta_1)/X.shape[0]

        grad_2 = -self.hidden_layer_a.transpose().dot(delta_y)/X.shape[0]
        #print("Dividing by ", X.shape[0])

        '''
        delta_prev =targets - outputs

        for i, w in reversed(list(enumerate(self.ws))):
            wd = delta_prev.dot(self.ws[i].transpose())
            
            self.grads[i] = -self.hidden_layer_a[i].transpose().dot(delta_prev)/X.shape[0]
            if i > 0:
                if self.use_improved_sigmoid:
                    f_derivative = 1.7159 * (2/3) * (1 - np.tanh((2/3) * self.hidden_layer_z[i])**2)
                else:
                    f_derivative = sigmoid(self.hidden_layer_z)*(1-sigmoid(self.hidden_layer_z[i]))
                delta_prev = f_derivative * wd


        #print("Shape of stuff", grad_1.shape)
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer


        
        for grad, w in zip(self.grads, self.ws):
            assert (
                grad.shape == w.shape
            ), f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."
        

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
    out = np.zeros(shape=(Y.shape[0], num_classes))
    for i in range(Y.shape[0]):
        out[i, Y[i]] = 1
    return out
    # TODO implement this function (Task 3a)
    raise NotImplementedError


def gradient_approximation_test(model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
    Numerical approximation for gradients. Should not be edited.
    Details about this test is given in the appendix in the assignment.
    """

    assert isinstance(X, np.ndarray) and isinstance(
        Y, np.ndarray
    ), f"X and Y should be of type np.ndarray!, got {type(X), type(Y)}"
    print("Gradiant test")
    epsilon = 1e-3
    
    logits = model.forward(X)
    model.backward(X, logits, Y)
    for layer_idx, w in enumerate(model.grads):
        for i in range(w.shape[0]):
            print(i)
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
                
                #print("\nw shape: ", w.shape)
                #print(gradient_approximation)
                #print(model.grads[layer_idx][i, j])
                assert abs(difference) <= epsilon**1, (
                    f"Calculated gradient is incorrect. "
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n"
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n"
                    f"If this test fails there could be errors in your cross entropy loss function, "
                    f"forward function or backward function"
                )
            


def main():
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert (
        Y[0, 3] == 1 and Y.sum() == 1
    ), f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert (
        X_train.shape[1] == 785
    ), f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_relu = True
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init, use_relu
    )

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)


if __name__ == "__main__":
    main()
