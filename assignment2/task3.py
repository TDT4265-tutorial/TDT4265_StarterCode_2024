import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import cross_entropy_loss, SoftmaxModel, one_hot_encode, pre_process_images
from trainer import BaseTrainer

np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model) -> float:
    """
    Calculate the accuracy of the model's predictions against the true targets.
    
    Args:
        X: images of shape [batch size, 785]
        targets: one-hot encoded labels/targets of each image of shape: [batch size, 10]
        model: an instance of SoftmaxModel
    
    Returns:
        Accuracy as a float
    """
    predictions = model.forward(X)
    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = np.argmax(targets, axis=1)
    correct_predictions = predicted_classes == actual_classes
    accuracy = np.mean(correct_predictions)
    print("accuracy", accuracy)
    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def __init__(
        self,
        momentum_gamma: float,
        use_momentum: bool,  # Task 3d hyperparmeter
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.momentum_gamma = momentum_gamma
        self.use_momentum = use_momentum
        # Init a history of previous gradients to use for implementing momentum
        self.previous_grads = [np.zeros_like(w) for w in self.model.ws]

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # TODO: Implement this function (task 2c)

        logits = self.model.forward(X_batch)
        loss = cross_entropy_loss(Y_batch, logits)
        self.model.backward(X_batch, logits, Y_batch)

        for layer_idx, grad in enumerate(self.model.grads):
            if self.use_momentum:
                self.previous_grads[layer_idx] = self.momentum_gamma * self.previous_grads[layer_idx] + grad
                self.model.ws[layer_idx] -= self.learning_rate*self.previous_grads[layer_idx]
            else:
                self.model.ws[layer_idx] -= self.learning_rate * grad

        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits=self.model.forward(self.X_val)
        loss=cross_entropy_loss(self.Y_val, logits)

        accuracy_train=calculate_accuracy(
            self.X_train, self.Y_train, self.model)
        accuracy_val=calculate_accuracy(self.X_val, self.Y_val, self.model)
        return loss, accuracy_train, accuracy_val


def create_trainers():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    # No tricks of the trade
    model_0 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_0 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_0, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )

    use_improved_sigmoid = False
    use_improved_weight_init = True
    use_momentum = False

    # Improved weights
    model_1 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_1 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_1, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = False

    # Improved sigmoid
    model_2 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_2 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_2, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    learning_rate=0.02

    # Momentum
    model_3 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_3 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_3, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )

    return [trainer_0, trainer_1, trainer_2, trainer_3]



def main():
    num_epochs = 50

    trainers = create_trainers()

    train_histories = []
    val_histories = []
    number = 0
    for trainer in trainers:
        number += 1
        print("Starting training model #", number, " of ", len(trainers))
        train_history, val_history = trainer.train(num_epochs)
        train_histories.append(train_history)
        val_histories.append(val_history)

    labels = ["Basic", "Improved weigths", "Improved sigmoid", "Momentum"]

    plt.subplot(1, 2, 1)
    for i in range(len(trainers)):
        utils.plot_loss(train_histories[i]["loss"], labels[i], npoints_to_average=10)
    plt.ylim([0, .4])
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")

    plt.subplot(1, 2, 2)
    plt.ylim([0.85, .95])
    for i in range(len(trainers)):
        utils.plot_loss(val_histories[i]["accuracy"], labels[i])
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task2c_train_loss.png")
    plt.show()


if __name__ == "__main__":
    main()
