import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (task 3c)
    outputs = model.forward(X=X)

    # Convert probabilities to class predictions
    predictions = np.argmax(outputs, axis=1)
    true_labels = np.argmax(targets, axis=1)

    correct_guesses = np.sum(predictions == true_labels)
    accuracy = correct_guesses / len(targets)

    return accuracy


def visualize_weights(model, title):
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i, ax in enumerate(axes):
        weight = model.w[:-1, i].reshape(28, 28)
        ax.imshow(weight, cmap='viridis')
        ax.set_title(f'Digit {i}')
        ax.axis('off')
    plt.suptitle(title)
    plt.show()


class SoftmaxTrainer(BaseTrainer):

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
        # TODO: Implement this function (task 3b)
        # Get predictions / outputs
        outputs = self.model.forward(X_batch)

        # Get the loss of this iteration (improvement)
        loss = cross_entropy_loss(targets=Y_batch, outputs=outputs)

        # Update the gradient to get weights that will increase the loss
        self.model.backward(X=X_batch, outputs=outputs, targets=Y_batch)

        # Update the weights to be the opposite of the gradient, times a scalar learning rate
        self.model.w += -self.model.grad * self.learning_rate

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
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(self.Y_val, logits)

        accuracy_train = calculate_accuracy(
            self.X_train, self.Y_train, self.model)
        accuracy_val = calculate_accuracy(
            self.X_val, self.Y_val, self.model)
        return loss, accuracy_train, accuracy_val


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)
    print(f"Training with λ = {l2_reg_lambda}")

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([0.2, .8])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    # Train a model with L2 regularization (task 4b)

    model1 = SoftmaxModel(l2_reg_lambda=1.0)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    # You can finish the rest of task 4 below this point.

    # Plotting of softmax weights (Task 4b)
    # plt.imsave("task4b_softmax_weight.png", weight, cmap="gray")

    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [1, .1, .01, .001]
    l2_norms = []
    validation_accuracies = {l2: [] for l2 in l2_lambdas}

    for l2_reg_lambda in l2_lambdas:
        print(f"Training with λ = {l2_reg_lambda}")

        model = SoftmaxModel(l2_reg_lambda)

        trainer = SoftmaxTrainer(
            model, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )

        train_history, val_history = trainer.train(num_epochs)
        l2_norm = np.linalg.norm(model.w)
        l2_norms.append(l2_norm)
        validation_accuracies[l2_reg_lambda] = val_history['accuracy']
        visualize_weights(
            model, f"Weights with L2 Regularization (λ = {l2_reg_lambda})")

        utils.plot_loss(
            val_history["accuracy"], f"Validation Accuracy for lambda value: {l2_reg_lambda}")

        print("Final Train Cross Entropy Loss:",
              cross_entropy_loss(Y_train, model.forward(X_train)))
        print("Final Validation Cross Entropy Loss:",
              cross_entropy_loss(Y_val, model.forward(X_val)))
        print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
        print("Final Validation accuracy:",
              calculate_accuracy(X_val, Y_val, model))

        # Task 4e, compute the L2 norm of the weights

    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4c_softmax_train_accuracy.png")
    plt.show()

    plt.figure()
    plt.plot(l2_lambdas, l2_norms, marker='o')
    plt.xscale('log')
    plt.xlabel('λ value')
    plt.ylabel('L2 norm of weights')
    plt.title('L2 Norm of Weights for different λ values')
    plt.savefig("task4c_l2_norm_vs_lambda.png")
    plt.show()


if __name__ == "__main__":
    main()
