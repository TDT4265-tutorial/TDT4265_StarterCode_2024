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
    '''
    Y_pred is the predictions, found by doing one forward step
    '''
    Y_pred = model.forward(X)


    '''
    Here, the program finds the indices of the maximum value across the columns for each row with np.argmax, both for
    the predictions and the targets. Then it sums up how many of these are equal
    '''
    predicted_classes = np.argmax(Y_pred, axis=1)
    actual_classes = np.argmax(targets, axis=1)
    Y_correct = np.sum(np.equal(predicted_classes, actual_classes))
    
    
    accuracy = Y_correct/Y_pred.shape[0]

    return accuracy
    

   


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

        outputs= self.model.forward(X_batch)
        loss = cross_entropy_loss(Y_batch, outputs)
        self.model.backward(X_batch, outputs, Y_batch)
        self.model.w -= self.learning_rate * self.model.grad

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

    model1 = SoftmaxModel(l2_reg_lambda=1.0) #Model with lambda = 1
    model2 = SoftmaxModel(l2_reg_lambda=0.0) #Model with lambda = 0
    
    trainer1 = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )   #Trainer for model with lambda = 1
    
    trainer2 = SoftmaxTrainer(
        model2, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )   #Trainer for model with lambda = 0
    
    train_history_reg01, val_history_reg01 = trainer1.train(num_epochs) #Training model with lambda = 1
    train_history_reg02, val_history_reg02 = trainer2.train(num_epochs) #Training model with lambda = 0
    
    # You can finish the rest of task 4 below this point.

    # Plotting of softmax weights (Task 4b)
    
    #Reshaping the weight matrix to 28x28 and removing the bias for each class
    imgWeights1 = model1.w[:-1, :].reshape(28, 28, 10)
    imgWeights2 = model2.w[:-1, :].reshape(28, 28, 10)
    
    #Iterating over each class and plotting the weights for both lambdas
    plt.figure()
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(imgWeights1[:, :, i], cmap="gray")
        plt.title(f"Class {i}")
        plt.axis('off')
    plt.suptitle("Lambda = 1")
    plt.savefig("task4b_softmax_weight_lambda1.png")
    
    plt.figure()
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(imgWeights2[:, :, i], cmap="gray")
        plt.title(f"Class {i}")
        plt.axis('off')
    plt.suptitle("Lambda = 0")
    plt.savefig("task4b_softmax_weight_lambda0.png")
        
    
    # Plotting of accuracy for difference values of lambdas (task 4c)
    l2_lambdas = [1, .1, .01, .001]
    
    #Creating a list to store the accuracy for each lambda
    accuracies = []
    
    #Iterating over each lambda and training a model with that lambda
    for i in l2_lambdas:
        model = SoftmaxModel(i)
        trainer = SoftmaxTrainer(
            model, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        train_history, val_history = trainer.train(num_epochs)
        accuracies.append(val_history["accuracy"])
        
    #Plotting the accuracy figures for each lambda
    
    plt.figure()
    plt.xlim([0, 5500])
    plt.ylim([0.6, .93])
    for i in range(len(l2_lambdas)):
        utils.plot_loss(accuracies[i], label=f"Lambda = {l2_lambdas[i]}")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task4c_l2_reg_accuracy.png")
    plt.show()

    #Task 4e - Plotting of the l2 norm for each weight
    
    

    #plt.savefig("task4d_l2_reg_norms.png")


if __name__ == "__main__":
    main()
