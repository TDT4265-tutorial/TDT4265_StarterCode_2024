import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer, calculate_accuracy


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False
    use_relu = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    
    for i in range(2):
        model.ws[i] = np.random.uniform(-1, 1, size=model.ws[i].shape)
    
    train_history, val_history = trainer.train(num_epochs)
    
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    # Example created for comparing with and without shuffling.
    # For comparison, show all loss/accuracy curves in the same plot
    # YOU CAN DELETE EVERYTHING BELOW!

    use_improved_weight_init = True

    # Train a new model with new parameters
    model_no_shuffle = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_no_shuffle, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_improved_weights, val_history_improved_weights = trainer_shuffle.train(
        num_epochs)

    use_improved_sigmoid = True

    # Train a new model with new parameters
    model_no_shuffle = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_no_shuffle, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_improved_sig, val_history_improved_sig = trainer_shuffle.train(
        num_epochs)
    
    use_momentum = True
    learning_rate = 0.02

    # Train a new model with new parameters
    model_no_shuffle = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init,
        use_relu)
    trainer_shuffle = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_no_shuffle, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_momentum, val_history_momentum = trainer_shuffle.train(
        num_epochs)
    
    # Make 2 plots one with loss and one with accuracy comparing all 4 netwroks
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0.0, 0.99])
    utils.plot_loss(val_history["loss"], "Validation Loss no improvements")
    utils.plot_loss(val_history_improved_weights["loss"], "Validation Loss adding improved weights")
    utils.plot_loss(val_history_improved_sig["loss"], "Validation Loss adding improved sigmoid")
    utils.plot_loss(val_history_momentum["loss"], "Validation Loss adding momentum")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.8, 0.99])
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy no improovements")
    utils.plot_loss(val_history_improved_weights["accuracy"], "Validation Accuracy adding improved weights")
    utils.plot_loss(val_history_improved_sig["accuracy"], "Validation Accuracy adding improved sigmoid")
    utils.plot_loss(val_history_momentum["accuracy"], "Validation Accuracy adding momentum")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3_train_loss.png")
    plt.show()
    

   

if __name__ == "__main__":
    main()
