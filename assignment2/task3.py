import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer

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

    for trainer in trainers:
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
