import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer


class ExampleModel(nn.Module):
    def __init__(self, image_channels, num_classes, kernel_size=5, start_filters=32, mid_filters=64, drop_out=False):
        """
        Is called when model is initialized.
        Args:
            image_channels. Number of color channels in image (3)
            num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)
        num_filters_1 = start_filters # Set number of filters in first conv layer
        num_filters_2 = num_filters_1 * 2
        num_filters_3 = num_filters_2 * 2
        kernel_size = kernel_size
        padding_size = kernel_size // 2
        self.num_classes = num_classes
        # Define the convolutional layers
        if drop_out:
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(
                    in_channels=image_channels,
                    out_channels=num_filters_1,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding_size,
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(
                    in_channels=num_filters_1,
                    out_channels=num_filters_2,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding_size,
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(
                    in_channels=num_filters_2,
                    out_channels=num_filters_3,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding_size,
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Dropout(0.5),
            )
        else:    
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(
                    in_channels=image_channels,
                    out_channels=num_filters_1,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding_size,
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(
                    in_channels=num_filters_1,
                    out_channels=num_filters_2,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding_size,
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(
                    in_channels=num_filters_2,
                    out_channels=num_filters_3,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding_size,
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
            )

        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = num_filters_3*4*4

        mid_features = mid_filters
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, mid_features),
            nn.ReLU(),
            nn.Linear(mid_features, num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]
        out1 = self.feature_extractor(x)
        # out1 = x.view(x.size(0), -1)
        out = self.classifier(out1)
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (
            batch_size,
            self.num_classes,
        ), f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(
        trainer.train_history["loss"], label="Training loss", npoints_to_average=10
    )
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()

def create_plots2(trainer: Trainer, name: str, trainer2: Trainer, name2: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(
        trainer.train_history["loss"], label="Training loss: " + name, npoints_to_average=10
    )
    utils.plot_loss(
        trainer2.train_history["loss"], label="Training loss: " + name2, npoints_to_average=10
    )
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss: " + name)
    utils.plot_loss(trainer2.validation_history["loss"], label="Validation loss: " + name2)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy: " + name)
    utils.plot_loss(trainer2.validation_history["accuracy"], label="Validation Accuracy: " + name2)
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name2}_plot.png"))
    plt.show()




def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    utils.set_seed(0)
    print(f"Using device: {utils.get_device()}")
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = ExampleModel(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size, learning_rate, early_stop_count, epochs, model, dataloaders
    )
    trainer.train()


    model1 = ExampleModel(image_channels=3, num_classes=10, kernel_size=3)
    model2 = ExampleModel(image_channels=3, num_classes=10, mid_filters=80)
    model3 = ExampleModel(image_channels=3, num_classes=10, drop_out=True)
    model4 = ExampleModel(image_channels=3, num_classes=10, start_filters=256)
    models = [model1, model2, model3, model4]
    names = ["kernel-size", "mid_filters", "Dropout", "increased start filters"]

    for i in range(4):
        trainer2 = Trainer(
            batch_size, learning_rate, early_stop_count, epochs, models[i], dataloaders
        )
        trainer2.train()
        create_plots2(trainer, "no changes", trainer2, names[i])


if __name__ == "__main__":
    main()
