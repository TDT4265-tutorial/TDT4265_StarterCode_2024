import pathlib
import matplotlib.pyplot as plt
import utils
import torch
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy


class ExampleModel(nn.Module):
    def __init__(self, image_channels, num_classes):
        """
        Is called when model is initialized.
        Args:
            image_channels. Number of color channels in image (3)
            num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # TODO: Implement this function (Task  2a)
        num_filters = 32  # Set number of filters in first conv layer
        self.num_classes = num_classes
        # Define the convolutional layers
        """
            After the first layer the output of feature_extractor would be [batch_size, num_filters, 32, 32]
            maxpool with stride=2 will half the size of the output (from 32L and 32 W to 16L and 16W)
            After the MaxPool2d layer the output of feature_extractor would be [batch_size, num_filters, 16, 16]
            that means after both the first convv and Pool layer  we would have:
            self.num_output_features = 32 * 16 * 16
        """
        self.feature_extractor = nn.Sequential(
            #layer1
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),

            #layer2
            nn.Conv2d(in_channels=num_filters, #equal to the output from prev layer
                out_channels=num_filters * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            #After this layer the output of feature_extractor would be [batch_size, num_filters * 2, 8, 8]

            #layer3
            nn.Conv2d(in_channels=num_filters * 2,
                out_channels=num_filters * 4,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
            #After this layer the output of feature_extractor would be [batch_size, num_filters * 4, 4, 4]

        
        )
        #The output of feature_extractor will be [batch_size, num_filters * 4, 4, 4]

        self.num_output_features = (num_filters * 4) * 4 * 4
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class. 
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.num_output_features, num_filters * 2),
            nn.ReLU(),
            nn.Linear(num_filters * 2, num_classes),

        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        features = self.feature_extractor(x)
        #make sure to flatten/reshape/view inbetween Convolution(feature_extract) and fullyconnected (classification) 
        out = self.classifier(features)
        
        batch_size = x.shape[0]
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
    create_plots(trainer, "task2")

    #try task 2b
    train_loss, train_accuracy = compute_loss_and_accuracy(dataloaders[0], model, torch.nn.CrossEntropyLoss())
    val_loss, val_accuracy = compute_loss_and_accuracy(dataloaders[1], model, torch.nn.CrossEntropyLoss())
    test_loss, test_accuracy = compute_loss_and_accuracy(dataloaders[2], model, torch.nn.CrossEntropyLoss())
    #Print the results
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
