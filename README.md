

trainset = torchvision.datasets.CIFAR10(
    root='./data',      # Directory where data will be saved
    train=True,         # Request the training dataset
    download=True,      # Download the dataset if not already present
    transform=transform # Apply the defined transformations
)