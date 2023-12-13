class ModelConfig:
    def __init__(self):
        # Hyperparameters

        self.learning_rate = 0.005
        self.num_epochs = 100
        self.batch_size = 32

        self.num_classes=1
        self.weight_decay = 1e-5