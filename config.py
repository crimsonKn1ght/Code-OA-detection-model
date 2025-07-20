BATCH_SIZE = 8                  # Batch size
LR = 1e-4                       # Learning rate
EPOCHS = 20                     # Number of epochs to train

NUM_CLASSES = 3                 # No of classes in your classification task

DATASET_PATH = '/app/dataset'   # Path to the dataset directory

TRAIN_RATIO = 0.7               # Train:Val:Test split ratios
VAL_RATIO = 0.15
TEST_RATIO = 0.15

LOG_DIR = 'runs'                # Directory for TensorBoard logs
CHECKPOINT_DIR = 'checkpoints'  # Directory for saving model checkpoints