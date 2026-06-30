from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 100
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = ROOT_DIR / "models" / "brain_tumor_model.keras"
