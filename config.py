LR_GEN = 0.0002
LR_DISC = 0.0002
BATCH_SIZE = 32
BETAS = (0.5, 0.999)

NUM_IMAGE_CHANNELS = 3
NUM_CONTOUR_CHANNELS = 1

IMAGE_DIM = 256

LAMBDA_L1 = 2
LAMBDA_ADVERSARIAL = 1

NUM_EPOCHES = 500

PRETRAINED = False
STATE_PATH = "./checkpoints/run 2/8/0.pth"

PHOTOS_PATH = "./dataset/archive/train/photos_286"
CONTOURS_PATH = "./dataset/archive/train/sketches_286"

PHOTOS_PATH_VAL = "./dataset/archive/val/photos_286"
CONTOURS_PATH_VAL = "./dataset/archive/val/sketches_286"

PHOTOS_PATH_TEST = "./dataset/archive/test/photos_286"
CONTOURS_PATH_TEST = "./dataset/archive/test/sketches_286"

CHECKPOINT_PATH = "./checkpoints"
RESULTS_PATH = "./results"
VAL_PATH = "./val"
UPDATES_PER_CHECKPOINT = 1000
UPDATES_PER_RESULT = 1000
UPDATES_PER_VAL = 1000
