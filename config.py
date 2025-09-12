LR_GEN = 0.0002
# LR_GEN = 0.00005
LR_DISC = 0.0002
# LR_DISC = 0.00005
BATCH_SIZE = 32
BETAS = (0.5, 0.999)

NUM_IMAGE_CHANNELS = 3
NUM_CONTOUR_CHANNELS = 1

IMAGE_DIM = 256

LAMBDA_L1 = 100  # 2 u run 4  #1 u run 3  #100 u run 1 i run 2 # run 5 100
LAMBDA_ADVERSARIAL = 1

NUM_EPOCHES = 500

PRETRAINED = False
STATE_PATH = "./checkpoints/run 6 part 2/65/0.pth"

PHOTOS_PATH = "./dataset/archive/train/photos_286"
PHOTOS_PATH_TRANSFER = "./dataset/transfer learning/archive/photos_256"
CONTOURS_PATH = "./dataset/archive/train/sketches_286"
CONTOURS_PATH_TRANSFER = "./dataset/transfer learning/archive/sketches_256"

PHOTOS_PATH_VAL = "./dataset/archive/val/photos_256_500"
CONTOURS_PATH_VAL = "./dataset/archive/val/sketches_256_500"

PHOTOS_PATH_TEST = "./dataset/archive/test/photos_286"
# PHOTOS_PATH_TEST = "./dataset/test/images"
CONTOURS_PATH_TEST = "./dataset/archive/test/sketches_286"
# CONTOURS_PATH_TEST = "./dataset/test/contours"


CHECKPOINT_PATH = "./checkpoints"
RESULTS_PATH = "./results"
VAL_PATH = "./val"
INFERENCE_PATH = "./inference"

UPDATES_PER_CHECKPOINT = 1000
# UPDATES_PER_CHECKPOINT = 12
UPDATES_PER_RESULT = 1000
# UPDATES_PER_RESULT = 12
UPDATES_PER_VAL = 1000
# UPDATES_PER_VAL = 12
