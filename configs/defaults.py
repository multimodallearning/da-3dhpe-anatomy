from yacs.config import CfgNode as CN

_C = CN()

_C.BASE_DIRECTORY = "../results"
_C.EXPERIMENT_NAME = ""


_C.INPUT = CN()
_C.INPUT.NUM_WORKERS = 8
_C.INPUT.NUM_POINTS = 2048
_C.INPUT.ROT_DEGREE = 0.
_C.INPUT.TRANSLATION = 0.


_C.OUTPUT = CN()
_C.OUTPUT.NUM_JOINTS = 24


_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHT = ""


_C.SOLVER = CN()
_C.SOLVER.USE_AMP = True
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.LR_MILESTONES = [60, 90]
_C.SOLVER.LR_GAMMA = 0.1
_C.SOLVER.NUM_EPOCHS = 100
_C.SOLVER.BATCH_SIZE_TRAIN = 8
_C.SOLVER.BATCH_SIZE_TEST = 8


_C.SLP_DATASET = CN()
_C.SLP_DATASET.POSITION = 'all'
_C.SLP_DATASET.ROOT = '../datasets/SLP_processed'


_C.DA = CN()
_C.DA.METHOD = ''
_C.DA.ANATOMY_SYMMETRY = 0.1
_C.DA.ANATOMY_RANGE = 0.1
_C.DA.ANATOMY_ANGLE = 0.1
_C.DA.ANATOMY_BONE_LENGTH_LOWER = [0.0970, 0.0970, 0.0993, 0.3202, 0.3202, 0.1240, 0.3302, 0.3302, 0.0535, 0.1183,
                                   0.1183, 0.1924, 0.1212, 0.1212, 0.0688, 0.0846, 0.0846, 0.2222, 0.2222, 0.2158,
                                   0.2158, 0.0752, 0.0752]
_C.DA.ANATOMY_BONE_LENGTH_UPPER = [0.1199, 0.1199, 0.1364, 0.4042, 0.4042, 0.1548, 0.4386, 0.4386, 0.0617, 0.1521,
                                   0.1521, 0.2175, 0.1432, 0.1432, 0.1127, 0.1347, 0.1347, 0.2666, 0.2666, 0.2748,
                                   0.2748, 0.0907, 0.0907]
_C.DA.ANATOMY_ANGLE_LOWER = [-0.46, 0.68, 0.68, -0.56, -0.85, -0.05, -0.56, -0.85, -0.05, -0.12, 0.55, 0.36, 0.21, 0.21,
                             -0.03, -0.05, -0.65, -0.89, 0.69, -0.05, -0.65, -0.89, 0.69, -0.86, -0.86]
_C.DA.ANATOMY_ANGLE_UPPER = [-0.11, 0.80, 0.80, 1.00, 0.99, 0.98, 1.00, 0.99, 0.98, 1.00, 1.00, 1.00, 0.92, 0.92, 0.98,
                             1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, -0.81, -0.81]


def get_cfg_defaults():
    return _C.clone()