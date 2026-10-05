from enum import IntEnum, StrEnum


class PatchSize(IntEnum):
    SMALLEST = 7
    SMALL = 11
    PAPER = 15
    LARGE = 19
    LARGEST = 23


class PaviaClass(IntEnum):
    UNLABELED = 0
    ASPHALT = 1
    MEADOWS = 2
    GRAVEL = 3
    TREES = 4
    PAINTED_METAL_SHEETS = 5
    BARE_SOIL = 6
    BITUMEN = 7
    SELF_BLOCKING_BRICKS = 8
    SHADOWS = 9

    @property
    def display_name(self):
        return {
            self.UNLABELED: "Unlabeled",
            self.ASPHALT: "Asphalt",
            self.MEADOWS: "Meadows",
            self.GRAVEL: "Gravel",
            self.TREES: "Trees",
            self.PAINTED_METAL_SHEETS: "Painted metal sheets",
            self.BARE_SOIL: "Bare Soil",
            self.BITUMEN: "Bitumen",
            self.SELF_BLOCKING_BRICKS: "Self-Blocking Bricks",
            self.SHADOWS: "Shadows",
        }[self]


class SplitName(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    EXCLUDED = "excluded"


class DataSplitType(StrEnum):
    RANDOM = "random"
    RANDOM_NO_CENTER_OVERLAP = "random_no_center_overlap"
    RANDOM_NO_OVERLAP = "random_no_overlap"


class RunMode(StrEnum):
    SMOKE = "smoke"
    FULL = "full"


class LRSchedulingType(StrEnum):
    FIXED = "fixed"
    COSINEANNEALING = "cosine_annealing"
    REDUCELRONPLATEAU = "reduce_lr_on_plateau"


class ModelArchitecture(StrEnum):
    CNN_ONLY = "cnn_only"
    CNN_CSA = "cnn_csa"
    CNN_TRANSFORMER = "cnn_transformer"
    CNN_TRANSFORMER_CSA = "cnn_transformer_csa"

    @property
    def display_name(self):
        return {
            self.CNN_ONLY: "CNN-Only",
            self.CNN_CSA: "CNN-CSA",
            self.CNN_TRANSFORMER: "CNN-Transformer",
            self.CNN_TRANSFORMER_CSA: "CNN-Transformer-CSA",
        }[self]

    @property
    def uses_transformer(self):
        return self in {
            self.CNN_TRANSFORMER,
            self.CNN_TRANSFORMER_CSA,
        }

    @property
    def uses_csa(self):
        return self in {
            self.CNN_CSA,
            self.CNN_TRANSFORMER_CSA,
        }
