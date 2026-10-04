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


class RunMode(StrEnum):
    SMOKE = "smoke"
    FULL = "full"
