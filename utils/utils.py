import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys_epsilon = sys.float_info.epsilon

MOTHERDIR = Path(__file__).resolve().parent.parent

HEADERS = [
    "t",
    "X",
    "Y",
    "Z",
    "Ux",
    "Uy",
    "Uz",
    "G1",
    "G2",
    "G3",
    "G4",
    "G5",
    "G6",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "UUp1",
    "UUp2",
    "UUp3",
    "UUp4",
    "UUp5",
    "UUp6",
    "Cs",
]

M1_HEADERS = ["Ux", "Uy", "Uz", "S1", "S2", "S3", "S4", "S5", "S6", "Cs"]
M2_HEADERS = ["G1", "G2", "G3", "G4", "G5", "G6", "S1", "S2", "S3", "S4", "S5", "S6", "Cs"]
M3_HEADERS = ["Ux", "Uy", "Uz", "UUp1", "UUp2", "UUp3", "UUp4", "UUp5", "UUp6", "Cs"]
M4_HEADERS = ["G1", "G2", "G3", "G4", "G5", "G6", "UUp1", "UUp2", "UUp3", "UUp4", "UUp5", "UUp6", "Cs"]
M5_HEADERS = [
    "Ux",
    "Uy",
    "Uz",
    "G1",
    "G2",
    "G3",
    "G4",
    "G5",
    "G6",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "UUp1",
    "UUp2",
    "UUp3",
    "UUp4",
    "UUp5",
    "UUp6",
    "Cs",
]
M6_HEADERS = [
    "t",
    "X",
    "Y",
    "Z",
    "Ux",
    "Uy",
    "Uz",
    "G1",
    "G2",
    "G3",
    "G4",
    "G5",
    "G6",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "UUp1",
    "UUp2",
    "UUp3",
    "UUp4",
    "UUp5",
    "UUp6",
    "Cs",
]


class OFLESDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.data = dataframe

    def __getitem__(self, index):
        if index < len(self.data):
            return torch.tensor(self.data.iloc[index].values, dtype=torch.float64)
        raise IndexError(f"Index {index} out of range for dataset with length {len(self.data)}")

    def __len__(self):
        return len(self.data)


def R2Score(y_true, y_pred):
    SS_res = torch.sum(torch.square(y_true - y_pred))
    SS_tot = torch.var(y_true, unbiased=False) * y_true.size(0)
    return 1 - SS_res / (SS_tot + sys_epsilon)


def trainDataCollecter(Re):
    base = MOTHERDIR / "datasets_csv"
    coeff_dir = base / "coeffs" / "train"
    norm_dir = base / "normalized" / "train"
    org_dir = base / "original" / "train"

    means = pd.read_csv(coeff_dir / f"fieldData_{Re}_seen_means.csv")
    scales = pd.read_csv(coeff_dir / f"fieldData_{Re}_seen_scales.csv")
    norm = pd.read_csv(norm_dir / f"fieldData_{Re}_seen_norm.csv")
    org = pd.read_csv(org_dir / f"fieldData_{Re}_seen.csv")
    return org, norm, means, scales


def testDataCollecter(Re):
    base = MOTHERDIR / "datasets_csv"
    coeff_dir = base / "coeffs" / "test"
    norm_dir = base / "normalized" / "test"
    org_dir = base / "original" / "test"

    means = pd.read_csv(coeff_dir / f"fieldData_{Re}_unseen_means.csv")
    scales = pd.read_csv(coeff_dir / f"fieldData_{Re}_unseen_scales.csv")
    norm = pd.read_csv(norm_dir / f"fieldData_{Re}_unseen_norm.csv")
    org = pd.read_csv(org_dir / f"fieldData_{Re}_unseen.csv")
    return org, norm, means, scales
