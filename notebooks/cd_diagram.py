import aeon.visualisation
import pathlib

from plots import *
from stats import *
from cd_diagram import *
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
)

base_path = pathlib.Path("/usr/src/code/data/outputs")
pmiss = 70
model = "transformer"
metric_name = "f1_score"

datasets = [
    "ArticularyWordRecognition",
    "BasicMotions",
    "Cricket",
    "ERing",
    "Epilepsy",
    "Heartbeat",
    "JapaneseVowels",
    "LSST",
    "Libras",
    "NATOPS",
    "PEMS-SF",
    "RacketSports",
    "SelfRegulationSCP1",
    "SpokenArabicDigits",
    "UWaveGestureLibrary",
]

if __name__ == "__main__":
    if model == "rnn":
        models = [
            "RNN",
            "RNNTimestamps",
            "RNNTimestampsRel",
            "RNNTime2Vec",
            "RNNPE",
            "RNNTPE",
            "RNNLinear",
            "RNNTime2VecRel",
            "RNNPERel",
            "RNNTPERel",
            "RNNLinearRel",
        ]
    elif model == "transformer":
        models = [
            "Transformer",
            "TransformerTimestamps",
            "TransformerTimestampsRel",
            "TransformerTime2Vec",
            "TransformerPE",
            "TransformerTPE",
            "TransformerLinear",
            "TransformerTime2VecRel",
            "TransformerPERel",
            "TransformerTPERel",
            "TransformerLinearRel",
        ]

    df = gather_metric_cd(
        f1_score,
        "f1_score",
        base_path,
        datasets,
        models,
        pmiss,
        func_params={"average": "weighted"},
    )
    fig, ax, p_values = aeon.visualisation.plot_critical_difference(
        df.to_numpy(),
        labels=models,
        alpha=0.05,
        return_p_values=True,
        width=6,
        textspace=1.25,
    )
    fig.suptitle(f"Missing percentage: {pmiss}%", y=1.05, size=16)
    fig.savefig(
        f"notebooks/figures/cd-diagram_{model}_{pmiss}.png", bbox_inches="tight"
    )
