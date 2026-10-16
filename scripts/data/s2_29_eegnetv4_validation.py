from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from s2_23_check_real_train_dataset import (
    FINAL_SPLIT_MANIFEST,
    NORMALIZATION_FILE,
    build_train_window_index,
)

from s2_24_full_train_loader_sanity import (
    CachedRealTrainWindowDataset,
)

from s2_27_logbandpower_logreg_validation import (
    build_validation_window_index,
    subject_macro_balanced_accuracy,
)


# ============================================================
# Paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    REPO_ROOT
    / "outputs"
    / "models"
    / "scenario2"
)

METRIC_DIR = (
    REPO_ROOT
    / "outputs"
    / "metrics"
    / "scenario2"
)

MODEL_PATH = (
    MODEL_DIR
    / "eegnetv4_real_baseline_v1.pt"
)

RESULT_PATH = (
    METRIC_DIR
    / "real_eegnetv4_validation_v1.json"
)


# ============================================================
# Frozen baseline configuration
# ============================================================

SEED = 20260907

N_CHANNELS = 19
N_TIMES = 512
N_CLASSES = 2

F1 = 8
D = 2
F2 = 16

TEMPORAL_KERNEL = 64
SEPARABLE_KERNEL = 16

DROPOUT = 0.50

BATCH_SIZE = 128

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0

MAX_EPOCHS = 25
MIN_EPOCHS = 8
PATIENCE = 6

EXPECTED_TRAIN_WINDOWS = 47_835
EXPECTED_VALIDATION_WINDOWS = 9_635
EXPECTED_SUBJECTS = 198


# ============================================================
# Determinism
# ============================================================

def set_seed(seed: int) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# EEGNetV4 baseline
# ============================================================

class EEGNetV4Baseline(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        # ----------------------------------------------------
        # Block 1:
        # temporal filtering
        # ----------------------------------------------------

        self.temporal = nn.Conv2d(
            in_channels=1,
            out_channels=F1,

            kernel_size=(
                1,
                TEMPORAL_KERNEL,
            ),

            padding="same",

            bias=False,
        )

        self.bn_temporal = nn.BatchNorm2d(
            F1
        )

        # ----------------------------------------------------
        # Depthwise spatial filtering
        #
        # One spatial filter spans all 19 EEG channels.
        # ----------------------------------------------------

        self.spatial = nn.Conv2d(
            in_channels=F1,

            out_channels=(
                F1 * D
            ),

            kernel_size=(
                N_CHANNELS,
                1,
            ),

            groups=F1,

            bias=False,
        )

        self.bn_spatial = nn.BatchNorm2d(
            F1 * D
        )

        self.activation1 = nn.ELU()

        self.pool1 = nn.AvgPool2d(
            kernel_size=(1, 4)
        )

        self.dropout1 = nn.Dropout(
            DROPOUT
        )

        # ----------------------------------------------------
        # Block 2:
        # depthwise separable temporal convolution
        # ----------------------------------------------------

        self.separable_depthwise = nn.Conv2d(
            in_channels=(
                F1 * D
            ),

            out_channels=(
                F1 * D
            ),

            kernel_size=(
                1,
                SEPARABLE_KERNEL,
            ),

            padding="same",

            groups=(
                F1 * D
            ),

            bias=False,
        )

        self.separable_pointwise = nn.Conv2d(
            in_channels=(
                F1 * D
            ),

            out_channels=F2,

            kernel_size=(1, 1),

            bias=False,
        )

        self.bn_separable = nn.BatchNorm2d(
            F2
        )

        self.activation2 = nn.ELU()

        self.pool2 = nn.AvgPool2d(
            kernel_size=(1, 8)
        )

        self.dropout2 = nn.Dropout(
            DROPOUT
        )

        # ----------------------------------------------------
        # After pooling:
        #
        # 512 / 4 / 8 = 16 time positions
        #
        # shape:
        # F2 × 1 × 16
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            F2 * 16,
            N_CLASSES,
        )

    # --------------------------------------------------------
    # EEGNet-style max-norm constraints
    # --------------------------------------------------------

    @torch.no_grad()
    def apply_max_norm(self) -> None:

        # Spatial depthwise filters:
        # maximum L2 norm = 1
        w = self.spatial.weight

        flat = w.view(
            w.shape[0],
            -1,
        )

        norms = torch.linalg.vector_norm(
            flat,
            ord=2,
            dim=1,
            keepdim=True,
        ).clamp_min(
            1e-12
        )

        scale = torch.clamp(
            1.0 / norms,
            max=1.0,
        )

        flat.mul_(scale)

        # Classifier rows:
        # conservative EEGNet-style constraint
        w = self.classifier.weight

        norms = torch.linalg.vector_norm(
            w,
            ord=2,
            dim=1,
            keepdim=True,
        ).clamp_min(
            1e-12
        )

        scale = torch.clamp(
            0.25 / norms,
            max=1.0,
        )

        w.mul_(scale)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # Input:
        # batch × channels × time
        #
        # Convert to:
        # batch × 1 × channels × time

        x = x.unsqueeze(1)

        x = self.temporal(x)
        x = self.bn_temporal(x)

        x = self.spatial(x)
        x = self.bn_spatial(x)

        x = self.activation1(x)

        x = self.pool1(x)

        x = self.dropout1(x)

        x = self.separable_depthwise(x)

        x = self.separable_pointwise(x)

        x = self.bn_separable(x)

        x = self.activation2(x)

        x = self.pool2(x)

        x = self.dropout2(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        logits = self.classifier(x)

        return logits


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:

    model.eval()

    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_examples = 0

    all_y = []
    all_pred = []
    all_prob = []
    all_subject = []

    for batch in loader:

        x = batch["x"].to(
            device,
            non_blocking=True,
        )

        y = batch["y"].to(
            device,
            non_blocking=True,
        )

        subject = batch[
            "subject_index"
        ]

        logits = model(x)

        loss = loss_fn(
            logits,
            y,
        )

        probability = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        prediction = torch.argmax(
            logits,
            dim=1,
        )

        batch_size = int(
            x.shape[0]
        )

        total_loss += (
            float(loss.item())
            * batch_size
        )

        total_examples += batch_size

        all_y.append(
            y.cpu().numpy()
        )

        all_pred.append(
            prediction.cpu().numpy()
        )

        all_prob.append(
            probability.cpu().numpy()
        )

        all_subject.append(
            subject.numpy()
        )

    y_true = np.concatenate(
        all_y
    )

    y_pred = np.concatenate(
        all_pred
    )

    y_prob = np.concatenate(
        all_prob
    )

    subject_index = np.concatenate(
        all_subject
    )

    subject_macro_ba, subject_scores = (
        subject_macro_balanced_accuracy(
            y_true,
            y_pred,
            subject_index,
        )
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    window_ba = balanced_accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob,
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    return {
        "loss": (
            total_loss
            / total_examples
        ),

        "subject_macro_ba": (
            subject_macro_ba
        ),

        "subject_scores": (
            subject_scores
        ),

        "accuracy": accuracy,

        "window_ba": window_ba,

        "macro_f1": macro_f1,

        "roc_auc": roc_auc,

        "mcc": mcc,

        "confusion_matrix": cm,

        "y_true": y_true,

        "y_pred": y_pred,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 80)
    print("SCENARIO 2 — EEGNETV4 REAL VALIDATION BASELINE")
    print("=" * 80)

    print(
        "\nTHIS IS AN UNTUNED PREDECLARED BASELINE."
    )

    print(
        f"\nseed={SEED}"
        f"\nF1={F1}"
        f"\nD={D}"
        f"\nF2={F2}"
        f"\ntemporal_kernel={TEMPORAL_KERNEL}"
        f"\nseparable_kernel={SEPARABLE_KERNEL}"
        f"\ndropout={DROPOUT}"
        f"\nbatch_size={BATCH_SIZE}"
        f"\nlr={LEARNING_RATE}"
        f"\nmax_epochs={MAX_EPOCHS}"
        f"\nmin_epochs={MIN_EPOCHS}"
        f"\npatience={PATIENCE}"
    )

    print(
        "\nTrain = model fitting"
        "\nValidation = checkpoint selection/evaluation"
        "\nFinal Test = SEALED"
    )

    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this baseline."
        )

    device = torch.device(
        "cuda"
    )

    print(
        f"\nGPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    # ========================================================
    # Step 1 — Frozen data definitions
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 1 — BUILD TRAIN / VALIDATION DATASETS")
    print("=" * 80)

    split = pd.read_csv(
        FINAL_SPLIT_MANIFEST
    )

    with NORMALIZATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        scaler = json.load(f)

    if scaler[
        "status"
    ] != "TRAIN_NORMALIZATION_LOCKED":

        raise RuntimeError(
            "Normalization not locked."
        )

    train_records, train_mapping = (
        build_train_window_index(
            split_manifest=split,
            scaler=scaler,
        )
    )

    val_records, val_mapping = (
        build_validation_window_index(
            split
        )
    )

    if train_mapping != val_mapping:
        raise RuntimeError(
            "Train/Validation subject mapping mismatch."
        )

    if len(train_records) != EXPECTED_TRAIN_WINDOWS:
        raise RuntimeError(
            "Unexpected Train window count."
        )

    if len(val_records) != EXPECTED_VALIDATION_WINDOWS:
        raise RuntimeError(
            "Unexpected Validation window count."
        )

    train_dataset = (
        CachedRealTrainWindowDataset(
            records=train_records,
            scaler=scaler,
            max_open_files=8,
        )
    )

    val_dataset = (
        CachedRealTrainWindowDataset(
            records=val_records,
            scaler=scaler,
            max_open_files=8,
        )
    )

    generator = torch.Generator()

    generator.manual_seed(
        SEED
    )

    train_loader = DataLoader(
        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=True,

        drop_last=False,

        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=True,

        drop_last=False,
    )

    print(
        f"Train windows: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation windows: "
        f"{len(val_dataset)}"
    )

    print(
        "Test EEG accessed: NO"
    )

    # ========================================================
    # Step 2 — Build model
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 2 — BUILD EEGNETV4")
    print("=" * 80)

    model = EEGNetV4Baseline().to(
        device
    )

    n_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    n_trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"parameters: "
        f"{n_parameters:,}"
    )

    print(
        f"trainable: "
        f"{n_trainable:,}"
    )

    # Smoke test
    test_input = torch.zeros(
        (
            2,
            19,
            512,
        ),
        device=device,
    )

    with torch.no_grad():

        test_output = model(
            test_input
        )

    print(
        f"input shape: "
        f"{tuple(test_input.shape)}"
    )

    print(
        f"output shape: "
        f"{tuple(test_output.shape)}"
    )

    if test_output.shape != (
        2,
        2,
    ):
        raise RuntimeError(
            "Unexpected EEGNet output shape."
        )

    # ========================================================
    # Step 3 — Training objects
    # ========================================================

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,
    )

    # ========================================================
    # Step 4 — Training
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 3 — TRAIN EEGNETV4")
    print("=" * 80)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_subject_ba = (
        float("-inf")
    )

    best_val_loss = (
        float("inf")
    )

    best_epoch = None

    epochs_without_improvement = 0

    history = []

    training_start = (
        time.perf_counter()
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        epoch_start = (
            time.perf_counter()
        )

        model.train()

        train_loss_sum = 0.0
        train_examples = 0

        train_correct = 0

        for batch in train_loader:

            x = batch["x"].to(
                device,
                non_blocking=True,
            )

            y = batch["y"].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(x)

            loss = loss_fn(
                logits,
                y,
            )

            loss.backward()

            optimizer.step()

            model.apply_max_norm()

            batch_size = int(
                x.shape[0]
            )

            train_loss_sum += (
                float(loss.item())
                * batch_size
            )

            train_examples += (
                batch_size
            )

            prediction = torch.argmax(
                logits,
                dim=1,
            )

            train_correct += int(
                (
                    prediction == y
                )
                .sum()
                .item()
            )

        train_loss = (
            train_loss_sum
            / train_examples
        )

        train_accuracy = (
            train_correct
            / train_examples
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_result = evaluate(
            model,
            val_loader,
            device,
        )

        val_subject_ba = float(
            val_result[
                "subject_macro_ba"
            ]
        )

        val_loss = float(
            val_result[
                "loss"
            ]
        )

        # ----------------------------------------------------
        # Checkpoint rule:
        #
        # 1. higher subject-macro BA
        # 2. tie -> lower Validation loss
        # ----------------------------------------------------

        improved = False

        if (
            val_subject_ba
            > best_subject_ba
            + 1e-12
        ):

            improved = True

        elif (
            abs(
                val_subject_ba
                - best_subject_ba
            )
            <= 1e-12
            and val_loss
            < best_val_loss
        ):

            improved = True

        if improved:

            best_subject_ba = (
                val_subject_ba
            )

            best_val_loss = (
                val_loss
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),

                    "epoch": epoch,

                    "subject_macro_ba": (
                        best_subject_ba
                    ),

                    "validation_loss": (
                        best_val_loss
                    ),

                    "architecture": {
                        "F1": F1,
                        "D": D,
                        "F2": F2,

                        "temporal_kernel": (
                            TEMPORAL_KERNEL
                        ),

                        "separable_kernel": (
                            SEPARABLE_KERNEL
                        ),

                        "dropout": (
                            DROPOUT
                        ),
                    },

                    "seed": SEED,
                },

                MODEL_PATH,
            )

        else:

            epochs_without_improvement += 1

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        print(
            f"epoch "
            f"{epoch:>2}/{MAX_EPOCHS}"
            f" | train_loss="
            f"{train_loss:.4f}"
            f" | train_acc="
            f"{train_accuracy:.4f}"
            f" | val_loss="
            f"{val_loss:.4f}"
            f" | val_subject_BA="
            f"{val_subject_ba:.4f}"
            f" | best="
            f"{best_subject_ba:.4f}"
            f" @ {best_epoch}"
            f" | {epoch_seconds:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,

                "train_loss": (
                    train_loss
                ),

                "train_accuracy": (
                    train_accuracy
                ),

                "validation_loss": (
                    val_loss
                ),

                "validation_subject_macro_ba": (
                    val_subject_ba
                ),
            }
        )

        # ----------------------------------------------------
        # Predeclared early stopping
        # ----------------------------------------------------

        if (
            epoch >= MIN_EPOCHS
            and epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEARLY STOPPING"
            )

            print(
                f"no improvement for "
                f"{PATIENCE} epochs"
            )

            break

    total_training_seconds = (
        time.perf_counter()
        - training_start
    )

    train_dataset.close()
    val_dataset.close()

    # ========================================================
    # Step 5 — Load best checkpoint
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 4 — LOAD BEST CHECKPOINT")
    print("=" * 80)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    print(
        f"best epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"best subject-macro BA: "
        f"{checkpoint['subject_macro_ba']:.6f}"
    )

    # Recreate Validation dataset because previous cache was closed.

    val_dataset = (
        CachedRealTrainWindowDataset(
            records=val_records,
            scaler=scaler,
            max_open_files=8,
        )
    )

    val_loader = DataLoader(
        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=True,

        drop_last=False,
    )

    # ========================================================
    # Step 6 — Final Validation evaluation
    # ========================================================

    print("\n" + "=" * 80)
    print("STEP 5 — FINAL REAL VALIDATION EVALUATION")
    print("=" * 80)

    result = evaluate(
        model,
        val_loader,
        device,
    )

    val_dataset.close()

    subject_scores = result[
        "subject_scores"
    ]

    cm = result[
        "confusion_matrix"
    ]

    print(
        "\nPRIMARY METRIC"
    )

    print(
        f"Subject-macro Balanced Accuracy: "
        f"{result['subject_macro_ba']:.6f}"
    )

    print(
        "\nSubject BA distribution:"
    )

    print(
        f"  min    = "
        f"{subject_scores.min():.6f}"
    )

    print(
        f"  Q1     = "
        f"{np.percentile(subject_scores, 25):.6f}"
    )

    print(
        f"  median = "
        f"{np.median(subject_scores):.6f}"
    )

    print(
        f"  Q3     = "
        f"{np.percentile(subject_scores, 75):.6f}"
    )

    print(
        f"  max    = "
        f"{subject_scores.max():.6f}"
    )

    print(
        "\nWINDOW METRICS"
    )

    print(
        f"Accuracy:          "
        f"{result['accuracy']:.6f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{result['window_ba']:.6f}"
    )

    print(
        f"Macro-F1:          "
        f"{result['macro_f1']:.6f}"
    )

    print(
        f"ROC-AUC:           "
        f"{result['roc_auc']:.6f}"
    )

    print(
        f"MCC:               "
        f"{result['mcc']:.6f}"
    )

    print(
        "\nCONFUSION MATRIX"
    )

    print(
        "rows=true [EO, EC]"
    )

    print(
        "cols=pred [EO, EC]"
    )

    print(cm)

    # ========================================================
    # Step 7 — Save metrics
    # ========================================================

    METRIC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "status": (
            "REAL_VALIDATION_BASELINE_COMPLETE"
        ),

        "model": (
            "EEGNetV4Baseline"
        ),

        "seed": SEED,

        "architecture": {
            "channels": (
                N_CHANNELS
            ),

            "times": N_TIMES,

            "F1": F1,

            "D": D,

            "F2": F2,

            "temporal_kernel": (
                TEMPORAL_KERNEL
            ),

            "separable_kernel": (
                SEPARABLE_KERNEL
            ),

            "dropout": (
                DROPOUT
            ),
        },

        "training": {
            "batch_size": (
                BATCH_SIZE
            ),

            "learning_rate": (
                LEARNING_RATE
            ),

            "weight_decay": (
                WEIGHT_DECAY
            ),

            "max_epochs": (
                MAX_EPOCHS
            ),

            "min_epochs": (
                MIN_EPOCHS
            ),

            "patience": (
                PATIENCE
            ),

            "best_epoch": int(
                checkpoint[
                    "epoch"
                ]
            ),

            "training_seconds": float(
                total_training_seconds
            ),
        },

        "train_windows": (
            EXPECTED_TRAIN_WINDOWS
        ),

        "validation_windows": (
            EXPECTED_VALIDATION_WINDOWS
        ),

        "primary_metric": {
            "name": (
                "subject_macro_balanced_accuracy"
            ),

            "value": float(
                result[
                    "subject_macro_ba"
                ]
            ),
        },

        "subject_balanced_accuracy": {
            "minimum": float(
                subject_scores.min()
            ),

            "q1": float(
                np.percentile(
                    subject_scores,
                    25,
                )
            ),

            "median": float(
                np.median(
                    subject_scores
                )
            ),

            "q3": float(
                np.percentile(
                    subject_scores,
                    75,
                )
            ),

            "maximum": float(
                subject_scores.max()
            ),
        },

        "window_metrics": {
            "accuracy": float(
                result[
                    "accuracy"
                ]
            ),

            "balanced_accuracy": float(
                result[
                    "window_ba"
                ]
            ),

            "macro_f1": float(
                result[
                    "macro_f1"
                ]
            ),

            "roc_auc": float(
                result[
                    "roc_auc"
                ]
            ),

            "mcc": float(
                result[
                    "mcc"
                ]
            ),
        },

        "confusion_matrix": (
            cm.astype(int).tolist()
        ),

        "checkpoint_rule": (
            "highest Validation "
            "subject-macro balanced accuracy; "
            "tie -> lowest Validation loss"
        ),

        "normalization": (
            "locked_global_train_only"
        ),

        "final_test_accessed": False,

        "history": history,
    }

    with RESULT_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            sort_keys=True,
        )

        f.write("\n")

    # ========================================================
    # Final comparison
    # ========================================================

    print("\n" + "=" * 80)
    print("EEGNETV4 BASELINE COMPLETE")
    print("=" * 80)

    print(
        f"best epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"PRIMARY subject-macro BA: "
        f"{result['subject_macro_ba']:.6f}"
    )

    print(
        f"training time: "
        f"{total_training_seconds / 60:.2f} min"
    )

    print(
        f"checkpoint: {MODEL_PATH}"
    )

    print(
        f"metrics: {RESULT_PATH}"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "\nDO NOT TUNE YET."
    )

    print(
        "First compare all three "
        "untuned baselines."
    )


if __name__ == "__main__":
    main()