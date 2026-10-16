"""Continue the existing B0 run; never start from scratch.

Place next to s2_35_b0_full_training.py in scripts/data/.
Default: check/backup/replay only. Add --execute to continue training.
Old scripts, checkpoints, history and protocol are READ ONLY.
All new artifacts: outputs/runs/scenario2/b0/full_amp_safe_v1/.

This is an engineering continuation: standard GradScaler skip/backoff,
EMA only after a real optimizer step, CPU-safe RNG restore, safe saves.
It is NOT an architecture, loss, learning-rate, or epoch-budget change.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from typing import Any
import uuid

# Avoid new __pycache__ writes when this file is imported for testing.
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/runs/scenario2/b0/full_amp_safe_v1"
CHECKPOINTS = RUN / "checkpoints"
LOGS = RUN / "logs"
METRICS = RUN / "metrics"
FIGURES = RUN / "figures"
BACKUP = RUN / "backup_original"
LATEST = CHECKPOINTS / "latest.pt"
PREVIOUS = CHECKPOINTS / "previous_complete.pt"
BEST = CHECKPOINTS / "best_ema.pt"
BINDING = RUN / "continuation_binding.json"
STATE = METRICS / "state.json"
EVENTS = LOGS / "events.jsonl"

# Safety limits, NOT tunable model hyperparameters.
PATCH_ID = "B0_AMP_SAFE_CONTINUATION_V1"
MAX_SKIPS_PER_EPOCH = 5
MAX_CONSECUTIVE_SKIPS = 3
REPLAY_ATOL = 1e-6
GIB = 1024 ** 3

# Accepted input bytes from the completed normalization/split phase.
ACCEPTED_INPUTS = {
    "final_cohort_manifest.csv": "e9405aacd8109700ed55ef3df49b8b5ec0187a65b1c3d79049597fa27b490bc9",
    "final_block_split_manifest.csv": "a58605010f7371fe1805a7834c205f2da7be816515779e15f2ba082bfd201f8b",
    "train_normalization.json": "82af5c196145756f09658ba5fd887a9953962349c23af922eca20df6b86d7379",
}
SCRIPT_NAMES = (
    "s2_23_check_real_train_dataset.py",
    "s2_24_full_train_loader_sanity.py",
    "s2_27_logbandpower_logreg_validation.py",
    "s2_32_b0_unet_shape_test.py",
    "s2_33_b0_one_batch_diffusion_sanity.py",
    "s2_35_b0_full_training.py",
    "s2_35b_b0_resume_amp_safe.py",
)
POLICY = {
    "patch_id": PATCH_ID,
    "nonfinite_gradients": "GradScaler.step skips; update backs off; EMA unchanged",
    "nonfinite_input_prediction_or_loss": "STOP",
    "max_skips_per_epoch": MAX_SKIPS_PER_EPOCH,
    "stop_at_consecutive_skips": MAX_CONSECUTIVE_SKIPS,
    "resume_granularity": "last complete epoch; no automatic fallback",
    "old_outputs": "read_only",
    "fresh_training": False,
    "final_test_access": False,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    require(path.is_file(), f"Missing required file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    # A failed replacement leaves the old destination AND the new temp evidence.
    os.replace(temporary, path)


def save_json(path: Path, content: dict) -> None:
    atomic_bytes(path, json_bytes(content))


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        result = json.load(stream)
    require(isinstance(result, dict), f"Expected JSON object: {path}")
    return result


def verified_copy(source: Path, destination: Path, *, immutable: bool = True) -> str:
    expected = sha256(source)
    if immutable and destination.exists():
        require(sha256(destination) == expected, f"Conflicting backup: {destination}")
        return expected
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{uuid.uuid4().hex}")
    with source.open("rb") as src, temporary.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    require(sha256(temporary) == expected, f"Backup verification failed: {source}")
    require(sha256(source) == expected, f"Source changed while copying: {source}")
    os.replace(temporary, destination)
    return expected


def event(kind: str, **fields: Any) -> None:
    record = {"utc": utc_now(), "pid": os.getpid(), "kind": kind, **fields}
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class Tee:
    def __init__(self, console: Any, logfile: Any):
        self.console, self.logfile = console, logfile

    def write(self, text: str) -> int:
        self.console.write(text)
        self.logfile.write(text)
        self.flush()
        return len(text)

    def flush(self) -> None:
        self.console.flush()
        self.logfile.flush()

    def isatty(self) -> bool:
        return False


@contextlib.contextmanager
def single_instance():
    """OS lock: released on process exit; never delete/reinterpret old PID files."""
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "execution.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another AMP-safe continuation holds the lock. Do not launch twice.") from exc
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def check_no_other_training_process() -> None:
    """Read-only Windows process check, including Python venv launcher ancestors."""
    require(os.name == "nt", "This launcher targets the existing Windows project.")
    command = (
        "Get-CimInstance Win32_Process | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
    )
    require(result.returncode == 0, "Cannot inspect live process state; no training was started.")
    rows = json.loads(result.stdout.lstrip("\ufeff"))
    if isinstance(rows, dict):
        rows = [rows]
    parents = {int(row["ProcessId"]): int(row["ParentProcessId"]) for row in rows}
    ancestors = {os.getpid()}
    pid = os.getpid()
    while parents.get(pid, 0) not in ancestors and parents.get(pid, 0) != 0:
        pid = parents[pid]
        ancestors.add(pid)
    matches = []
    for row in rows:
        if int(row["ProcessId"]) in ancestors:
            continue
        if not str(row.get("Name", "")).lower().startswith("python"):
            continue
        cmd = str(row.get("CommandLine") or "").lower()
        if "s2_35_b0_full_training.py" in cmd or "s2_35b_b0_resume_amp_safe.py" in cmd:
            matches.append(int(row["ProcessId"]))
    require(not matches, f"Existing B0 training process(es): {matches}. Leave them untouched; do not duplicate.")


def check_environment_and_disk(extra_required: int = GIB) -> None:
    expected_python = ROOT / ".venv/Scripts/python.exe"
    require(Path(sys.executable).resolve() == expected_python.resolve(),
            f"Use the Scenario 2 interpreter: {expected_python}")
    temp_root = Path(r"D:\SHARIF\PROJECT\Datasets\MPI-LEMON-Scenario2\temp").resolve()
    actual_temp = Path(tempfile.gettempdir()).resolve()
    require(actual_temp == temp_root or temp_root in actual_temp.parents,
            "Project D: temp environment is not active. Dot-source setup_scenario2_environment.ps1 first.")
    require(shutil.disk_usage("C:\\").free >= 3 * GIB, "C: has less than the required 3 GiB free.")
    require(shutil.disk_usage(ROOT).free >= extra_required,
            f"Insufficient output-drive space: need >= {extra_required / GIB:.2f} GiB free.")


def load_runtime() -> None:
    global np, pd, torch, nn, DataLoader, plt, old
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Import definitions only. The old file's main() is NEVER called.
    import s2_35_b0_full_training as old


def finite_tree(value: Any, prefix: str = "checkpoint") -> None:
    if torch.is_tensor(value):
        if value.is_floating_point() or value.is_complex():
            require(bool(torch.isfinite(value).all().item()), f"Nonfinite stored tensor: {prefix}")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite_tree(item, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite_tree(item, f"{prefix}[{index}]")


def validate_checkpoint(cp: dict, contract: dict, protocol: dict) -> None:
    require(cp.get("status") == "B0_FULL_TRAINING_CHECKPOINT", "Not a full B0 checkpoint.")
    require(cp.get("contract") == contract, "Checkpoint architecture/diffusion contract differs.")
    require(cp.get("full_training_protocol") == protocol, "Checkpoint full protocol differs.")
    require(cp.get("final_test_accessed") is False, "Checkpoint Test-access field is inconsistent.")
    require(cp.get("validation_accessed") is True, "Checkpoint Validation field is inconsistent.")
    require(cp.get("seed") == old.SEED and cp.get("validation_seed") == old.VALIDATION_SEED,
            "Checkpoint RNG seeds differ from the existing run.")
    epoch = int(cp["epoch"])
    require(1 <= epoch <= old.MAX_EPOCHS, f"Invalid complete epoch: {epoch}")
    history = cp["history"]
    require([int(row["epoch"]) for row in history] == list(range(1, epoch + 1)),
            "Checkpoint epoch history is missing, duplicated or incomplete.")
    for row in history:
        for key in ("train_mse", "ema_validation_mse"):
            require(math.isfinite(float(row[key])) and float(row[key]) >= 0,
                    f"Invalid history value at epoch {row['epoch']}: {key}")
    selected = min(history, key=lambda row: float(row["ema_validation_mse"]))
    require(int(cp["best_epoch"]) == int(selected["epoch"]), "Best epoch does not match history.")
    require(math.isclose(float(cp["best_ema_validation_mse"]), float(selected["ema_validation_mse"]),
                         rel_tol=0, abs_tol=1e-12), "Best MSE does not match history.")
    for key in ("model_state_dict", "ema_state_dict", "optimizer_state_dict", "grad_scaler_state_dict"):
        require(key in cp, f"Missing resumable component: {key}")
        finite_tree(cp[key], key)
    for key in ("python", "numpy", "torch_cpu", "torch_cuda"):
        require(key in cp.get("rng_state", {}), f"Missing RNG state: {key}")
    require("train_generator_state" in cp, "Missing shuffle-generator state.")
    scale = float(cp["grad_scaler_state_dict"].get("scale", 0))
    require(math.isfinite(scale) and scale > 0, "Invalid saved GradScaler state.")
    groups = cp["optimizer_state_dict"]["param_groups"]
    require(len(groups) == 1, "Unexpected AdamW parameter-group structure.")
    require(groups[0]["lr"] == old.LEARNING_RATE and groups[0]["weight_decay"] == old.WEIGHT_DECAY,
            "Saved optimizer LR/weight decay differs from the contract.")


def read_trusted_checkpoint(path: Path, contract: dict, protocol: dict) -> dict:
    # Only use your own project checkpoint. weights_only=False is needed for
    # the original checkpoint's Python/NumPy RNG state; never use remote files.
    cp = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(cp, dict), "Checkpoint is not a dictionary.")
    validate_checkpoint(cp, contract, protocol)
    return cp


def capture_rng(generator: Any) -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
        "train_generator": generator.get_state().clone(),
    }


def restore_rng(state: dict, generator: Any) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    require(len(state["torch_cuda"]) == torch.cuda.device_count(),
            "CUDA RNG device count changed; stop for review.")
    torch.cuda.set_rng_state_all([tensor.cpu() for tensor in state["torch_cuda"]])
    generator.set_state(state["train_generator"].cpu())


class StepCounter:
    """Public optimizer hook: did GradScaler really call optimizer.step()?"""
    def __init__(self, optimizer: Any):
        self.count = 0
        self.handle = optimizer.register_step_post_hook(self._after_step)

    def _after_step(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        self.count += 1

    def close(self) -> None:
        self.handle.remove()


def checked_amp_step(model: Any, optimizer: Any, scaler: Any, counter: StepCounter,
                     clip_norm: float) -> dict:
    """Called AFTER scaler.scale(loss).backward(), outside autocast.

    Bad gradients are never clipped, zero-filled, or applied to parameters.
    scaler.step() is called in BOTH branches, followed by scaler.update().
    """
    scale_before = float(scaler.get_scale())
    require(math.isfinite(scale_before) and scale_before > 0, "Invalid AMP scale.")
    scaler.unscale_(optimizer)
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    missing = [name for name, p in named if p.grad is None]
    require(not missing, f"Trainable tensors did not receive gradients: {missing[:8]}")
    finite_flags = torch.stack([torch.isfinite(p.grad).all() for _, p in named]).cpu().tolist()
    bad_names = [name for (name, _), ok in zip(named, finite_flags) if not ok]
    gradient_norm = None
    if not bad_names:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
        gradient_norm = float(norm.item())
        require(math.isfinite(gradient_norm), "Nonfinite global gradient norm.")
    before_calls = counter.count
    scaler.step(optimizer)       # Safely SKIPS when unscale_ recorded NaN/Inf.
    scaler.update()              # Backoff is still necessary after a skipped step.
    did_step = counter.count == before_calls + 1
    scale_after = float(scaler.get_scale())
    require(counter.count - before_calls in (0, 1), "Unexpected optimizer-step count.")
    require(math.isfinite(scale_after) and scale_after > 0, "AMP scale became invalid.")
    if bad_names:
        require(not did_step, "Invalid gradients were unexpectedly applied; stop immediately.")
        require(scale_after < scale_before, "AMP did not reduce its scale after nonfinite gradients.")
    else:
        require(did_step, "Finite-gradient optimizer step was unexpectedly skipped.")
        require(scale_after >= scale_before, "Unexpected AMP scale reduction on finite gradients.")
    optimizer.zero_grad(set_to_none=True)
    return {"executed": did_step, "gradient_norm": gradient_norm, "bad_names": bad_names,
            "scale_before": scale_before, "scale_after": scale_after}


def amp_self_test(device: Any) -> dict:
    """Tiny artificial model, not B0: check finite update and forced overflow."""
    model = nn.Linear(4, 2, bias=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device.type, init_scale=8.0, growth_interval=100)
    counter = StepCounter(optimizer)
    x = torch.ones(3, 4, device=device)
    before = model.weight.detach().clone()
    try:
        scaler.scale(model(x).square().mean()).backward()
        normal = checked_amp_step(model, optimizer, scaler, counter, 1.0)
        require(normal["executed"] and not torch.equal(before, model.weight), "Finite-step self-test failed.")
        before = model.weight.detach().clone()
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        scaler.scale(model(x).square().mean()).backward()
        model.weight.grad.view(-1)[0] = float("inf")
        overflow = checked_amp_step(model, optimizer, scaler, counter, 1.0)
        require(not overflow["executed"] and torch.equal(before, model.weight),
                "Overflow self-test changed model weights.")
        for key, state in optimizer_before["state"].items():
            for name, value in state.items():
                if torch.is_tensor(value):
                    require(torch.equal(value, optimizer.state_dict()["state"][key][name]),
                            "Overflow self-test changed optimizer state.")
        return {"finite_update": "PASS", "overflow_skip": "PASS", "optimizer_preserved": "PASS",
                "scale_before_overflow": overflow["scale_before"],
                "scale_after_overflow": overflow["scale_after"]}
    finally:
        counter.close()
        del model, optimizer


def bind_and_back_up(contract: dict, protocol: dict) -> tuple[dict, Path]:
    """Snapshot original outputs ONCE. Never overwrite an original file."""
    fixed_hashes = {}
    for filename, expected in ACCEPTED_INPUTS.items():
        path = ROOT / "outputs/metrics/scenario2" / filename
        require(sha256(path) == expected, f"Accepted input hash changed: {filename}")
        fixed_hashes[filename] = expected
    code_hashes = {name: sha256(ROOT / "scripts/data" / name) for name in SCRIPT_NAMES}
    identities = {
        "accepted_inputs": fixed_hashes,
        "contract_sha256": sha256(old.CONTRACT_FILE),
        "full_protocol_sha256": sha256(old.PROTOCOL_FILE),
        "script_hashes_at_continuation": code_hashes,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "numpy": str(np.__version__), "cuda_runtime": torch.version.cuda},
        "engineering_policy": POLICY,
    }
    if BINDING.exists():
        binding = read_json(BINDING)
        require(binding["identities"] == identities,
                "Bound code/environment/input differs. Do not alter a continuing run silently.")
        for name, expected in binding["backups"].items():
            require(sha256(BACKUP / name) == expected, f"Preserved backup changed: {name}")
    else:
        require(not LATEST.exists(), "Unbound continuation checkpoint exists; manual review required.")
        require(old.LATEST_CHECKPOINT_FILE.is_file(),
                "No completed B0 latest checkpoint. Refusing to start from scratch.")
        cp = read_trusted_checkpoint(old.LATEST_CHECKPOINT_FILE, contract, protocol)
        backups = {}
        inputs = {
            "original_latest.pt": old.LATEST_CHECKPOINT_FILE,
            "original_best.pt": old.BEST_CHECKPOINT_FILE,
            "original_protocol.json": old.PROTOCOL_FILE,
            "original_contract.json": old.CONTRACT_FILE,
        }
        if old.HISTORY_FILE.is_file():
            inputs["original_history.csv"] = old.HISTORY_FILE
        for name, path in inputs.items():
            backups[name] = verified_copy(path, BACKUP / name)
        for name in SCRIPT_NAMES:
            verified_copy(ROOT / "scripts/data" / name, BACKUP / "scripts" / name)
        best = read_trusted_checkpoint(BACKUP / "original_best.pt", contract, protocol)
        require(int(best["epoch"]) == int(cp["best_epoch"]),
                "Original best/latest disagree. No training was started.")
        require(float(best["best_ema_validation_mse"]) == float(cp["best_ema_validation_mse"]),
                "Original best MSE differs from latest metadata.")
        binding = {"created_utc": utc_now(), "identities": identities, "backups": backups,
                   "source_complete_epoch": int(cp["epoch"]),
                   "original_latest_path": str(old.LATEST_CHECKPOINT_FILE),
                   "previous_run_prefix_preserved": True}
        save_json(BINDING, binding)
    source = LATEST if LATEST.is_file() else BACKUP / "original_latest.pt"
    # No auto-fallback from a corrupt new latest to an older epoch.
    return binding, source


def verify_hdf5_sources(split: Any, records: list) -> str:
    """Opaque SHA256 file reads, not Test EEG dataset indexing."""
    paths = {record.subject_id: record.h5_path for record in records}
    bundle = []
    for position, (subject, path) in enumerate(sorted(paths.items()), 1):
        expected = split.loc[split.subject_id.astype(str) == subject, "source_output_sha256"].unique()
        require(len(expected) == 1, f"Ambiguous HDF5 identity for {subject}")
        measured = sha256(path)
        require(measured == str(expected[0]), f"Processed HDF5 changed: {subject}")
        bundle.append([subject, measured])
        if position == len(paths):
            print(f"Source HDF5 hashes verified: {position} (opaque integrity reads only)")
    return hashlib.sha256(json_bytes(bundle)).hexdigest()


class IndexedDataset:
    """Adds row IDs for exact accounting; signal loading is unchanged."""
    def __init__(self, base: Any):
        self.base = base
    def __len__(self) -> int:
        return len(self.base)
    def __getitem__(self, index: int) -> dict:
        item = self.base[index]
        item["record_index"] = index
        return item


def cpu_state(model: Any) -> dict:
    return {name: value.detach().to("cpu", copy=True) for name, value in model.state_dict().items()}


def save_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    with temporary.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    # Validate a CPU round-trip before replacing any destination.
    readback = torch.load(temporary, map_location="cpu", weights_only=False)
    require(readback.get("epoch") == payload.get("epoch"), "Saved checkpoint epoch verification failed.")
    for name, value in payload["ema_state_dict"].items():
        require(torch.equal(value.detach().cpu(), readback["ema_state_dict"][name]),
                f"Saved EMA verification failed: {name}")
    del readback
    os.replace(temporary, path)


def publish_epoch(payload: dict) -> None:
    candidate = CHECKPOINTS / "candidate_complete.pt"
    save_torch(candidate, payload)
    if LATEST.exists():
        verified_copy(LATEST, PREVIOUS, immutable=False)
    os.replace(candidate, LATEST)
    # latest.pt, not the CSV or state.json, is the authoritative complete epoch.


def export_best(cp: dict) -> None:
    best_payload = {
        "status": "B0_FULL_BEST_EMA_EXPORT", "epoch": int(cp["best_epoch"]),
        "contract": cp["contract"], "full_training_protocol": cp["full_training_protocol"],
        "best_ema_validation_mse": float(cp["best_ema_validation_mse"]),
        "ema_state_dict": cp["best_ema_state_dict"],
        "model_state_dict": cp["best_ema_state_dict"],
        "model_state_dict_is_ema": True,
        "selection": "lowest deterministic EMA Validation epsilon MSE",
        "not_an_optimizer_resume_checkpoint": True, "final_test_accessed": False,
    }
    save_torch(BEST, best_payload)


def write_history(history: list[dict]) -> None:
    fields = list(history[0])
    for row in history:
        fields.extend(key for key in row if key not in fields)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(history)
    atomic_bytes(LOGS / "history.csv", stream.getvalue().encode("utf-8"))


def plot_history(history: list[dict]) -> None:
    epochs = [row["epoch"] for row in history]
    specifications = (
        ("loss.png", "B0 epsilon-prediction MSE", "MSE",
         [("train_mse", "Train (online model)"), ("ema_validation_mse", "Validation (EMA)")]),
        ("gradient.png", "B0 gradient norms before clipping", "Gradient L2 norm",
         [("mean_gradient_norm", "Mean"), ("max_gradient_norm", "Maximum")]),
        ("amp_skips.png", "B0 AMP skipped optimizer steps", "Skipped steps",
         [("amp_skipped_steps", "Count; earlier unrecorded epochs omitted")]),
        ("amp_scale.png", "B0 loss-scaling factor", "GradScaler scale", [("amp_scale_end", "Scale")]),
        ("throughput.png", "B0 epoch throughput (includes Validation)", "Train presentations / second",
         [("train_windows_per_second", "Epoch throughput")]),
        ("memory.png", "B0 GPU peak allocation", "MiB", [("peak_allocated_mib", "Peak allocation")]),
    )
    for filename, title, ylabel, series in specifications:
        fig, ax = plt.subplots(figsize=(9, 5))
        for key, label in series:
            values = [row.get(key, float("nan")) for row in history]
            ax.plot(epochs, values, label=label)
        ax.set(title=title, xlabel="Complete epoch", ylabel=ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        temporary = FIGURES / (filename + ".tmp")
        fig.savefig(temporary, format="png", dpi=160)
        plt.close(fig)
        os.replace(temporary, FIGURES / filename)


def run(args: Any) -> None:
    check_no_other_training_process()
    check_environment_and_disk()
    load_runtime()
    require(torch.cuda.is_available(), "CUDA not available. No CPU fallback and no fresh run.")
    require(str(torch.__version__) == "2.11.0+cu128", "PyTorch version differs from the reported original run.")
    device = torch.device("cuda:0")
    old.set_seed(old.SEED)
    print("\nSTEP 1 - VERIFY EXISTING CONTRACT; NEVER START FROM SCRATCH")
    require(old.PROTOCOL_FILE.is_file(), "Existing full-training protocol is missing; will not recreate it.")
    protocol, contract = read_json(old.PROTOCOL_FILE), read_json(old.CONTRACT_FILE)
    require(protocol == old.FULL_PROTOCOL, "Original code/protocol mismatch.")
    require(old.BATCH_SIZE == 64 and old.MAX_EPOCHS == 200 and old.TIMESTEPS == 1000,
            "This continuation expects the already-agreed B0 configuration.")
    require(old.LEARNING_RATE == 2e-4 and old.WEIGHT_DECAY == 1e-4 and old.EMA_DECAY == 0.9999,
            "Optimizer/EMA settings changed.")
    require(old.GRADIENT_CLIP_NORM == 1.0 and old.MIXED_PRECISION is True and old.COSINE_S == 0.008,
            "Numerical/diffusion settings changed.")
    require(contract["conditioning"]["subject_condition"] is False and
            contract["diffusion"]["prediction_target"] == "epsilon", "Not the expected B0 contract.")
    binding, source = bind_and_back_up(contract, protocol)
    cp = read_trusted_checkpoint(source, contract, protocol)
    binding_hash = sha256(BINDING)
    if source == LATEST:
        require(cp.get("continuation_binding_sha256") == binding_hash, "Continuation checkpoint binding changed.")
    completed = int(cp["epoch"])
    print(f"READ-ONLY source checkpoint: {source}")
    print(f"Completed epoch: {completed}; next epoch: {completed + 1}")
    print(f"Verified original backup: {BACKUP}")
    print(f"New outputs only: {RUN}")
    event("resume_source", path=str(source), sha256=sha256(source), completed_epoch=completed)
    estimated_checkpoint = source.stat().st_size + 40 * 1024**2
    milestones_left = sum(1 for e in range(completed + 1, old.MAX_EPOCHS + 1)
                          if e % old.CHECKPOINT_EVERY_EPOCHS == 0)
    required_free = (milestones_left + 4) * estimated_checkpoint + GIB
    check_environment_and_disk(required_free)
    print(f"Conservative additional disk reserve: {required_free / GIB:.2f} GiB")

    print("\nSTEP 2 - VERIFY INPUTS AND BUILD THE SAME TRAIN / VALIDATION LOADERS")
    split = pd.read_csv(old.FINAL_SPLIT_MANIFEST)
    scaler_data = read_json(old.NORMALIZATION_FILE)
    train_records, train_mapping = old.build_train_window_index(split, scaler_data)
    val_records, val_mapping = old.build_validation_window_index(split)
    require(len(train_records) == 47835 and len(val_records) == 9635 and train_mapping == val_mapping,
            "Train/Validation identity/count mismatch.")
    require(len(train_mapping) == 198, "Subject count changed.")
    source_bundle = verify_hdf5_sources(split, train_records)
    if source == LATEST:
        require(cp["continuation_source_bundle_sha256"] == source_bundle, "HDF5 source bundle changed.")
    train_base = old.CachedRealTrainWindowDataset(train_records, scaler_data, max_open_files=8)
    val_base = old.CachedRealTrainWindowDataset(val_records, scaler_data, max_open_files=8)
    try:
        generator = torch.Generator()
        train_loader = DataLoader(IndexedDataset(train_base), batch_size=64, shuffle=True,
                                  num_workers=0, pin_memory=True, drop_last=False, generator=generator)
        val_loader = DataLoader(val_base, batch_size=64, shuffle=False,
                                num_workers=0, pin_memory=True, drop_last=False)
        require(len(train_loader) == 748 and len(val_loader) == 151, "Batch layout changed.")

        print("\nSTEP 3 - RESTORE MODEL, EMA, ADAMW, GRADSCALER AND CPU/CUDA RNG")
        model = old.B0StateOnlyUNet1D(contract).to(device)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        model.load_state_dict(cp["model_state_dict"], strict=True)
        ema.load_state_dict(cp["ema_state_dict"], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=old.LEARNING_RATE, weight_decay=old.WEIGHT_DECAY)
        # PyTorch restores state onto the corresponding parameter devices;
        # do NOT blindly move AdamW's CPU step counters onto CUDA.
        optimizer.load_state_dict(cp["optimizer_state_dict"])
        grad_scaler = torch.amp.GradScaler("cuda", enabled=True)
        grad_scaler.load_state_dict(cp["grad_scaler_state_dict"])
        for parameter, state in optimizer.state.items():
            for key in ("exp_avg", "exp_avg_sq"):
                require(state[key].device == parameter.device, f"Wrong restored AdamW device: {key}")
        schedule = {key: value.to(device=device, dtype=torch.float32)
                    for key, value in old.build_cosine_schedule(old.TIMESTEPS, old.COSINE_S).items()}
        history = copy.deepcopy(cp["history"])
        best_epoch = int(cp["best_epoch"])
        best_mse = float(cp["best_ema_validation_mse"])
        cumulative_seconds = float(cp["cumulative_training_seconds"])
        total_skips = int(cp.get("continuation_total_amp_skips", 0))
        if "best_ema_state_dict" in cp:
            best_ema = cp["best_ema_state_dict"]
        else:
            old_best = read_trusted_checkpoint(BACKUP / "original_best.pt", contract, protocol)
            best_ema = old_best["ema_state_dict"]
            del old_best
        finite_tree(best_ema, "best_ema_state_dict")
        saved_rng = {**cp["rng_state"], "train_generator": cp["train_generator_state"]}
        restore_rng(saved_rng, generator)

        print("\nSTEP 4 - TEST AMP RECOVERY AND REPLAY SAVED EMA VALIDATION")
        # Artificial self-test consumes RNG; restore it afterwards.
        probe_rng = capture_rng(generator)
        self_test = amp_self_test(device)
        restore_rng(probe_rng, generator)
        observed = old.evaluate_validation_mse(ema, val_loader, schedule, device)
        expected = float(history[-1]["ema_validation_mse"])
        difference = abs(observed - expected)
        require(math.isfinite(observed) and difference <= REPLAY_ATOL,
                f"Saved Validation replay mismatch: saved={expected}, replay={observed}; not continuing.")
        restore_rng(probe_rng, generator)
        check = {"status": "RESUME_CHECK_PASS", "source_epoch": completed, "next_epoch": completed + 1,
                 "saved_ema_validation_mse": expected, "replayed_ema_validation_mse": observed,
                 "difference": difference, "tolerance": REPLAY_ATOL, "amp_self_test": self_test,
                 "source_hdf5_bundle_sha256": source_bundle, "rng_restored_after_check": True,
                 "b0_optimizer_steps_in_check": 0, "final_test_accessed": False}
        save_json(METRICS / "resume_check.json", check)
        print(f"AMP finite/overflow tests: PASS; Validation replay difference: {difference:.3e}")
        print(f"RESUME CHECK PASS - completed {completed}; next {completed + 1}")
        if not args.execute:
            print("CHECK ONLY. No B0 training update. Run THIS file with --execute to continue.")
            return
        require(completed <= args.until_epoch, "Requested pause epoch is earlier than the checkpoint.")
        print("\nSTEP 5 - CONTINUE B0 (ORIGINAL RUN/SCIENTIFIC SETTINGS UNCHANGED)")
        counter = StepCounter(optimizer)
        # Initial saved best is embedded, so later exports cannot lose old best weights.
        cp.update({"best_ema_state_dict": best_ema,
                   "continuation_patch_id": PATCH_ID, "engineering_policy": POLICY,
                   "continuation_binding_sha256": binding_hash,
                   "continuation_source_bundle_sha256": source_bundle,
                   "continuation_total_amp_skips": total_skips})
        # Seed the new namespace with the restored COMPLETE checkpoint first.
        # This is a copy/continuation, never a new randomly initialized run.
        if not LATEST.exists():
            publish_epoch(cp)
        # Rebuild a stale/missing export from the embedded best EMA even when
        # the previous process stopped between checkpoint and report writes.
        export_best(cp)
        del cp
        try:
            for epoch in range(completed + 1, min(args.until_epoch, old.MAX_EPOCHS) + 1):
                check_environment_and_disk(GIB + 3 * estimated_checkpoint)
                torch.cuda.reset_peak_memory_stats()
                epoch_start = time.perf_counter()
                model.train()
                seen = np.zeros(len(train_records), dtype=np.bool_)
                loss_sum = 0.0
                presented = 0
                update_examples = 0
                grad_norms = []
                skipped = consecutive = skipped_examples = 0
                update_steps = 0
                event("epoch_started", epoch=epoch, source_complete_epoch=completed)
                save_json(STATE, {"status": "RUNNING", "completed_epoch": completed,
                                  "active_epoch": epoch, "best_epoch": best_epoch,
                                  "latest_checkpoint": str(LATEST if LATEST.exists() else source),
                                  "final_test_accessed": False})
                for batch_index, batch in enumerate(train_loader, 1):
                    indexes = batch["record_index"].numpy()
                    require(len(np.unique(indexes)) == len(indexes) and not seen[indexes].any(),
                            "Duplicate Train-window presentation in this epoch.")
                    seen[indexes] = True
                    x0 = batch["x"].to(device, non_blocking=True)
                    states = batch["y"].to(device, non_blocking=True)
                    require(bool(torch.isfinite(x0).all().item()), "Nonfinite input; not an AMP skip.")
                    t = torch.randint(0, old.TIMESTEPS, (len(x0),), device=device, dtype=torch.long)
                    epsilon = torch.randn_like(x0)
                    xt = old.q_sample(x0, t, epsilon, schedule["sqrt_alpha_cumprod"],
                                      schedule["sqrt_one_minus_alpha_cumprod"])
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
                        prediction = model(xt, t, states)
                        loss = nn.functional.mse_loss(prediction, epsilon)
                    require(bool(torch.isfinite(loss).item()) and bool(torch.isfinite(prediction).all().item()),
                            f"Nonfinite forward/loss at epoch {epoch}, batch {batch_index}; hard stop.")
                    loss_value = float(loss.item())
                    grad_scaler.scale(loss).backward()
                    step = checked_amp_step(model, optimizer, grad_scaler, counter, old.GRADIENT_CLIP_NORM)
                    presented += len(x0)
                    loss_sum += loss_value * len(x0)   # All finite presentations, including skipped updates.
                    if step["executed"]:
                        old.update_ema(ema, model, old.EMA_DECAY)
                        grad_norms.append(step["gradient_norm"])
                        update_steps += 1
                        update_examples += len(x0)
                        consecutive = 0
                    else:
                        skipped += 1
                        total_skips += 1
                        consecutive += 1
                        skipped_examples += len(x0)
                        event("amp_nonfinite_gradient_step_skipped", epoch=epoch, batch=batch_index,
                              loss=loss_value, scale_before=step["scale_before"], scale_after=step["scale_after"],
                              nonfinite_parameter_names=step["bad_names"], ema_updated=False,
                              sample_keys=[f"{train_records[int(k)].block_uid}:window-"
                                           f"{train_records[int(k)].candidate_window_index}" for k in indexes],
                              timesteps=t.cpu().tolist(), cause="NOT_PROVEN; AMP backoff applied")
                        print(f"AMP SKIP | epoch={epoch} batch={batch_index} | scale "
                              f"{step['scale_before']:g}->{step['scale_after']:g} | optimizer=SKIP EMA=SKIP")
                        require(skipped <= MAX_SKIPS_PER_EPOCH and consecutive < MAX_CONSECUTIVE_SKIPS,
                                "Repeated nonfinite gradients: safety stop. Preserve logs and investigate.")
                    if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(train_loader):
                        print(f"epoch {epoch:3d}/200 | batch {batch_index:3d}/748 | "
                              f"train_MSE={loss_sum/presented:.6f} | amp_skips={skipped} | "
                              f"scale={grad_scaler.get_scale():g}")
                require(presented == 47835 and seen.all(), "Full-epoch Train accounting failed.")
                require(update_steps + skipped == 748 and update_examples + skipped_examples == presented,
                        "Optimizer/presentation accounting failed.")
                require(bool(grad_norms), "No successful optimizer steps in the epoch.")
                # Detect invalid stored weights before any checkpoint can be published.
                finite_tree(model.state_dict(), "live_model")
                finite_tree(ema.state_dict(), "live_ema")
                finite_tree(optimizer.state_dict(), "live_optimizer")
                validation_mse = old.evaluate_validation_mse(ema, val_loader, schedule, device)
                require(math.isfinite(validation_mse), "Nonfinite Validation loss.")
                torch.cuda.synchronize()
                epoch_seconds = time.perf_counter() - epoch_start
                cumulative_seconds += epoch_seconds
                improved = validation_mse < best_mse
                if improved:
                    best_epoch, best_mse, best_ema = epoch, validation_mse, cpu_state(ema)
                row = {
                    "epoch": epoch, "train_mse": loss_sum / presented, "ema_validation_mse": validation_mse,
                    "mean_gradient_norm": float(np.mean(grad_norms)), "max_gradient_norm": float(np.max(grad_norms)),
                    "epoch_seconds": epoch_seconds, "train_windows_per_second": presented / epoch_seconds,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                    "best_epoch": best_epoch, "best_ema_validation_mse": best_mse,
                    "amp_skipped_steps": skipped, "amp_scale_end": float(grad_scaler.get_scale()),
                    "optimizer_steps": update_steps, "windows_presented": presented,
                    "windows_with_optimizer_update": update_examples, "amp_skipped_windows": skipped_examples,
                }
                history.append(row)
                payload = old.build_checkpoint(
                    epoch=epoch, model=model, ema_model=ema, optimizer=optimizer, grad_scaler=grad_scaler,
                    train_generator=generator, history=history, best_epoch=best_epoch,
                    best_ema_validation_mse=best_mse, cumulative_training_seconds=cumulative_seconds, contract=contract)
                payload.update({"continuation_patch_id": PATCH_ID, "engineering_policy": POLICY,
                                "continuation_binding_sha256": binding_hash,
                                "continuation_source_bundle_sha256": source_bundle,
                                "continuation_total_amp_skips": total_skips, "best_ema_state_dict": best_ema})
                # Save the complete numerical state BEFORE plotting/report generation.
                publish_epoch(payload)
                completed = epoch
                event("epoch_committed", epoch=epoch, checkpoint=str(LATEST), sha256=sha256(LATEST),
                      amp_skips=skipped, optimizer_steps=update_steps, best_epoch=best_epoch)
                if improved or not BEST.exists():
                    export_best(payload)
                if epoch % old.CHECKPOINT_EVERY_EPOCHS == 0:
                    verified_copy(LATEST, CHECKPOINTS / f"epoch_{epoch:03d}.pt")
                write_history(history)
                plot_history(history)
                save_json(STATE, {"status": "EPOCH_COMPLETE", "completed_epoch": epoch, "max_epochs": 200,
                                  "best_epoch": best_epoch, "best_ema_validation_mse": best_mse,
                                  "latest_train_mse": row["train_mse"], "amp_skips_this_epoch": skipped,
                                  "continuation_total_amp_skips": total_skips, "latest_checkpoint": str(LATEST),
                                  "best_checkpoint": str(BEST), "final_test_accessed": False})
                print(f"\nEPOCH {epoch} COMPLETE | Train={row['train_mse']:.6f} | EMA Val={validation_mse:.6f}\n"
                      f"best={best_mse:.6f} @ {best_epoch} | successful_updates={update_steps} | "
                      f"AMP skips={skipped} | time={epoch_seconds/60:.2f} min\n"
                      f"Checkpoint safely written: {LATEST}\n")
                # Optional graceful pause; not an early-stopping/model-selection rule.
                if (RUN / "STOP_AFTER_EPOCH.txt").exists():
                    print("STOP_AFTER_EPOCH.txt found: pausing after a complete saved epoch.")
                    break
            status = "B0_FULL_TRAINING_COMPLETE" if completed == 200 else "B0_FULL_TRAINING_PAUSED"
            if completed == 200:
                verified_copy(LATEST, CHECKPOINTS / "final.pt")
            summary = {
                "status": status, "completed_epoch": completed, "max_epochs": 200,
                "best_epoch": best_epoch, "best_ema_validation_mse": best_mse,
                "last_train_mse": history[-1]["train_mse"],
                "cumulative_epoch_compute_hours": cumulative_seconds / 3600,
                "continuation_total_amp_skips": total_skips,
                "all_original_files_preserved": True,
                "continuation_binding": str(BINDING), "latest_checkpoint": str(LATEST),
                "best_checkpoint": str(BEST), "new_output_root": str(RUN),
                "history": str(LOGS / "history.csv"), "validation_accessed": True,
                "final_test_accessed": False,
                "sampling_performed": False,
                "note": "Resume-only continuation; no model hyperparameter or selection-rule change.",
            }
            save_json(METRICS / "result.json", summary)
            save_json(STATE, summary)
            event("run_complete_or_paused", **summary)
            print(f"\n{status}\nBest EMA Validation MSE: {best_mse:.6f} @ epoch {best_epoch}\nOutputs: {RUN}")
        finally:
            counter.close()
    finally:
        train_base.close()
        val_base.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Resume real B0 training after all checks pass.")
    parser.add_argument("--check-only", action="store_true", help="Explicit preflight-only mode (also the default).")
    parser.add_argument("--until-epoch", type=int, default=200,
                        help="Optional graceful pause epoch; total frozen protocol remains 200.")
    args = parser.parse_args()
    require(not (args.execute and args.check_only), "Choose --execute OR --check-only.")
    require(1 <= args.until_epoch <= 200, "--until-epoch must be within 1..200.")
    for path in (CHECKPOINTS, LOGS, METRICS, FIGURES, BACKUP):
        path.mkdir(parents=True, exist_ok=True)
    # Lock before touching logs/state, so a duplicate invocation cannot alter the active run.
    with single_instance():
        label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
        with (LOGS / f"console_{label}.log").open("x", encoding="utf-8") as logfile:
            with contextlib.redirect_stdout(Tee(sys.stdout, logfile)), contextlib.redirect_stderr(Tee(sys.stderr, logfile)):
                print("SCENARIO 2 - B0 AMP-SAFE CHECKPOINT CONTINUATION")
                print(f"Mode: {'EXECUTE' if args.execute else 'CHECK ONLY'}")
                try:
                    run(args)
                except BaseException as exc:
                    if isinstance(exc, (SystemExit, GeneratorExit)):
                        raise
                    failure = {"status": "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "BLOCKED",
                               "utc": utc_now(), "type": type(exc).__name__, "message": str(exc),
                               "new_latest_if_present": str(LATEST),
                               "epoch_completion_authority": "checkpoint history, not unfinished console progress",
                               "original_checkpoints_untouched": True}
                    try:
                        save_json(METRICS / f"failure_{label}.json", failure)
                        event("run_stopped", **failure)
                    except Exception as log_error:
                        print(f"Could not write diagnostic evidence: {log_error}", file=sys.stderr)
                    traceback.print_exc()
                    print("STOPPED. No fallback to random initialization. Keep old and new checkpoints.")
                    raise SystemExit(130 if isinstance(exc, KeyboardInterrupt) else 1)


if __name__ == "__main__":
    main()
