from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn


# ============================================================
# Paths
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

CONTRACT_FILE = (
    REPO_ROOT
    / "outputs"
    / "metrics"
    / "scenario2"
    / "b0_state_only_ddpm_contract_v1.json"
)


# ============================================================
# Small helpers
# ============================================================

def valid_group_count(
    channels: int,
    preferred: int = 8,
) -> int:
    """
    Pick the largest GroupNorm group count <= preferred
    that divides channels exactly.
    """

    for groups in range(
        min(preferred, channels),
        0,
        -1,
    ):
        if channels % groups == 0:
            return groups

    return 1


# ============================================================
# Sinusoidal timestep embedding
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):

    def __init__(
        self,
        dim: int,
    ) -> None:
        super().__init__()

        if dim % 2 != 0:
            raise ValueError(
                "Time embedding dimension must be even."
            )

        self.dim = dim

    def forward(
        self,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:

        if timesteps.ndim != 1:
            raise RuntimeError(
                f"Expected timesteps shape (B,), "
                f"got {timesteps.shape}"
            )

        half_dim = self.dim // 2

        exponent = (
            -math.log(10_000.0)
            * torch.arange(
                half_dim,
                device=timesteps.device,
                dtype=torch.float32,
            )
            / (half_dim - 1)
        )

        frequencies = torch.exp(
            exponent
        )

        angles = (
            timesteps.float()[:, None]
            * frequencies[None, :]
        )

        embedding = torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=1,
        )

        return embedding


# ============================================================
# Residual block
# ============================================================

class ResidualBlock1D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.norm1 = nn.GroupNorm(
            valid_group_count(
                in_channels
            ),
            in_channels,
        )

        self.activation1 = nn.SiLU()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        self.condition_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                condition_dim,
                out_channels,
            ),
        )

        self.norm2 = nn.GroupNorm(
            valid_group_count(
                out_channels
            ),
            out_channels,
        )

        self.activation2 = nn.SiLU()

        self.dropout = nn.Dropout(
            dropout
        )

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

        if in_channels != out_channels:
            self.skip = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
        else:
            self.skip = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:

        residual = self.skip(x)

        h = self.norm1(x)
        h = self.activation1(h)
        h = self.conv1(h)

        condition_term = (
            self.condition_projection(
                condition
            )
            .unsqueeze(-1)
        )

        h = h + condition_term

        h = self.norm2(h)
        h = self.activation2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + residual


# ============================================================
# Self-attention block
# ============================================================

class AttentionBlock1D(nn.Module):

    def __init__(
        self,
        channels: int,
        heads: int,
    ) -> None:
        super().__init__()

        if channels % heads != 0:
            raise ValueError(
                f"{channels=} must be divisible by "
                f"{heads=}"
            )

        self.norm = nn.GroupNorm(
            valid_group_count(
                channels
            ),
            channels,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = x

        h = self.norm(x)

        # B × C × T
        # ->
        # B × T × C
        h = h.transpose(
            1,
            2,
        )

        h, _ = self.attention(
            h,
            h,
            h,
            need_weights=False,
        )

        h = h.transpose(
            1,
            2,
        )

        return residual + h


# ============================================================
# Down level
# ============================================================

class DownLevel(nn.Module):

    def __init__(
        self,
        channels: int,
        condition_dim: int,
        residual_blocks: int,
        dropout: float,
        use_attention: bool,
        attention_heads: int,
        next_channels: int | None,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ResidualBlock1D(
                    channels,
                    channels,
                    condition_dim,
                    dropout,
                )
                for _ in range(
                    residual_blocks
                )
            ]
        )

        if use_attention:
            self.attention = AttentionBlock1D(
                channels,
                attention_heads,
            )
        else:
            self.attention = nn.Identity()

        if next_channels is None:
            self.downsample = None
        else:
            self.downsample = nn.Conv1d(
                channels,
                next_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        for block in self.blocks:
            x = block(
                x,
                condition,
            )

        x = self.attention(x)

        skip = x

        if self.downsample is not None:
            x = self.downsample(x)

        return x, skip


# ============================================================
# Up level
# ============================================================

class UpLevel(nn.Module):

    def __init__(
        self,
        current_channels: int,
        skip_channels: int,
        output_channels: int,
        condition_dim: int,
        residual_blocks: int,
        dropout: float,
        use_attention: bool,
        attention_heads: int,
        next_channels: int | None,
    ) -> None:
        super().__init__()

        blocks = []

        # First block consumes concatenated
        # current + skip channels.
        blocks.append(
            ResidualBlock1D(
                current_channels
                + skip_channels,

                output_channels,

                condition_dim,

                dropout,
            )
        )

        for _ in range(
            residual_blocks - 1
        ):
            blocks.append(
                ResidualBlock1D(
                    output_channels,
                    output_channels,
                    condition_dim,
                    dropout,
                )
            )

        self.blocks = nn.ModuleList(
            blocks
        )

        if use_attention:
            self.attention = AttentionBlock1D(
                output_channels,
                attention_heads,
            )
        else:
            self.attention = nn.Identity()

        if next_channels is None:
            self.upsample = None
        else:
            self.upsample = nn.ConvTranspose1d(
                output_channels,
                next_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:

        if x.shape[-1] != skip.shape[-1]:
            raise RuntimeError(
                f"Skip temporal mismatch: "
                f"{x.shape} vs {skip.shape}"
            )

        x = torch.cat(
            [
                x,
                skip,
            ],
            dim=1,
        )

        for block in self.blocks:
            x = block(
                x,
                condition,
            )

        x = self.attention(x)

        if self.upsample is not None:
            x = self.upsample(x)

        return x


# ============================================================
# B0 1D U-Net
# ============================================================

class B0StateOnlyUNet1D(nn.Module):

    def __init__(
        self,
        contract: dict,
    ) -> None:
        super().__init__()

        data_config = contract[
            "data"
        ]

        network_config = contract[
            "network"
        ]

        conditioning_config = contract[
            "conditioning"
        ]

        self.input_channels = int(
            data_config[
                "input_channels"
            ]
        )

        self.input_samples = int(
            data_config[
                "input_samples"
            ]
        )

        self.timesteps = int(
            contract[
                "diffusion"
            ][
                "timesteps"
            ]
        )

        base_channels = int(
            network_config[
                "base_channels"
            ]
        )

        multipliers = list(
            network_config[
                "channel_multipliers"
            ]
        )

        self.level_channels = [
            base_channels
            * int(multiplier)
            for multiplier
            in multipliers
        ]

        residual_blocks = int(
            network_config[
                "residual_blocks_per_level"
            ]
        )

        condition_dim = int(
            network_config[
                "time_embedding_dim"
            ]
        )

        state_embedding_dim = int(
            network_config[
                "state_embedding_dim"
            ]
        )

        dropout = float(
            network_config[
                "dropout"
            ]
        )

        attention_levels = set(
            int(level)
            for level
            in network_config[
                "attention"
            ][
                "levels"
            ]
        )

        attention_heads = int(
            network_config[
                "attention"
            ][
                "heads"
            ]
        )

        if conditioning_config[
            "subject_condition"
        ]:
            raise RuntimeError(
                "B0 must NOT use subject conditioning."
            )

        # ----------------------------------------------------
        # Condition embeddings
        # ----------------------------------------------------

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(
                base_channels
            ),

            nn.Linear(
                base_channels,
                condition_dim,
            ),

            nn.SiLU(),

            nn.Linear(
                condition_dim,
                condition_dim,
            ),
        )

        self.state_embedding = nn.Embedding(
            num_embeddings=2,
            embedding_dim=state_embedding_dim,
        )

        self.state_projection = nn.Sequential(
            nn.Linear(
                state_embedding_dim,
                condition_dim,
            ),

            nn.SiLU(),

            nn.Linear(
                condition_dim,
                condition_dim,
            ),
        )

        # ----------------------------------------------------
        # Input projection
        # ----------------------------------------------------

        self.input_projection = nn.Conv1d(
            self.input_channels,
            self.level_channels[0],
            kernel_size=3,
            padding=1,
        )

        # ----------------------------------------------------
        # Down path
        # ----------------------------------------------------

        down_levels = []

        for level, channels in enumerate(
            self.level_channels
        ):

            if level + 1 < len(
                self.level_channels
            ):
                next_channels = (
                    self.level_channels[
                        level + 1
                    ]
                )
            else:
                next_channels = None

            down_levels.append(
                DownLevel(
                    channels=channels,

                    condition_dim=(
                        condition_dim
                    ),

                    residual_blocks=(
                        residual_blocks
                    ),

                    dropout=dropout,

                    use_attention=(
                        level
                        in attention_levels
                    ),

                    attention_heads=(
                        attention_heads
                    ),

                    next_channels=(
                        next_channels
                    ),
                )
            )

        self.down_levels = nn.ModuleList(
            down_levels
        )

        # ----------------------------------------------------
        # Bottleneck
        # ----------------------------------------------------

        bottleneck_channels = (
            self.level_channels[-1]
        )

        self.mid_block1 = ResidualBlock1D(
            bottleneck_channels,
            bottleneck_channels,
            condition_dim,
            dropout,
        )

        self.mid_attention = AttentionBlock1D(
            bottleneck_channels,
            attention_heads,
        )

        self.mid_block2 = ResidualBlock1D(
            bottleneck_channels,
            bottleneck_channels,
            condition_dim,
            dropout,
        )

        # ----------------------------------------------------
        # Up path
        # ----------------------------------------------------

        up_levels = []

        reversed_levels = list(
            reversed(
                range(
                    len(
                        self.level_channels
                    )
                )
            )
        )

        current_channels = (
            bottleneck_channels
        )

        for level in reversed_levels:

            skip_channels = (
                self.level_channels[
                    level
                ]
            )

            output_channels = (
                skip_channels
            )

            if level > 0:
                next_channels = (
                    self.level_channels[
                        level - 1
                    ]
                )
            else:
                next_channels = None

            up_levels.append(
                UpLevel(
                    current_channels=(
                        current_channels
                    ),

                    skip_channels=(
                        skip_channels
                    ),

                    output_channels=(
                        output_channels
                    ),

                    condition_dim=(
                        condition_dim
                    ),

                    residual_blocks=(
                        residual_blocks
                    ),

                    dropout=dropout,

                    use_attention=(
                        level
                        in attention_levels
                    ),

                    attention_heads=(
                        attention_heads
                    ),

                    next_channels=(
                        next_channels
                    ),
                )
            )

            if next_channels is None:
                current_channels = (
                    output_channels
                )
            else:
                current_channels = (
                    next_channels
                )

        self.up_levels = nn.ModuleList(
            up_levels
        )

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        final_channels = (
            self.level_channels[0]
        )

        self.output_norm = nn.GroupNorm(
            valid_group_count(
                final_channels
            ),
            final_channels,
        )

        self.output_activation = nn.SiLU()

        self.output_projection = nn.Conv1d(
            final_channels,
            self.input_channels,
            kernel_size=3,
            padding=1,
        )

    # --------------------------------------------------------
    # Condition construction
    # --------------------------------------------------------

    def build_condition(
        self,
        timesteps: torch.Tensor,
        eye_state: torch.Tensor,
    ) -> torch.Tensor:

        if timesteps.ndim != 1:
            raise RuntimeError(
                "timesteps must have shape (B,)"
            )

        if eye_state.ndim != 1:
            raise RuntimeError(
                "eye_state must have shape (B,)"
            )

        if timesteps.shape[0] != (
            eye_state.shape[0]
        ):
            raise RuntimeError(
                "Condition batch sizes differ."
            )

        if torch.any(
            timesteps < 0
        ) or torch.any(
            timesteps >= self.timesteps
        ):
            raise RuntimeError(
                "Diffusion timestep out of range."
            )

        if torch.any(
            eye_state < 0
        ) or torch.any(
            eye_state > 1
        ):
            raise RuntimeError(
                "eye_state must be 0=EO or 1=EC."
            )

        time_condition = (
            self.time_embedding(
                timesteps
            )
        )

        state_condition = (
            self.state_projection(
                self.state_embedding(
                    eye_state
                )
            )
        )

        return (
            time_condition
            + state_condition
        )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        eye_state: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise RuntimeError(
                f"Expected x=(B,C,T), "
                f"got {x.shape}"
            )

        if x.shape[1] != (
            self.input_channels
        ):
            raise RuntimeError(
                f"Expected "
                f"{self.input_channels} channels, "
                f"got {x.shape}"
            )

        if x.shape[2] != (
            self.input_samples
        ):
            raise RuntimeError(
                f"Expected "
                f"{self.input_samples} time samples, "
                f"got {x.shape}"
            )

        condition = self.build_condition(
            timesteps,
            eye_state,
        )

        x = self.input_projection(x)

        skips = []

        # Down
        for down_level in (
            self.down_levels
        ):

            x, skip = down_level(
                x,
                condition,
            )

            skips.append(skip)

        # Middle
        x = self.mid_block1(
            x,
            condition,
        )

        x = self.mid_attention(x)

        x = self.mid_block2(
            x,
            condition,
        )

        # Up
        for (
            up_level,
            skip,
        ) in zip(
            self.up_levels,
            reversed(skips),
        ):

            x = up_level(
                x,
                skip,
                condition,
            )

        x = self.output_norm(x)

        x = self.output_activation(x)

        epsilon_pred = (
            self.output_projection(x)
        )

        if epsilon_pred.shape != (
            x.shape[0],
            self.input_channels,
            self.input_samples,
        ):
            raise RuntimeError(
                f"Unexpected model output: "
                f"{epsilon_pred.shape}"
            )

        return epsilon_pred


# ============================================================
# Parameter summary
# ============================================================

def parameter_summary(
    model: nn.Module,
) -> None:

    total = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    trainable = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    parameter_mib = (
        total
        * 4
        / 1024**2
    )

    print(
        f"total parameters: "
        f"{total:,}"
    )

    print(
        f"trainable parameters: "
        f"{trainable:,}"
    )

    print(
        f"FP32 parameter memory: "
        f"{parameter_mib:.2f} MiB"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 80)
    print(
        "SCENARIO 2 — "
        "B0 1D U-NET SHAPE TEST"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Step 1 — Read frozen contract
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 1 — LOAD B0 CONTRACT")
    print("=" * 80)

    if not CONTRACT_FILE.exists():
        raise FileNotFoundError(
            CONTRACT_FILE
        )

    with CONTRACT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:
        contract = json.load(f)

    if contract[
        "model_name"
    ] != "B0_STATE_ONLY_DDPM":
        raise RuntimeError(
            "Wrong B0 contract."
        )

    print(
        f"model: "
        f"{contract['model_name']}"
    )

    print(
        f"target: "
        f"{contract['diffusion']['prediction_target']}"
    )

    print(
        f"timesteps: "
        f"{contract['diffusion']['timesteps']}"
    )

    # --------------------------------------------------------
    # Step 2 — Device
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 2 — DEVICE")
    print("=" * 80)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required for B0 development."
        )

    device = torch.device(
        "cuda"
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA runtime: "
        f"{torch.version.cuda}"
    )

    # --------------------------------------------------------
    # Step 3 — Build model
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 3 — BUILD B0 U-NET")
    print("=" * 80)

    model = B0StateOnlyUNet1D(
        contract
    ).to(
        device
    )

    parameter_summary(
        model
    )

    # --------------------------------------------------------
    # Step 4 — Forward shape test
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 4 — FORWARD SHAPE TEST")
    print("=" * 80)

    batch_size = 4

    x = torch.randn(
        (
            batch_size,
            19,
            512,
        ),
        device=device,
    )

    timesteps = torch.tensor(
        [
            0,
            10,
            500,
            999,
        ],
        dtype=torch.long,
        device=device,
    )

    # EO / EC / EO / EC
    states = torch.tensor(
        [
            0,
            1,
            0,
            1,
        ],
        dtype=torch.long,
        device=device,
    )

    torch.cuda.reset_peak_memory_stats()

    model.eval()

    with torch.no_grad():

        output = model(
            x,
            timesteps,
            states,
        )

    torch.cuda.synchronize()

    print(
        f"input shape:  "
        f"{tuple(x.shape)}"
    )

    print(
        f"output shape: "
        f"{tuple(output.shape)}"
    )

    print(
        f"output dtype: "
        f"{output.dtype}"
    )

    print(
        f"finite output: "
        f"{bool(torch.isfinite(output).all())}"
    )

    print(
        f"peak GPU allocated: "
        f"{torch.cuda.max_memory_allocated() / 1024**2:.2f} MiB"
    )

    if output.shape != x.shape:
        raise RuntimeError(
            "B0 output shape differs from input."
        )

    if not torch.isfinite(
        output
    ).all():
        raise RuntimeError(
            "B0 output contains NaN/Inf."
        )

    print(
        "forward shape test: PASS"
    )

    # --------------------------------------------------------
    # Step 5 — Eye-state conditioning sanity
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 5 — STATE CONDITIONING SANITY")
    print("=" * 80)

    x_single = torch.randn(
        (
            1,
            19,
            512,
        ),
        device=device,
    )

    timestep_single = torch.tensor(
        [400],
        device=device,
        dtype=torch.long,
    )

    eo = torch.tensor(
        [0],
        device=device,
        dtype=torch.long,
    )

    ec = torch.tensor(
        [1],
        device=device,
        dtype=torch.long,
    )

    with torch.no_grad():

        output_eo = model(
            x_single,
            timestep_single,
            eo,
        )

        output_ec = model(
            x_single,
            timestep_single,
            ec,
        )

    state_difference = (
        torch.mean(
            torch.abs(
                output_eo
                - output_ec
            )
        )
        .item()
    )

    print(
        f"mean |EO output - EC output|: "
        f"{state_difference:.8f}"
    )

    if state_difference <= 0:
        raise RuntimeError(
            "State condition has no effect."
        )

    print(
        "state conditioning path: PASS"
    )

    # --------------------------------------------------------
    # Step 6 — Timestep conditioning sanity
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 6 — TIMESTEP CONDITIONING SANITY")
    print("=" * 80)

    t_early = torch.tensor(
        [10],
        device=device,
        dtype=torch.long,
    )

    t_late = torch.tensor(
        [900],
        device=device,
        dtype=torch.long,
    )

    with torch.no_grad():

        output_early = model(
            x_single,
            t_early,
            eo,
        )

        output_late = model(
            x_single,
            t_late,
            eo,
        )

    timestep_difference = (
        torch.mean(
            torch.abs(
                output_early
                - output_late
            )
        )
        .item()
    )

    print(
        f"mean |t=10 - t=900 output|: "
        f"{timestep_difference:.8f}"
    )

    if timestep_difference <= 0:
        raise RuntimeError(
            "Timestep condition has no effect."
        )

    print(
        "timestep conditioning path: PASS"
    )

    # --------------------------------------------------------
    # Step 7 — Invalid-condition checks
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 7 — FAIL-CLOSED CONDITION CHECKS")
    print("=" * 80)

    invalid_state_pass = False

    try:
        model(
            x_single,
            timestep_single,
            torch.tensor(
                [2],
                device=device,
            ),
        )

    except RuntimeError:
        invalid_state_pass = True

    if not invalid_state_pass:
        raise RuntimeError(
            "Invalid eye state was not rejected."
        )

    print(
        "invalid eye_state rejected: PASS"
    )

    invalid_timestep_pass = False

    try:
        model(
            x_single,
            torch.tensor(
                [1000],
                device=device,
            ),
            eo,
        )

    except RuntimeError:
        invalid_timestep_pass = True

    if not invalid_timestep_pass:
        raise RuntimeError(
            "Invalid timestep was not rejected."
        )

    print(
        "invalid timestep rejected: PASS"
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("B0 U-NET ARCHITECTURE SHAPE TEST PASS")
    print("=" * 80)

    print(
        "model training started: NO"
    )

    print(
        "optimizer created: NO"
    )

    print(
        "backward pass executed: NO"
    )

    print(
        "Validation accessed: NO"
    )

    print(
        "Final Test accessed: NO"
    )

    print(
        "\nNEXT:"
    )

    print(
        "ONE-BATCH B0 DIFFUSION "
        "FORWARD/BACKWARD SANITY"
    )


if __name__ == "__main__":
    main()