"""Train the policy/value network.

Usage:
    python -m cube_solver.train --steps 5000
    python -m cube_solver.train --labels exact --encoding pieces --out models/exact_pieces.pt
    python -m cube_solver.train --resume models/policy.pt --steps 5000 --lr 3e-4
"""
from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .cube import MoveSet
from .data import ENCODINGS, encode, exact_batch, input_dim, load_heldout, scramble_batch, training_exclusion
from .evaluate import policy_report
from .model import PolicyValueNet, load_checkpoint, save_checkpoint
from .search import network_evaluator
from .state_space import decode, load_distance_table


@dataclass
class TrainConfig:
    steps: int = 5000
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-5
    labels: str = "scramble"          # "scramble" (self-supervised) or "exact" (from the table)
    encoding: str = "onehot"
    half_turns: bool = False
    max_depth: int = 0                # 0 picks 20 (quarter turns) or 16 (half turns)
    curriculum_steps: int = 0         # grow scramble depth from 1 to max_depth over this many steps
    label_smoothing: float = 0.0
    value_weight: float = 0.1
    stem_width: int = 1024
    width: int = 512
    n_blocks: int = 3
    eval_every: int = 250
    seed: int = 0
    out: str = "models/policy.pt"
    resume: str = ""                  # checkpoint to keep training from; its architecture and encoding win


def validate(model, encoding, states, distances, table, move_set, device) -> tuple[float, float]:
    """Return (optimal-move rate averaged over distances, value MAE) on held-out states.

    Averaging per distance first keeps the tiny short-distance buckets from
    being drowned out (or over-weighted).
    """
    report = policy_report(network_evaluator(model, encoding, device), states, distances, table, move_set)
    buckets = np.unique(distances)
    optimal_rate = np.mean([report["top1"][distances == d].mean() for d in buckets])
    model.train()
    return float(optimal_rate), float(report["value_error"].mean())


def train(cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resuming reuses the saved weights, so the model shape and inputs must match the checkpoint.
    resumed, step_offset = None, 0
    if cfg.resume:
        resumed, saved = load_checkpoint(Path(cfg.resume), device)
        cfg = replace(cfg, encoding=saved["encoding"], half_turns=saved["half_turns"],
                      stem_width=saved["model"]["stem_width"], width=saved["model"]["width"],
                      n_blocks=saved["model"]["n_blocks"])
        step_offset = saved.get("step", saved["train"]["steps"])

    move_set = MoveSet.fixed_corner(cfg.half_turns)
    max_depth = cfg.max_depth or (16 if cfg.half_turns else 20)

    # The exact table grades every prediction; held-out states never reach training.
    table = load_distance_table(move_set)
    heldout = load_heldout(table, move_set)
    exclude = training_exclusion(table, heldout)
    val_states = decode(heldout["val"])
    val_distances = table[heldout["val"]].astype(np.float32)

    model_kwargs = dict(input_dim=input_dim(cfg.encoding), n_moves=len(move_set),
                        stem_width=cfg.stem_width, width=cfg.width, n_blocks=cfg.n_blocks)
    model = resumed if resumed is not None else PolicyValueNet(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # One-cycle: short warmup, then cosine decay to near zero.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, total_steps=cfg.steps)
    checkpoint_config = {"model": model_kwargs, "encoding": cfg.encoding,
                         "half_turns": cfg.half_turns, "train": asdict(cfg)}

    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".csv")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{n_params:,} parameters on {device}; logging to {log_path}")

    best_rate = -1.0
    if resumed is not None:
        # Only overwrite the checkpoint once this run actually beats it.
        best_rate, _ = validate(model, cfg.encoding, val_states, val_distances, table, move_set, device)
        print(f"resumed {cfg.resume} at step {step_offset}, val optimal {best_rate:.3f}")

    append = resumed is not None and log_path.exists()
    start = time.perf_counter()
    with open(log_path, "a" if append else "w", newline="") as log_file:
        log = csv.writer(log_file)
        if not append:
            log.writerow(["step", "depth", "loss", "policy_loss", "value_loss", "label_acc",
                          "val_optimal_rate", "val_value_mae", "lr", "seconds"])
        model.train()
        for step in range(1, cfg.steps + 1):
            # Fresh data every step, straight from the vectorized simulator.
            if cfg.labels == "exact":
                depth = 0
                batch = exact_batch(cfg.batch_size, table, move_set, rng, exclude)
            else:
                ramp = step / cfg.curriculum_steps if cfg.curriculum_steps else 1.0
                depth = int(np.clip(round(1 + (max_depth - 1) * ramp), 1, max_depth))
                batch = scramble_batch(-(-cfg.batch_size // depth), depth, move_set, rng, exclude)

            x = torch.from_numpy(encode(batch.states, cfg.encoding)).to(device)
            policy_target = torch.from_numpy(batch.policy_target).to(device)
            value_target = torch.from_numpy(batch.value_target).to(device)

            logits, value = model(x)
            policy_loss = F.cross_entropy(logits, policy_target, label_smoothing=cfg.label_smoothing)
            value_loss = F.smooth_l1_loss(value, value_target)
            loss = policy_loss + cfg.value_weight * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % cfg.eval_every == 0 or step == cfg.steps:
                # Label accuracy is capped by ambiguous labels; the optimal-move
                # rate on held-out states is the metric that matters.
                label_acc = (logits.argmax(1) == policy_target.argmax(1)).float().mean().item()
                rate, mae = validate(model, cfg.encoding, val_states, val_distances, table, move_set, device)
                elapsed = time.perf_counter() - start
                log.writerow([step_offset + step, depth, loss.item(), policy_loss.item(), value_loss.item(), label_acc,
                              rate, mae, scheduler.get_last_lr()[0], elapsed])
                log_file.flush()
                print(f"step {step_offset + step:>6} | depth {depth:>2} | loss {loss.item():.4f} | label acc {label_acc:.3f} "
                      f"| val optimal {rate:.3f} | value MAE {mae:.2f} | {elapsed:.0f}s")
                if rate > best_rate:
                    best_rate = rate
                    checkpoint_config["step"] = step_offset + step
                    save_checkpoint(out, model, checkpoint_config)

    print(f"best validation optimal-move rate {best_rate:.3f}, saved to {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Build one flag per config field so the CLI and the dataclass never drift apart.
    for field in fields(TrainConfig):
        flag = "--" + field.name.replace("_", "-")
        if field.type == "bool":
            parser.add_argument(flag, action="store_true")
        elif field.name == "labels":
            parser.add_argument(flag, choices=("scramble", "exact"), default=field.default)
        elif field.name == "encoding":
            parser.add_argument(flag, choices=ENCODINGS, default=field.default)
        else:
            parser.add_argument(flag, type=type(field.default), default=field.default)
    train(TrainConfig(**vars(parser.parse_args(argv))))


if __name__ == "__main__":
    main()
