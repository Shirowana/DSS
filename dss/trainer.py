from __future__ import annotations

import torch
from tqdm.auto import tqdm
from transformers import Trainer
from transformers.trainer_utils import TrainOutput

from peft.tuners.dss import create_dss_optimizer


def _resolve_precision(args) -> tuple[torch.dtype, bool]:
    if args.fp16:
        return torch.float16, True
    if getattr(args, "bf16", False):
        return torch.bfloat16, True
    return torch.float32, False


def _get_refresh_target(model):
    if hasattr(model, "refresh_dss_layout"):
        return model
    if hasattr(model, "base_model") and hasattr(model.base_model, "refresh_dss_layout"):
        return model.base_model
    raise AttributeError("Could not find `refresh_dss_layout` on the PEFT model or its `base_model`.")


def _emit_debug(message: str, progress_bar=None) -> None:
    if progress_bar is not None:
        try:
            progress_bar.write(message)
        except Exception:
            pass
    print(message, flush=True)


def _collect_dss_health_metrics(refresh_target) -> dict[str, float]:
    if not hasattr(refresh_target, "active_dss_layers"):
        return {}
    metrics: dict[str, float] = {}
    wanted_groups = {"q_proj", "k_proj", "v_proj"}
    for _module_name, layer, adapter_name in refresh_target.active_dss_layers():
        last_health_stats = getattr(layer, "last_health_stats", None)
        if not last_health_stats:
            continue
        stats = last_health_stats.get(adapter_name)
        if not stats:
            continue
        group = str(stats.get("group", "unknown"))
        if group not in wanted_groups:
            continue
        prefix = f"dss/{group}"
        metrics[f"{prefix}/delta_base_ratio"] = float(stats["delta_base_ratio"])
        metrics[f"{prefix}/delta_abs_max"] = float(stats["delta_abs_max"])
        metrics[f"{prefix}/base_abs_max"] = float(stats["base_abs_max"])
        metrics[f"{prefix}/coeff_abs_max"] = float(stats["coeff_abs_max"])
        metrics[f"{prefix}/coeff_rms"] = float(stats["coeff_rms"])
        metrics[f"{prefix}/active_slots"] = float(stats["active_slots"])
    return metrics


class DSSTrainer(Trainer):
    def create_optimizer(self):
        if self.optimizer is None:
            self.optimizer = create_dss_optimizer(
                self.model,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer

    def train(self, *args, **kwargs):
        del args, kwargs

        train_dataloader = self.get_train_dataloader()
        if len(train_dataloader) == 0:
            raise ValueError("Training dataloader is empty.")

        updates_per_epoch = max(1, len(train_dataloader) // self.args.gradient_accumulation_steps)
        total_steps = self.args.max_steps if self.args.max_steps > 0 else max(1, updates_per_epoch * int(self.args.num_train_epochs))

        self.create_optimizer()
        self.create_scheduler(num_training_steps=total_steps, optimizer=self.optimizer)

        model = self.model
        model.train()
        refresh_target = _get_refresh_target(model)
        device = self.args.device
        torch_dtype, use_amp = _resolve_precision(self.args)
        scaler = torch.amp.GradScaler("cuda", enabled=(self.args.fp16 and torch.cuda.is_available()))

        global_step = 0
        total_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        progress_bar = tqdm(total=total_steps, dynamic_ncols=True, leave=True)

        for _epoch in range(int(self.args.num_train_epochs)):
            for micro_step, batch in enumerate(train_dataloader):
                batch = self._prepare_inputs(batch)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch_dtype,
                    enabled=(use_amp and device.type == "cuda"),
                ):
                    loss = model(**batch).loss
                    loss = loss / self.args.gradient_accumulation_steps

                if not torch.isfinite(loss):
                    _emit_debug(
                        f"[error] non-finite loss detected before backward: "
                        f"micro_step={micro_step + 1}, global_step={global_step}"
                    , progress_bar)
                    raise FloatingPointError("Encountered non-finite loss during DSS training.")

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                is_update_step = (micro_step + 1) % self.args.gradient_accumulation_steps == 0
                if not is_update_step:
                    continue

                if self.args.max_grad_norm and self.args.max_grad_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                if scaler.is_enabled():
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()

                global_step += 1
                total_loss += loss.item() * self.args.gradient_accumulation_steps
                refresh_target.refresh_dss_layout(
                    global_step=global_step,
                    total_steps=total_steps,
                    optimizer=self.optimizer,
                    grad_accumulation_steps=self.args.gradient_accumulation_steps,
                )

                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                else:
                    current_lr = self.args.learning_rate
                self.optimizer.zero_grad(set_to_none=True)

                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{loss.item() * self.args.gradient_accumulation_steps:.4f}", lr=f"{current_lr:.2e}")
                self.state.global_step = global_step
                self.state.max_steps = total_steps

                if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    logs = {
                        "loss": loss.item() * self.args.gradient_accumulation_steps,
                        "learning_rate": current_lr,
                    }
                    logs.update(_collect_dss_health_metrics(refresh_target))
                    self.log(logs)

                if self.args.save_strategy == "steps" and self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    checkpoint_dir = f"{self.args.output_dir}/checkpoint-{global_step}"
                    self.save_model(checkpoint_dir)
                    processing = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
                    if processing is not None:
                        processing.save_pretrained(checkpoint_dir)

                if global_step >= total_steps:
                    break
            if global_step >= total_steps:
                break

        progress_bar.close()

        self.state.global_step = global_step
        self.state.max_steps = total_steps
        self.state.epoch = float(self.args.num_train_epochs)
        self.save_model(self.args.output_dir)
        processing = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        if processing is not None:
            processing.save_pretrained(self.args.output_dir)

        mean_loss = total_loss / max(global_step, 1)
        return TrainOutput(global_step=global_step, training_loss=mean_loss, metrics={})
