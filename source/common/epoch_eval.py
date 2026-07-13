"""Fast-LeWAM 训练期间的进程内环境评测 callback。"""

import os
import random
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf

from source.common.eval import (
    evaluate_from_dataset_compat,
    fit_eval_processors,
    get_dataset,
    get_episodes_length,
    img_transform,
)
from source.policy.fast_lewam_eval import make_fast_lewam_policy


_ROOT = Path(__file__).resolve().parents[2]


def _load_eval_config(config_name, num_eval, solver_overrides, seed):
    """加载独立 eval YAML，并合并本地 launcher 与 CEM solver 默认配置。"""
    cfg = OmegaConf.load(_ROOT / "config" / "eval" / f"{config_name}.yaml")
    cfg.pop("defaults", None)
    launcher = OmegaConf.load(_ROOT / "config" / "eval" / "launcher" / "local.yaml")
    launcher.pop("defaults", None)
    solver = OmegaConf.load(_ROOT / "config" / "eval" / "solver" / "cem.yaml")
    cfg = OmegaConf.merge(launcher, cfg)
    cfg.seed = int(seed)
    cfg.solver = OmegaConf.merge(solver, solver_overrides or {})
    cfg.eval.num_eval = int(num_eval)
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    OmegaConf.resolve(cfg)
    return cfg


class FastLeWAMEpochEvaluator:
    """缓存 Cube eval 数据上下文，并用当前内存模型评测指定 Fast-LeWAM mode。"""

    def __init__(
        self,
        config_name="cube",
        num_eval=5,
        solver_overrides=None,
        seed=42,
        save_video=False,
    ):
        """记录轻量 epoch-eval 配置；数据集在首次实际评测时懒加载。"""
        self.cfg = _load_eval_config(config_name, num_eval, solver_overrides, seed)
        self.save_video = save_video
        self._context = None

    def _prepare_context(self):
        """固定评测起点并缓存数据集、processor 和图像 transform。"""
        if self._context is not None:
            return self._context
        cfg = self.cfg
        dataset = get_dataset(cfg, cfg.eval.dataset_name)
        col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
        episodes = np.unique(dataset.get_col_data(col))
        lengths = get_episodes_length(dataset, episodes)
        max_start = lengths - cfg.eval.goal_offset_steps - 1
        limits = dict(zip(episodes, max_start))
        valid = np.nonzero(
            dataset.get_col_data("step_idx")
            <= np.array([limits[value] for value in dataset.get_col_data(col)])
        )[0]
        if len(valid) < cfg.eval.num_eval:
            raise ValueError(
                f"epoch eval needs {cfg.eval.num_eval} starts, only {len(valid)} available"
            )
        selected = np.sort(
            np.random.default_rng(cfg.seed).choice(
                valid, size=cfg.eval.num_eval, replace=False
            )
        )
        rows = dataset.get_row_data(selected)
        self._context = {
            "dataset": dataset,
            "eval_episodes": rows[col],
            "eval_start_idx": rows["step_idx"],
            "process": fit_eval_processors(dataset, cfg.dataset.keys_to_cache),
            "transform": {
                "pixels": img_transform(cfg),
                "goal": img_transform(cfg),
            },
        }
        return self._context

    def __call__(self, *, model, mode, epoch, output_dir):
        """用给定 epoch 的当前模型运行一次环境评测并返回 metrics。"""
        cfg = self.cfg
        context = self._prepare_context()
        device = next(model.parameters()).device
        cfg.solver.device = str(device)
        mode_dir = Path(output_dir) / f"epoch_{epoch}" / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        video_dir = mode_dir / "videos"
        if self.save_video:
            video_dir.mkdir(parents=True, exist_ok=True)
        world = None
        try:
            world = swm.World(**cfg.world, image_shape=(224, 224))
            policy = make_fast_lewam_policy(
                model,
                solver_cfg=cfg.solver,
                plan_config=cfg.plan_config,
                process=context["process"],
                transform=context["transform"],
                device=str(device),
                mode=mode,
                seed=cfg.seed,
            )
            world.set_policy(policy)
            metrics = evaluate_from_dataset_compat(
                world=world,
                dataset=context["dataset"],
                eval_start_idx=context["eval_start_idx"],
                eval_episodes=context["eval_episodes"],
                cfg=cfg,
                video_path=video_dir,
                save_video=self.save_video,
            )
            with (mode_dir / "metrics.txt").open("a") as file:
                file.write(f"{metrics}\n")
            return metrics
        finally:
            env = getattr(world, "env", None) if world is not None else None
            if env is not None and hasattr(env, "close"):
                env.close()


class FastLeWAMEpochEvalCallback(Callback):
    """每隔若干 epoch 在 global rank 0 上依次运行 Stage A/B 环境评测。"""

    def __init__(
        self,
        config_name="cube",
        modes=("stage_a", "stage_b"),
        every_n_epochs=1,
        num_eval=5,
        solver=None,
        seed=42,
        save_video=False,
        evaluator=None,
    ):
        """构造 callback；默认 Stage B 使用配置传入的轻量 CEM 参数。"""
        super().__init__()
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be positive")
        if not modes or any(mode not in {"stage_a", "stage_b"} for mode in modes):
            raise ValueError("modes must contain stage_a and/or stage_b")
        self.modes = tuple(modes)
        self.every_n_epochs = every_n_epochs
        self.evaluator = evaluator or FastLeWAMEpochEvaluator(
            config_name=config_name,
            num_eval=num_eval,
            solver_overrides=OmegaConf.to_container(solver) if solver else None,
            seed=seed,
            save_video=save_video,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        """在到期 epoch 同步所有 rank，由 rank 0 评测并在完成后恢复训练状态。"""
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs:
            return
        trainer.strategy.barrier("fast-lewam-epoch-eval-start")
        if trainer.is_global_zero:
            model = pl_module.model
            output_dir = Path(trainer.default_root_dir) / "epoch_eval"
            was_training = model.training
            requires_grad = [parameter.requires_grad for parameter in model.parameters()]
            python_rng_state = random.getstate()
            numpy_rng_state = np.random.get_state()
            torch_rng_state = torch.get_rng_state()
            model_device = next(model.parameters()).device
            cuda_rng_state = (
                torch.cuda.get_rng_state(model_device)
                if model_device.type == "cuda"
                else None
            )
            try:
                for mode in self.modes:
                    try:
                        metrics = self.evaluator(
                            model=model,
                            mode=mode,
                            epoch=epoch,
                            output_dir=output_dir,
                        )
                        self._log_metrics(trainer, mode, metrics)
                    except Exception:
                        failure_dir = output_dir / f"epoch_{epoch}" / mode
                        failure_dir.mkdir(parents=True, exist_ok=True)
                        with (failure_dir / "failure.txt").open("a") as file:
                            file.write(traceback.format_exc())
                        print(f"Fast-LeWAM {mode} epoch eval failed; continuing.")
                        traceback.print_exc()
                        self._log_metrics(trainer, mode, {"failed": 1.0})
            finally:
                random.setstate(python_rng_state)
                np.random.set_state(numpy_rng_state)
                torch.set_rng_state(torch_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state(cuda_rng_state, model_device)
                model.train(was_training)
                for parameter, flag in zip(model.parameters(), requires_grad):
                    parameter.requires_grad_(flag)
        trainer.strategy.barrier("fast-lewam-epoch-eval-end")

    @staticmethod
    def _log_metrics(trainer, mode, metrics):
        """将标量评测指标写入已配置的 logger；复杂结构仍保存在 metrics.txt。"""
        if trainer.logger is None or not isinstance(metrics, dict):
            return
        scalar = {}
        for key, value in metrics.items():
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                value = value.detach().cpu().item()
            elif np.asarray(value).size != 1:
                continue
            scalar[f"epoch_eval/{mode}/{key}"] = float(value)
        if scalar:
            trainer.logger.log_metrics(scalar, step=trainer.global_step)


__all__ = ["FastLeWAMEpochEvalCallback", "FastLeWAMEpochEvaluator"]
