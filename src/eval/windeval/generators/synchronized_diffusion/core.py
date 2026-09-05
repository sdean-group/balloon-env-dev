"""Alternative overlap-synchronization rules for a frozen local wind denoiser.

All strategies use the same canonical chart atlas as CFGD. They differ only in how
overlapping local diffusion paths communicate inside one chart:

``sync_tweedies``
    Synchronize predicted clean fields at every solver evaluation.
``overlap_guided``
    Keep local paths separate and add a normalized gradient that reduces disagreement
    between overlapping predicted clean fields.
``consensus_equilibrium``
    Run a fixed number of denoiser/consensus/dual-correction rounds at every solver
    evaluation.
"""
from __future__ import annotations

from typing import Literal

import torch

try:
    from ..canonical_factor_graph.core import (
        CanonicalFactorGraphField,
        _coordinate_noise,
    )
except ImportError:  # standalone generator/test execution
    from canonical_factor_graph.core import (
        CanonicalFactorGraphField,
        _coordinate_noise,
    )

Strategy = Literal[
    "sync_tweedies",
    "overlap_guided",
    "consensus_equilibrium",
]


class SynchronizedChartField(CanonicalFactorGraphField):
    """Canonical atlas whose chart factors retain separate diffusion states."""

    def __init__(
        self,
        *args,
        strategy: Strategy = "sync_tweedies",
        guidance_strength: float = 0.15,
        consensus_rounds: int = 2,
        consensus_relaxation: float = 0.5,
        dual_scale: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if strategy not in {
            "sync_tweedies",
            "overlap_guided",
            "consensus_equilibrium",
        }:
            raise ValueError(f"unknown synchronization strategy: {strategy}")
        if guidance_strength < 0:
            raise ValueError("guidance_strength must be nonnegative")
        if consensus_rounds < 1:
            raise ValueError("consensus_rounds must be positive")
        if not 0 < consensus_relaxation <= 1:
            raise ValueError("consensus_relaxation must be in (0, 1]")
        if dual_scale < 0:
            raise ValueError("dual_scale must be nonnegative")
        self.strategy = strategy
        self.guidance_strength = float(guidance_strength)
        self.consensus_rounds = int(consensus_rounds)
        self.consensus_relaxation = float(consensus_relaxation)
        self.dual_scale = float(dual_scale)
        self.overlap_objective_evaluations = 0
        self.consensus_iterations = 0

    def _initial_factor_states(
        self,
        chart_noise: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        return torch.stack(
            [
                chart_noise[
                    :,
                    dt:dt + self.tau,
                    dy:dy + cfg.window_size,
                    dx:dx + cfg.window_size,
                ].permute(1, 0, 2, 3)
                for dt, dy, dx in self._factor_offsets
            ],
            dim=0,
        )

    def _predict_clean(
        self,
        states: torch.Tensor,
        sigma: torch.Tensor,
        chart_origin: tuple[int, int, int],
        *,
        track_grad: bool = False,
    ) -> torch.Tensor:
        predictions = []
        cfg = self.config
        for start in range(0, len(self._factor_offsets), cfg.window_batch_size):
            offsets = self._factor_offsets[start:start + cfg.window_batch_size]
            batch = states[start:start + len(offsets)]
            conditions, time_features = [], []
            for offset in offsets:
                cond, tfeat = self._factor_condition(chart_origin, offset)
                conditions.append(cond)
                time_features.append(tfeat)
            cond_batch = torch.cat(conditions, dim=0) if conditions[0] is not None else None
            time_batch = (
                torch.cat(time_features, dim=0)
                if time_features[0] is not None
                else None
            )
            context = torch.enable_grad() if track_grad else torch.no_grad()
            with context:
                predictions.append(
                    self.sampler.model(
                        batch,
                        sigma.expand(len(offsets)),
                        cond=cond_batch,
                        tfeat=time_batch,
                    )
                )
            self.model_batch_calls += 1
            self.model_window_evaluations += len(offsets)
        return torch.cat(predictions, dim=0)

    def _assemble(self, local_values: torch.Tensor) -> torch.Tensor:
        """Weighted local windows -> one chart tensor in (channel, time, y, x)."""
        cfg = self.config
        chart = torch.zeros(
            (self.C, cfg.support_time, cfg.support_size, cfg.support_size),
            device=self.device,
            dtype=self.dtype,
        )
        normalizer = torch.zeros(
            (1, cfg.support_time, cfg.support_size, cfg.support_size),
            device=self.device,
            dtype=self.dtype,
        )
        weight = self._window_weight.unsqueeze(0)
        for value, (dt, dy, dx) in zip(local_values, self._factor_offsets):
            placed = value.permute(1, 0, 2, 3)
            chart[
                :,
                dt:dt + self.tau,
                dy:dy + cfg.window_size,
                dx:dx + cfg.window_size,
            ] += weight * placed
            normalizer[
                :,
                dt:dt + self.tau,
                dy:dy + cfg.window_size,
                dx:dx + cfg.window_size,
            ] += weight
        if torch.any(normalizer <= 0):
            raise RuntimeError("factor windows did not cover the complete chart")
        return chart / normalizer

    def _extract(self, chart: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        return torch.stack(
            [
                chart[
                    :,
                    dt:dt + self.tau,
                    dy:dy + cfg.window_size,
                    dx:dx + cfg.window_size,
                ].permute(1, 0, 2, 3)
                for dt, dy, dx in self._factor_offsets
            ],
            dim=0,
        )

    def _sync_tweedies_direction(
        self,
        states: torch.Tensor,
        sigma: torch.Tensor,
        chart_origin: tuple[int, int, int],
    ) -> torch.Tensor:
        predicted_clean = self._predict_clean(states, sigma, chart_origin)
        consensus_clean = self._extract(self._assemble(predicted_clean))
        return (states - consensus_clean) / sigma

    def _overlap_guided_direction(
        self,
        states: torch.Tensor,
        sigma: torch.Tensor,
        chart_origin: tuple[int, int, int],
    ) -> torch.Tensor:
        """SyncDiffusion-style gradient guidance using overlap disagreement."""
        predicted_clean = self._predict_clean(states, sigma, chart_origin)
        with torch.enable_grad():
            proxy = predicted_clean.detach().requires_grad_(True)
            proxy_consensus = self._extract(self._assemble(proxy))
            disagreement = proxy - proxy_consensus
            overlap_loss = 0.5 * disagreement.square().mean()
            prediction_grad = torch.autograd.grad(overlap_loss, proxy)[0].detach()

        input_grads = []
        cfg = self.config
        for start in range(0, len(self._factor_offsets), cfg.window_batch_size):
            count = min(cfg.window_batch_size, len(self._factor_offsets) - start)
            state_batch = states[start:start + count].detach().requires_grad_(True)
            offsets = self._factor_offsets[start:start + count]
            conditions, time_features = [], []
            for offset in offsets:
                cond, tfeat = self._factor_condition(chart_origin, offset)
                conditions.append(cond)
                time_features.append(tfeat)
            cond_batch = torch.cat(conditions, dim=0) if conditions[0] is not None else None
            time_batch = (
                torch.cat(time_features, dim=0)
                if time_features[0] is not None
                else None
            )
            with torch.enable_grad():
                output = self.sampler.model(
                    state_batch,
                    sigma.expand(count),
                    cond=cond_batch,
                    tfeat=time_batch,
                )
                gradient = torch.autograd.grad(
                    output,
                    state_batch,
                    grad_outputs=prediction_grad[start:start + count],
                )[0]
            input_grads.append(gradient.detach())
            self.model_batch_calls += 1
            self.model_window_evaluations += count

        base_direction = (states - predicted_clean) / sigma
        loss_gradient = torch.cat(input_grads, dim=0)
        reduce_dims = tuple(range(1, loss_gradient.ndim))
        gradient_rms = loss_gradient.square().mean(dim=reduce_dims, keepdim=True).sqrt()
        direction_rms = base_direction.square().mean(
            dim=reduce_dims, keepdim=True
        ).sqrt()
        gradient_scale = torch.where(
            gradient_rms > 1e-10,
            direction_rms / gradient_rms.clamp_min(1e-12),
            torch.zeros_like(gradient_rms),
        )
        normalized_gradient = loss_gradient * gradient_scale
        self.overlap_objective_evaluations += 1
        return base_direction + self.guidance_strength * normalized_gradient

    def _consensus_equilibrium_direction(
        self,
        states: torch.Tensor,
        sigma: torch.Tensor,
        chart_origin: tuple[int, int, int],
    ) -> torch.Tensor:
        dual = torch.zeros_like(states)
        consensus_clean = None
        for _ in range(self.consensus_rounds):
            effective_state = states - self.dual_scale * dual
            predicted_clean = self._predict_clean(
                effective_state,
                sigma,
                chart_origin,
            )
            consensus_clean = self._extract(
                self._assemble(predicted_clean + dual)
            )
            residual = predicted_clean - consensus_clean
            dual = (
                dual + self.consensus_relaxation * residual
            ).detach()
            self.consensus_iterations += 1
        assert consensus_clean is not None
        return (states - consensus_clean) / sigma

    def _strategy_direction(
        self,
        states: torch.Tensor,
        sigma: torch.Tensor,
        chart_origin: tuple[int, int, int],
    ) -> torch.Tensor:
        if self.strategy == "sync_tweedies":
            return self._sync_tweedies_direction(states, sigma, chart_origin)
        if self.strategy == "overlap_guided":
            return self._overlap_guided_direction(states, sigma, chart_origin)
        return self._consensus_equilibrium_direction(states, sigma, chart_origin)

    def _generate_chart(self, key: tuple[int, int, int]) -> torch.Tensor:
        cfg = self.config
        origin = self._chart_origin(key)
        noise = _coordinate_noise(
            self.C,
            cfg.support_time,
            cfg.support_size,
            cfg.support_size,
            t0=origin[0],
            y0=origin[1],
            x0=origin[2],
            seed=self.seed,
            device=self.device,
            dtype=self.dtype,
        )
        states = self._initial_factor_states(noise)
        schedule = self.sampler.sigma_schedule(
            device=self.device,
            dtype=self.dtype,
        )
        states = states * schedule[0]
        for step in range(self.sampler.num_steps):
            sigma, sigma_next = schedule[step], schedule[step + 1]
            direction = self._strategy_direction(states, sigma, origin)
            proposal = states + (sigma_next - sigma) * direction
            if step + 1 < self.sampler.num_steps:
                corrected = self._strategy_direction(
                    proposal.detach(),
                    sigma_next,
                    origin,
                )
                states = (
                    states
                    + 0.5
                    * (sigma_next - sigma)
                    * (direction + corrected)
                ).detach()
            else:
                states = proposal.detach()
        self.charts_generated += 1
        return self._assemble(states)
