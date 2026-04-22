"""
Measurement frequency calculator for training loop.

Each measurement can be controlled by a fixed step interval (every N steps),
or fall back to adaptive rules from the original codebase.
"""

from typing import Dict, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class MeasurementContext:
    """Container for variables needed by frequency calculation rules."""
    step_number: int
    batch_size: int
    epoch: int = 0
    device: str = "cpu"
    lr: float = 0.01
    steps_since_restart: int = 0


class FrequencyCalculator:
    """
    Frequency calculator for training measurements.

    For each measurement type, you can either:
      - set a fixed interval via set_interval(name, N)  →  measure every N steps
      - or leave it to the default adaptive rule
    """

    def __init__(self):
        self.rules: Dict[str, Callable[[MeasurementContext], bool]] = {}
        self._intervals: Dict[str, Optional[int]] = {}
        # When set, intervals are halved for steps in [track_from, track_until].
        self._window_halve: bool = False
        self._window_from: Optional[int] = None
        self._window_until: Optional[int] = None
        self._setup_default_rules()

    def set_interval(self, measurement_type: str, every_n_steps: Optional[int]):
        """
        Set a fixed measurement interval. Pass None to revert to adaptive rule.
        """
        self._intervals[measurement_type] = every_n_steps

    def set_window_halve(self, track_from: Optional[int], track_until: Optional[int], enabled: bool):
        """Halve every fixed interval for steps in [track_from, track_until] when enabled."""
        self._window_halve = bool(enabled) and track_from is not None and track_until is not None
        self._window_from = track_from
        self._window_until = track_until

    def should_measure(self, measurement_type: str, ctx: MeasurementContext) -> bool:
        """Check if a measurement should be performed at this step."""
        # Fixed interval takes priority
        interval = self._intervals.get(measurement_type)
        if interval is not None:
            effective = interval
            if (self._window_halve
                and self._window_from <= ctx.step_number <= self._window_until):
                effective = max(1, interval // 2)
            return ctx.step_number % effective == 0

        # Fall back to adaptive rule
        if measurement_type not in self.rules:
            raise ValueError(f"Unknown measurement type: {measurement_type}")
        return self.rules[measurement_type](ctx)

    def _setup_default_rules(self):
        """Adaptive rules (used when no fixed interval is set)."""

        def full_batch_lambda_max_rule(ctx: MeasurementContext) -> bool:
            freq = 256
            if ctx.step_number > 10_000:
                freq *= 2
            if ctx.step_number > 30_000:
                freq *= 2
            return ctx.step_number % freq == 0

        def batch_sharpness_rule(ctx: MeasurementContext) -> bool:
            freq = 128
            if ctx.step_number > 10_000:
                freq *= 2
            if ctx.step_number > 50_000:
                freq *= 2
            if ctx.step_number > 100_000:
                freq *= 2
            return ctx.step_number % freq == 0

        def step_sharpness_rule(ctx: MeasurementContext) -> bool:
            return ctx.step_number % 16 == 0

        def checkpoint_rule(ctx: MeasurementContext) -> bool:
            return ctx.step_number % 32 == 0

        def full_loss_rule(ctx: MeasurementContext) -> bool:
            freq = 32
            if ctx.step_number > 10_000:
                freq *= 2
            if ctx.step_number > 50_000:
                freq *= 2
            return ctx.step_number % max(1, freq) == 0

        def full_loss_warmup_rule(ctx: MeasurementContext) -> bool:
            return ctx.steps_since_restart < 128

        def gen_batch_sharpness_outside_rule(ctx: MeasurementContext) -> bool:
            freq = 128
            if ctx.step_number > 10_000:
                freq *= 2
            if ctx.step_number > 50_000:
                freq *= 2
            if ctx.step_number > 100_000:
                freq *= 2
            return ctx.step_number % freq == 0

        self.rules.update({
            'full_batch_lambda_max': full_batch_lambda_max_rule,
            'batch_sharpness': batch_sharpness_rule,
            'gen_batch_sharpness_outside': gen_batch_sharpness_outside_rule,
            'step_sharpness': step_sharpness_rule,
            'checkpoint': checkpoint_rule,
            'full_loss': full_loss_rule,
            'full_loss_warmup': full_loss_warmup_rule,
        })


# Global instance
frequency_calculator = FrequencyCalculator()
