"""Step-based pipeline orchestration.

A deliberately small alternative to heavyweight DAG schedulers for the
common case: a named sequence of steps sharing a context, with skip
conditions, per-step error policy, timing, and observer hooks. Subclass
:class:`BasePipeline`, return your steps from ``define_steps``, call
``run()``.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

__all__ = [
    "StepResult",
    "StepDefinition",
    "PipelineContext",
    "PipelineResult",
    "BasePipeline",
]


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class StepResult:
    """Result of a single pipeline step execution."""

    name: str
    success: bool
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    stats: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    skipped: bool = False

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class StepDefinition:
    """Definition of a pipeline step.

    Attributes:
        name: Step name (key in result/context lookups).
        fn: Callable receiving the :class:`PipelineContext`.
        skip_condition: Optional predicate on the context; truthy = skip.
        description: Human-readable description.
        continue_on_failure: When True, a raised exception records a failed
            StepResult but the pipeline proceeds to the next step.
    """

    name: str
    fn: Callable
    skip_condition: Optional[Callable[["PipelineContext"], bool]] = None
    description: str = ""
    continue_on_failure: bool = False


@dataclass
class PipelineContext:
    """Shared context passed through pipeline steps.

    Steps communicate by reading/writing ``extra`` and by inspecting
    earlier results in ``step_results``.
    """

    name: str = ""
    config: Any = None
    dry_run: bool = False
    verbose: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, StepResult] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""

    success: bool
    step_results: List[StepResult] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    @property
    def steps_completed(self) -> List[str]:
        return [sr.name for sr in self.step_results if sr.success and not sr.skipped]

    @property
    def steps_failed(self) -> List[str]:
        return [sr.name for sr in self.step_results if not sr.success and not sr.skipped]

    @property
    def steps_skipped(self) -> List[str]:
        return [sr.name for sr in self.step_results if sr.skipped]

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


def _normalize_step(step) -> StepDefinition:
    """Convert a ``(name, fn)`` tuple to a StepDefinition."""
    if isinstance(step, StepDefinition):
        return step
    name, fn = step
    return StepDefinition(name=name, fn=fn)


class BasePipeline(ABC):
    """Abstract pipeline: subclasses define the steps.

    Usage::

        class NightlyPipeline(BasePipeline):
            def define_steps(self):
                return [
                    StepDefinition("pull_data", self.pull_data),
                    StepDefinition("compute_ratings", self.compute_ratings,
                                   skip_condition=lambda ctx: ctx.dry_run),
                    ("build_report", self.build_report),  # tuple also works
                ]

            def pull_data(self, context):
                context.extra["games"] = ...

        result = NightlyPipeline("nightly").run(dry_run=True)
    """

    def __init__(self, name: str = ""):
        self.name = name

    @abstractmethod
    def define_steps(self) -> List[Union[StepDefinition, tuple]]:
        """Return steps as StepDefinitions or ``(name, callable)`` tuples.

        Each callable receives the :class:`PipelineContext`.
        """
        ...

    def on_step_start(self, step: StepDefinition, index: int, total: int):
        """Hook called before a step runs. Override for custom output."""

    def on_step_complete(self, step: StepDefinition, result: StepResult, index: int, total: int):
        """Hook called after a step completes. Override for custom output."""

    def on_step_skip(self, step: StepDefinition, index: int, total: int):
        """Hook called when a step is skipped. Override for custom output."""

    def run(self, **kwargs) -> PipelineResult:
        """Execute the pipeline steps in order.

        Recognized kwargs: ``dry_run``, ``verbose``, ``config``. Everything
        else lands in ``context.extra``.
        """
        result = PipelineResult(success=True, started_at=_now())

        context = PipelineContext(
            name=self.name,
            dry_run=kwargs.pop("dry_run", False),
            verbose=kwargs.pop("verbose", False),
            config=kwargs.pop("config", None),
            extra=kwargs,
        )

        steps = [_normalize_step(s) for s in self.define_steps()]
        total = len(steps)

        for index, step in enumerate(steps):
            if step.skip_condition and step.skip_condition(context):
                step_result = StepResult(name=step.name, success=True, skipped=True)
                result.step_results.append(step_result)
                context.step_results[step.name] = step_result
                self.on_step_skip(step, index, total)
                continue

            step_result = StepResult(name=step.name, success=False, started_at=_now())
            context.step_results[step.name] = step_result
            self.on_step_start(step, index, total)

            try:
                logger.info(f"[{self.name}] Running step: {step.name}")
                step.fn(context)
                step_result.success = True
            except Exception as e:
                logger.error(f"[{self.name}] Step '{step.name}' failed: {e}")
                step_result.error = str(e)
                if not step.continue_on_failure:
                    step_result.completed_at = _now()
                    result.step_results.append(step_result)
                    self.on_step_complete(step, step_result, index, total)
                    result.success = False
                    result.error = str(e)
                    break

            step_result.completed_at = _now()
            result.step_results.append(step_result)
            self.on_step_complete(step, step_result, index, total)

        result.completed_at = _now()
        return result
