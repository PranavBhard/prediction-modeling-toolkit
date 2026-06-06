import threading
import time

import pytest

from pmt.pipeline import (
    BasePipeline,
    ParallelItemProcessor,
    ParallelStepGroup,
    ProgressTracker,
    StepDefinition,
)


class RecordingPipeline(BasePipeline):
    """Pipeline whose steps append to a log so order/effects are observable."""

    def __init__(self, steps):
        super().__init__(name="test")
        self._steps = steps
        self.hook_log = []

    def define_steps(self):
        return self._steps

    def on_step_start(self, step, index, total):
        self.hook_log.append(("start", step.name))

    def on_step_complete(self, step, result, index, total):
        self.hook_log.append(("complete", step.name, result.success))

    def on_step_skip(self, step, index, total):
        self.hook_log.append(("skip", step.name))


def test_steps_run_in_order_and_share_context():
    def a(ctx):
        ctx.extra["chain"] = ["a"]

    def b(ctx):
        ctx.extra["chain"].append("b")

    p = RecordingPipeline([("a", a), ("b", b)])
    result = p.run()
    assert result.success
    assert result.steps_completed == ["a", "b"]
    assert p.hook_log == [
        ("start", "a"), ("complete", "a", True),
        ("start", "b"), ("complete", "b", True),
    ]


def test_kwargs_land_in_context_extra():
    seen = {}

    def step(ctx):
        seen["dry_run"] = ctx.dry_run
        seen["season"] = ctx.extra["season"]

    RecordingPipeline([("s", step)]).run(dry_run=True, season="2024")
    assert seen == {"dry_run": True, "season": "2024"}


def test_skip_condition():
    ran = []
    steps = [
        StepDefinition("always", lambda ctx: ran.append("always")),
        StepDefinition("skipped", lambda ctx: ran.append("skipped"),
                       skip_condition=lambda ctx: ctx.dry_run),
    ]
    result = RecordingPipeline(steps).run(dry_run=True)
    assert ran == ["always"]
    assert result.steps_skipped == ["skipped"]
    assert result.success


def test_failure_stops_pipeline():
    def boom(ctx):
        raise RuntimeError("kaput")

    ran = []
    steps = [
        StepDefinition("boom", boom),
        StepDefinition("after", lambda ctx: ran.append("after")),
    ]
    result = RecordingPipeline(steps).run()
    assert not result.success
    assert result.error == "kaput"
    assert result.steps_failed == ["boom"]
    assert ran == []  # never reached


def test_continue_on_failure_proceeds():
    def boom(ctx):
        raise RuntimeError("kaput")

    ran = []
    steps = [
        StepDefinition("boom", boom, continue_on_failure=True),
        StepDefinition("after", lambda ctx: ran.append("after")),
    ]
    result = RecordingPipeline(steps).run()
    assert result.success  # pipeline-level success preserved
    assert result.steps_failed == ["boom"]
    assert ran == ["after"]


def test_step_results_visible_to_later_steps():
    def first(ctx):
        ctx.step_results["first"].stats["n"] = 42

    seen = {}

    def second(ctx):
        seen["n"] = ctx.step_results["first"].stats["n"]

    RecordingPipeline([("first", first), ("second", second)]).run()
    assert seen["n"] == 42


def test_durations_populated():
    result = RecordingPipeline([("s", lambda ctx: None)]).run()
    assert result.duration_seconds is not None
    assert result.step_results[0].duration_seconds is not None


# --- parallel -----------------------------------------------------------------


def test_parallel_step_group_results():
    def ok():
        return "fine"

    def bad():
        raise ValueError("nope")

    errors = []
    group = ParallelStepGroup(
        tasks=[("ok", ok, {}), ("bad", bad, {})],
        on_error=lambda name, tr: errors.append(name),
    )
    results = group.execute()
    assert results["ok"].success and results["ok"].result == "fine"
    assert not results["bad"].success and "nope" in results["bad"].error
    assert errors == ["bad"]


def test_parallel_step_group_runs_concurrently():
    barrier = threading.Barrier(3, timeout=5)

    def wait():
        barrier.wait()  # deadlocks unless all 3 run simultaneously
        return True

    results = ParallelStepGroup(
        tasks=[(f"t{i}", wait, {}) for i in range(3)], max_workers=3
    ).execute()
    assert all(r.success for r in results.values())


def test_parallel_item_processor():
    progress = []
    proc = ParallelItemProcessor(
        items=[1, 2, 3, 4],
        process_fn=lambda x: 1 / (x - 3),  # x=3 raises ZeroDivisionError
        max_workers=2,
        progress_callback=lambda done, total, item: progress.append((done, total)),
    )
    results = proc.execute()
    assert len(results) == 4
    by_item = {item: (res, exc) for item, res, exc in results}
    assert by_item[1][0] == pytest.approx(-0.5)
    assert isinstance(by_item[3][1], ZeroDivisionError)
    assert progress[-1] == (4, 4)


def test_chunked_parallel_processor():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"x": range(10)})
    proc_calls = []

    def double(chunk):
        proc_calls.append(len(chunk))
        return chunk.assign(x=chunk.x * 2)

    from pmt.pipeline import ChunkedParallelProcessor

    out = ChunkedParallelProcessor(df, double, chunk_size=3, max_workers=2).execute()
    assert sorted(proc_calls) == [1, 3, 3, 3]
    assert sorted(out.x.tolist()) == [v * 2 for v in range(10)]


def test_progress_tracker():
    t = ProgressTracker(total=10)
    t.update(completed=3)
    t.update(completed=2, failed=1)
    assert t.completed == 5
    assert t.failed == 1
    assert t.percent == pytest.approx(50.0)


def test_progress_tracker_background_render():
    renders = []
    t = ProgressTracker(total=2, render_fn=lambda tr: renders.append(tr.completed),
                        render_interval=0.01)
    t.start_background_render()
    t.update(completed=2)
    time.sleep(0.05)
    t.stop_background_render()
    assert renders  # rendered at least once
    assert renders[-1] == 2  # final render after stop
