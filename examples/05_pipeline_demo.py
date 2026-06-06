"""Pipeline orchestration demo: skip conditions, failure policy, parallelism.

A toy nightly-refresh pipeline showing the orchestration vocabulary:
- a step skipped by condition (dry-run guard)
- a step that fails but lets the pipeline continue
- a parallel fan-out step (three independent computations)
- step results inspected by later steps

Run:
    python examples/05_pipeline_demo.py
"""

import time

from pmt.pipeline import BasePipeline, ParallelStepGroup, StepDefinition


class NightlyRefresh(BasePipeline):
    def define_steps(self):
        return [
            StepDefinition("pull_games", self.pull_games,
                           description="Fetch yesterday's results"),
            StepDefinition("write_to_store", self.write_to_store,
                           skip_condition=lambda ctx: ctx.dry_run,
                           description="Persist (skipped on dry runs)"),
            StepDefinition("optional_enrichment", self.optional_enrichment,
                           continue_on_failure=True,
                           description="Nice-to-have; failure shouldn't stop the run"),
            StepDefinition("compute_features", self.compute_features,
                           description="Fan out independent computations"),
            StepDefinition("summarize", self.summarize),
        ]

    def pull_games(self, ctx):
        time.sleep(0.05)  # pretend I/O
        ctx.extra["games"] = [{"home": "A", "away": "B"}, {"home": "C", "away": "D"}]
        ctx.step_results["pull_games"].stats["n_games"] = 2

    def write_to_store(self, ctx):
        raise AssertionError("never reached on a dry run")

    def optional_enrichment(self, ctx):
        raise ConnectionError("third-party enrichment API timed out")

    def compute_features(self, ctx):
        def ratings():
            time.sleep(0.1)
            return {"A": 1520, "B": 1480}

        def schedule_strength():
            time.sleep(0.1)
            return {"A": 0.52, "B": 0.48}

        def rest_days():
            time.sleep(0.1)
            return {"A": 2, "B": 1}

        group = ParallelStepGroup(tasks=[
            ("ratings", ratings, {}),
            ("schedule_strength", schedule_strength, {}),
            ("rest_days", rest_days, {}),
        ], max_workers=3)
        started = time.time()
        results = group.execute()
        ctx.extra["features"] = {k: r.result for k, r in results.items()}
        ctx.step_results["compute_features"].stats["wall_seconds"] = round(
            time.time() - started, 3
        )

    def summarize(self, ctx):
        print(f"  games pulled:   {len(ctx.extra['games'])}")
        print(f"  features:       {sorted(ctx.extra['features'])}")
        print(f"  parallel wall:  {ctx.step_results['compute_features'].stats['wall_seconds']}s"
              f"  (3 x 0.1s tasks)")

    def on_step_complete(self, step, result, index, total):
        mark = "ok " if result.success else "FAIL"
        print(f"[{index + 1}/{total}] {mark} {step.name}"
              + (f"  ({result.error})" if result.error else ""))

    def on_step_skip(self, step, index, total):
        print(f"[{index + 1}/{total}] skip {step.name}  ({step.description})")


result = NightlyRefresh("nightly-refresh").run(dry_run=True)

print(f"\npipeline success: {result.success}")
print(f"completed: {result.steps_completed}")
print(f"failed:    {result.steps_failed}  (continue_on_failure)")
print(f"skipped:   {result.steps_skipped}")
print(f"duration:  {result.duration_seconds:.2f}s")
