"""Run tracking. Local JSON is primary; MLflow is optional and additive.

PRD §4 specifies a sink that "dual-writes MLflow + local JSON". The ordering
matters and is deliberate here: local JSON is written first and unconditionally,
so a run survives an MLflow outage, a missing tracking URI, or a serverless
quota kill mid-loop (PRD §3.2 requires every long job to checkpoint). MLflow is
attempted afterwards and its failure is recorded, never raised.

**MLflow was not exercised in this project's runs.** No Databricks workspace was
used, so there is no tracking server behind this. The module is written and
unit-tested against the local path; the MLflow branch is inert unless a URI is
configured. Stated here rather than implied, per R28.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunSink:
    """Append-only local run log, one JSON file per run id."""

    artifacts: Path
    run_id: str
    experiment: str | None = None
    params: dict = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    mlflow_status: str = "not_attempted"

    @property
    def path(self) -> Path:
        return Path(self.artifacts) / "runs" / f"{self.run_id}.json"

    def log_params(self, **kw) -> None:
        self.params.update(kw)
        self._flush()

    def log_metrics(self, fold_idx: int, model: str, feature_view: str, metrics: dict) -> None:
        """One row per (fold, model, feature_view, metric). Matches the
        run_metrics schema in PRD §5.4."""
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and value == value:  # drop NaN
                self.rows.append({
                    "run_id": self.run_id, "fold_idx": fold_idx, "model": model,
                    "feature_view": feature_view,
                    "metric_name": name, "metric_value": float(value),
                })
        self._flush()

    def _flush(self) -> None:
        """Written after every call, so a kill mid-loop loses at most one fold."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({
                "run_id": self.run_id,
                "experiment": self.experiment,
                "params": self.params,
                "metrics": self.rows,
                "mlflow_status": self.mlflow_status,
            }, indent=1, default=str),
            encoding="utf-8",
        )

    # -- optional MLflow mirror -----------------------------------------
    def mirror_to_mlflow(self, tracking_uri: str | None = None) -> str:
        """Best effort. Returns a status string and never raises."""
        if not self.experiment:
            self.mlflow_status = "skipped: no experiment configured"
            self._flush()
            return self.mlflow_status
        try:
            import mlflow
        except ImportError:
            self.mlflow_status = "skipped: mlflow not installed"
            self._flush()
            return self.mlflow_status
        try:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.experiment)
            with mlflow.start_run(run_name=self.run_id):
                mlflow.log_params({k: str(v) for k, v in self.params.items()})
                for r in self.rows:
                    mlflow.log_metric(
                        f"{r['feature_view']}.{r['model']}.{r['metric_name']}",
                        r["metric_value"], step=int(r["fold_idx"]),
                    )
            self.mlflow_status = "ok"
        except Exception as e:  # noqa: BLE001 -- tracking must never fail a run
            self.mlflow_status = f"failed: {type(e).__name__}: {e}"
        self._flush()
        return self.mlflow_status


def load(artifacts: Path, run_id: str) -> dict:
    p = Path(artifacts) / "runs" / f"{run_id}.json"
    return json.loads(p.read_text(encoding="utf-8"))
