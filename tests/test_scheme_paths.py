"""Environment overrides for the D3 sensitivity schemes: a config file, a scenario catalog, and a results folder."""

from pathlib import Path

import yaml

from src.sim_module.config import DEFAULT_CONFIG, load_config, results_dir, scenarios_path

ROOT = Path(__file__).resolve().parent.parent


def test_defaults_when_no_override_is_set(monkeypatch):
    for name in ("PHASE2_CONFIG", "PHASE2_SCENARIOS", "PHASE2_RESULTS"):
        monkeypatch.delenv(name, raising=False)
    assert load_config().harm.weights == load_config(DEFAULT_CONFIG).harm.weights
    assert scenarios_path() == ROOT / "phase2" / "scenarios.yaml"
    assert results_dir() == ROOT / "phase2" / "results"


def test_config_override_changes_the_harm_weights(monkeypatch, tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    raw["harm"]["weights"], raw["harm"]["fn_factor"] = [1.0, 1.0, 1.0], 1.0
    alt = tmp_path / "flat.yaml"
    alt.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setenv("PHASE2_CONFIG", str(alt))
    cfg = load_config()
    assert cfg.harm.weights == (1.0, 1.0, 1.0) and cfg.harm.fn_factor == 1.0
    default_factor = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))["harm"]["fn_factor"]
    assert load_config(DEFAULT_CONFIG).harm.fn_factor == default_factor != 1.0          # an explicit path still wins over the environment


def test_catalog_and_results_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("PHASE2_SCENARIOS", str(tmp_path / "s.yaml"))
    monkeypatch.setenv("PHASE2_RESULTS", str(tmp_path / "out" / "run"))
    assert scenarios_path() == tmp_path / "s.yaml"
    made = results_dir()
    assert made == tmp_path / "out" / "run" and made.is_dir()          # created on demand, with its parents


def test_report_path_follows_the_results_override(monkeypatch, tmp_path):
    from src.sim_module.config import report_path
    monkeypatch.delenv("PHASE2_RESULTS", raising=False)
    assert report_path("X.md") == ROOT / "phase2" / "X.md"
    monkeypatch.setenv("PHASE2_RESULTS", str(tmp_path / "scheme"))
    assert report_path("X.md") == tmp_path / "scheme" / "X.md"


def test_false_alarm_budget_comes_from_the_config_unless_overridden(monkeypatch):
    from src.sim_module.config import false_alarm_budget
    monkeypatch.delenv("M5_BUDGET", raising=False)
    monkeypatch.delenv("PHASE2_CONFIG", raising=False)
    assert false_alarm_budget() == load_config().budget.false_alarms_per_1000_windows > 0
    monkeypatch.setenv("M5_BUDGET", "150")
    assert false_alarm_budget() == 150.0
