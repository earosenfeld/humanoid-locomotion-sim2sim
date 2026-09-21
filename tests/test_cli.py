"""End-to-end CLI on the synthetic run: the same commands CI and the README use."""

import json

from humanoid_loco.cli import main


def test_sim2sim_sweep_report_prepare(run_dir, menagerie, capsys):
    main(["sim2sim", "--run", str(run_dir), "--seeds", "2", "--menagerie", str(menagerie)])
    report = json.loads((run_dir / "sim2sim-walk.json").read_text())
    assert report["summary"]["n_runs"] == 2 and "fall_rate" in report["summary"]

    main(
        [
            "sweep",
            "--run",
            str(run_dir),
            "--seeds",
            "1",
            "--mass",
            "1.0",
            "--friction",
            "0.5",
            "1.0",
            "--delay",
            "0",
            "2",
            "--push",
            "0.5",
            "--menagerie",
            str(menagerie),
        ]
    )
    grid = json.loads((run_dir / "sweep-walk.json").read_text())["grid"]
    assert len(grid) == 4 and {"mass_scale", "action_delay", "push_scale"} <= set(grid[0])

    main(["report", str(run_dir / "sim2sim-walk.json"), str(run_dir / "sweep-walk.json")])
    out = capsys.readouterr().out
    assert "| schedule |" in out and "| action_delay |" in out or "action_delay" in out

    main(["prepare-cpp", "--run", str(run_dir), "--menagerie", str(menagerie)])
    assert (run_dir / "scene.mjb").stat().st_size > 1e6
    assert json.loads((run_dir / "schedule-walk.json").read_text())["name"] == "walk"
