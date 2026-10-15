"""Freeze the R4 flat checkpoint, record it and resume the terrain curriculum."""
import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

base = Path(__file__).resolve().parents[2]
validation = base / "validation"
run_root = base / "logs/rsl_rl/smalldog_v67_refined_gait_moe_cts"
source_run = run_root / "2026-10-14_17-44-07_v67_r4_pitch_refinement_plus10000"
state_path = validation / "r5_terrain_recovery_state.json"
state = {"pipeline_pid": os.getpid(),
         "expected_last_iteration": 12796, "num_envs": 4096,
         "task": "RobotLab-SmallDog-V67-RoughRefinedGait-v0",
         "initial_terrain_level": 0, "terrain_curriculum": True}
environment = os.environ.copy()
environment.update(
    VK_ICD_FILENAMES="/root/autodl-tmp/vulkan/nvidia_egl_icd.json",
    ACCEPT_EULA="Y", OMNI_KIT_ACCEPT_EULA="YES", PYTHONUNBUFFERED="1",
)


def save_state(**changes):
    state.update(changes)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=2))


def stop_owned(pid, expected, first_signal):
    proc = Path(f"/proc/{pid}")
    if not proc.exists():
        return
    command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    assert expected in command and (proc / "cwd").resolve() == base, command
    os.kill(pid, first_signal)
    for _ in range(30):
        if process_exited(proc):
            return
        time.sleep(1)
    os.kill(pid, signal.SIGTERM)
    for _ in range(15):
        if process_exited(proc):
            return
        time.sleep(1)
    raise RuntimeError(f"Owned process {pid} did not exit")


def process_exited(proc):
    try:
        return ") Z " in (proc / "stat").read_text()
    except FileNotFoundError:
        return True


def run_step(phase, arguments, report=None):
    log = validation / f"r5_terrain_{phase}.log"
    with log.open("w") as stdout:
        child = subprocess.Popen(["./python.sh", *arguments, "--headless", "--device", "cuda:0"],
                                 cwd=base, env=environment, stdout=stdout,
                                 stderr=subprocess.STDOUT)
        save_state(status="running", phase=phase, child_pid=child.pid, log=str(log))
        return_code = child.wait()
    assert return_code == 0, f"{phase} failed; see {log}"
    if report is not None:
        result = json.loads((base / report).read_text())
        assert result.get("pass", True), f"{phase} check failed"


try:
    old_path = validation / "r4_quality_10k_state.json"
    old = json.loads(old_path.read_text())
    assert old["phase"] == "training" and old["status"] == "running"
    tail_path = validation / "r4_quality_10k_training.log"
    with tail_path.open("rb") as log:
        log.seek(max(0, tail_path.stat().st_size - 65536))
        iterations = re.findall(r"Learning iteration\s+(\d+)/(\d+)", log.read().decode())
    last_logged = int(iterations[-1][0])
    candidates = sorted((p for p in source_run.glob("model_*.pt")
                         if int(p.stem[6:]) < last_logged),
                        key=lambda p: int(p.stem[6:]))
    source = candidates[-1]
    starting_iteration = int(source.stem[6:])
    payload = torch.load(source, map_location="cpu", weights_only=False)
    assert payload["iter"] == starting_iteration and "optimizer_state_dict" in payload
    del payload
    save_state(status="running", phase="stop_flat_training",
               source_checkpoint=str(source), starting_iteration=starting_iteration,
               additional_iterations=12797-starting_iteration,
               flat_last_logged_iteration=last_logged,
               unsaved_iterations_not_transferred=max(0, last_logged - starting_iteration))
    stop_owned(old["pipeline_pid"], "scripts/smalldog/run_quality_r4_10k.py", signal.SIGTERM)
    stop_owned(old["child_pid"], "scripts/rsl_rl/train.py", signal.SIGINT)
    old.update(status="superseded", phase="replaced_by_terrain_training",
               preserved_checkpoint=str(source),
               recovery_state=str(state_path),
               updated_at_utc=datetime.now(timezone.utc).isoformat())
    old_path.write_text(json.dumps(old, indent=2))
    run_step("recording", [
        "scripts/smalldog/record_gait_motion.py", "--refined", "--checkpoint", str(source),
        "--command-vx", "1.0", "--output", "validation/terrain_transition_recording.json",
    ], "validation/terrain_transition_recording.json")
    run_step("import_check", [
        "scripts/smalldog/check_refinement_import.py", "--rough", "--checkpoint", str(source),
        "--output", "validation/r5_terrain_import.json",
    ], "validation/r5_terrain_import.json")
    previous_runs = set(run_root.glob("*"))
    run_step("training", [
        "scripts/rsl_rl/train.py", "--task", state["task"], "--num_envs", "4096",
        "--max_iterations", str(state["additional_iterations"]), "--logger", "tensorboard",
        "--run_name", "v67_r5_terrain_recovery", "--resume",
        "--load_run", source_run.name, "--checkpoint", source.name,
    ])
    created = {p for p in run_root.glob("*") if p.is_dir()} - previous_runs
    assert len(created) == 1
    run_dir = created.pop()
    final = run_dir / "model_12796.pt"
    assert final.is_file(), "Terrain training final checkpoint is missing"
    save_state(run_dir=str(run_dir), final_checkpoint=str(final))
    for speed, label in ((0.3, "0p3"), (0.6, "0p6"), (1.0, "1p0")):
        output = f"validation/r5_terrain_final_{label}.json"
        run_step(f"evaluation_{label}", [
            "scripts/smalldog/evaluate_gait.py", "--refined", "--rough",
            "--max-terrain-level", "3", "--num-envs", "80", "--checkpoint", str(final),
            "--command-vx", str(speed), "--output", output,
        ], output)
    save_state(status="complete", phase="terrain_training_and_evaluation_complete", child_pid=None)
except Exception as error:
    save_state(status="failed", error=f"{type(error).__name__}: {error}", child_pid=None)
    raise
