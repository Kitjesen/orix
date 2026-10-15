"""Run the authorized 10,000-update terrain continuation and full-range evaluations."""
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

base = Path(__file__).resolve().parents[2]
validation = base / "validation"
state_path = validation / "r6_training_state.json"
source = base / (
    "logs/rsl_rl/smalldog_v67_refined_gait_moe_cts/"
    "2026-10-15_13-05-49_v67_r5_sampling_cache/model_12796.pt"
)
state = {"pipeline_pid": os.getpid(), "task": "RobotLab-SmallDog-V67-RoughR6-v0",
         "source_checkpoint": str(source), "source_iteration": 12796,
         "starting_iteration": 12797, "additional_iterations": 10000,
         "expected_last_iteration": 22796, "num_envs": 4096,
         "initial_terrain_level_range": [0, 10], "terrain_curriculum": True,
         "terrain_initialization": "uniform full range; legacy checkpoint has no per-env terrain state",
         "initial_speed_ceiling_m_s": 1.0, "maximum_speed_ceiling_m_s": 1.5,
         "terrain_state_saved_in_future_checkpoints": True}
environment = os.environ.copy()
environment.update(VK_ICD_FILENAMES="/root/autodl-tmp/vulkan/nvidia_egl_icd.json",
                   ACCEPT_EULA="Y", OMNI_KIT_ACCEPT_EULA="YES", PYTHONUNBUFFERED="1")


def save(**changes):
    state.update(changes)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=2))


def run_step(phase, arguments, report=None):
    log = validation / f"r6_{phase}.log"
    if report is not None:
        (base / report).unlink(missing_ok=True)
    with log.open("w") as stream:
        child = subprocess.Popen(["./python.sh", *arguments, "--headless", "--device", "cuda:0"],
                                 cwd=base, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        save(status="running", phase=phase, child_pid=child.pid, log=str(log))
        result = child.wait()
    assert result == 0, f"{phase} failed; see {log}"
    if report is not None:
        payload = json.loads((base / report).read_text())
        assert payload.get("pass", True), f"{phase} check failed"


try:
    assert source.is_file()
    run_step("preflight", [
        "scripts/smalldog/train_terrain_r6.py", "--checkpoint", str(source),
        "--num-envs", "220", "--check-only", "--output", "validation/r6_preflight.json",
    ], "validation/r6_preflight.json")
    run_step("training", [
        "scripts/smalldog/train_terrain_r6.py", "--checkpoint", str(source),
        "--num-envs", "4096", "--iterations", "10000",
    ])
    manifest = json.loads((validation / "r6_latest_run.json").read_text())
    final = Path(manifest["log_dir"]) / "model_22796.pt"
    assert final.is_file()
    save(run_dir=manifest["log_dir"], final_checkpoint=str(final))
    for speed, label in ((0.3, "0p3"), (0.6, "0p6"), (1.0, "1p0"), (1.5, "1p5")):
        report = f"validation/r6_final_{label}.json"
        run_step(f"evaluation_{label}", [
            "scripts/smalldog/evaluate_gait.py", "--r6", "--stratified-terrain-levels",
            "--max-terrain-level", "10", "--num-envs", "220", "--checkpoint", str(final),
            "--command-vx", str(speed), "--output", report,
        ], report)
    save(status="complete", phase="terrain_training_and_evaluation_complete", child_pid=None)
except Exception as error:
    save(status="failed", error=f"{type(error).__name__}: {error}", child_pid=None)
    raise
