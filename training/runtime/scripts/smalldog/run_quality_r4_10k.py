"""Continue the completed R4 pilot for 10,000 iterations, then evaluate."""
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

base = Path(__file__).resolve().parents[2]
validation = base / "validation"
state_path = validation / "r4_quality_10k_state.json"
run_root = base / "logs/rsl_rl/smalldog_v67_refined_gait_moe_cts"
state = {"pipeline_pid": os.getpid(), "num_envs": 4096,
         "additional_iterations": 10000, "expected_last_iteration": 12796}
environment = os.environ.copy()
environment.update(
    VK_ICD_FILENAMES="/root/autodl-tmp/vulkan/nvidia_egl_icd.json",
    ACCEPT_EULA="Y", OMNI_KIT_ACCEPT_EULA="YES", PYTHONUNBUFFERED="1",
)


def save_state(**changes):
    state.update(changes)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=2))


def evaluate(final, speed, label):
    output = validation / f"r4_quality_10k_final_flat_{label}.json"
    log = validation / f"r4_quality_10k_final_flat_{label}.log"
    with log.open("w") as stdout:
        child = subprocess.Popen([
            "./python.sh", "scripts/smalldog/evaluate_gait.py", "--headless",
            "--device", "cuda:0", "--refined", "--checkpoint", str(final),
            "--command-vx", str(speed), "--num-envs", "16", "--output", str(output),
        ], cwd=base, env=environment, stdout=stdout, stderr=subprocess.STDOUT)
        save_state(status="running", phase=f"evaluate_{label}", child_pid=child.pid, log=str(log))
        return_code = child.wait()
    assert return_code == 0 and output.is_file(), f"Evaluation failed; see {log}"
    report = json.loads(output.read_text())
    assert report["checkpoint"] == str(final) and report["refined"]


try:
    pilot = json.loads((validation / "r4_quality_pipeline_state.json").read_text())
    assert pilot["status"] == "complete", "The R4 pilot and evaluations must finish first"
    source = Path(pilot["final_checkpoint"])
    assert source.is_file() and source.name == "model_2797.pt"
    manifest = json.loads((source.parent / "refinement_run.json").read_text())
    assert manifest["task"] == "RobotLab-SmallDog-V67-FlatRefinedGait-v0"
    assert source.parent.name == "2026-10-14_14-19-28_v67_r4_pitch_refinement_flat_pilot"
    save_state(source_checkpoint=str(source), starting_iteration=2797, load_optimizer=True)
    existing_runs = set(run_root.glob("*"))
    log = validation / "r4_quality_10k_training.log"
    with log.open("w") as stdout:
        child = subprocess.Popen(["bash", "continue_v67_quality_r4_10k.sh"], cwd=base,
                                 env=environment, stdout=stdout, stderr=subprocess.STDOUT)
        save_state(status="running", phase="training", child_pid=child.pid, log=str(log))
        return_code = child.wait()
    assert return_code == 0, f"Training failed; see {log}"
    created = {p for p in run_root.glob("*") if p.is_dir()} - existing_runs
    assert len(created) == 1, "Expected one continuation run"
    run_dir = created.pop()
    final = run_dir / "model_12796.pt"
    assert final.is_file(), "The 10,000-iteration final checkpoint is missing"
    save_state(run_dir=str(run_dir), final_checkpoint=str(final))
    for speed, label in ((0.3, "0p3"), (0.6, "0p6"), (1.0, "1p0")):
        evaluate(final, speed, label)
    save_state(status="complete", phase="continuation_and_evaluation_complete", child_pid=None)
except Exception as error:
    save_state(status="failed", error=f"{type(error).__name__}: {error}", child_pid=None)
    raise
