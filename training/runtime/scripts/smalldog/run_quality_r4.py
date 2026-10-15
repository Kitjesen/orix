"""Run the bounded R4 pilot, followed by the requested motion-quality checks."""
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

base = Path(__file__).resolve().parents[2]
validation = base / "validation"
state_path = validation / "r4_quality_pipeline_state.json"
checkpoint = base / (
    "logs/rsl_rl/smalldog_v67_gait_moe_cts/"
    "2026-10-14_01-44-14_v67_joint_window_r3_rough_gait/model_2398.pt"
)
environment = os.environ.copy()
environment.update(
    VK_ICD_FILENAMES="/root/autodl-tmp/vulkan/nvidia_egl_icd.json",
    ACCEPT_EULA="Y", OMNI_KIT_ACCEPT_EULA="YES", PYTHONUNBUFFERED="1",
)
state = {"pipeline_pid": os.getpid(), "source_checkpoint": str(checkpoint),
         "num_envs": 4096, "additional_iterations": 400}


def save_state(**changes):
    state.update(changes)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, indent=2))


def run_step(name, script, arguments, report_path=None):
    log = validation / f"{name}.log"
    with log.open("w") as output:
        child = subprocess.Popen(
            ["./python.sh", f"scripts/smalldog/{script}", "--headless", "--device", "cuda:0",
             *arguments], cwd=base, env=environment, stdout=output, stderr=subprocess.STDOUT,
        )
        save_state(status="running", phase=name, child_pid=child.pid, log=str(log))
        return_code = child.wait()
    if return_code:
        raise RuntimeError(f"{name} exited with {return_code}; see {log}")
    if report_path is not None:
        report = json.loads((base / report_path).read_text())
        if report.get("pass") is False:
            raise RuntimeError(f"{name} reported a failed check")
    return log


try:
    flat_check = json.loads((validation / "r4_quality_import.json").read_text())
    assert flat_check["pass"] is True
    run_step("r4_quality_import_rough", "check_refinement_import.py", [
        "--rough", "--checkpoint", str(checkpoint),
        "--output", "validation/r4_quality_import_rough.json",
    ], "validation/r4_quality_import_rough.json")
    run_step("r4_quality_r3_baseline_1p0", "evaluate_gait.py", [
        "--checkpoint", str(checkpoint), "--command-vx", "1.0", "--num-envs", "16",
        "--output", "validation/r4_quality_r3_baseline_1p0.json",
    ], "validation/r4_quality_r3_baseline_1p0.json")
    run_root = base / "logs/rsl_rl/smalldog_v67_refined_gait_moe_cts"
    previous = set(run_root.glob("*/refinement_run.json"))
    run_step("r4_quality_training", "train_refinement.py", [
        "--checkpoint", str(checkpoint), "--num-envs", "4096", "--iterations", "400",
    ])
    created = set(run_root.glob("*/refinement_run.json")) - previous
    assert len(created) == 1
    manifest_path = created.pop()
    manifest = json.loads(manifest_path.read_text())
    final = manifest_path.parent / f"model_{manifest['expected_last_iteration']}.pt"
    assert final.is_file(), "Training did not produce its expected final checkpoint"
    save_state(run_dir=str(manifest_path.parent), final_checkpoint=str(final))
    for speed, label in ((0.3, "0p3"), (0.6, "0p6"), (1.0, "1p0")):
        report = f"validation/r4_quality_final_flat_{label}.json"
        run_step(f"r4_quality_final_flat_{label}", "evaluate_gait.py", [
            "--refined", "--checkpoint", str(final), "--command-vx", str(speed),
            "--num-envs", "16", "--output", report,
        ], report)
    run_step("r4_quality_final_recording", "record_gait_motion.py", [
        "--refined", "--checkpoint", str(final), "--command-vx", "1.0",
        "--output", "validation/r4_quality_recording.json",
    ], "validation/r4_quality_recording.json")
    save_state(status="complete", phase="pilot_and_evaluation_complete", child_pid=None)
except Exception as error:
    save_state(status="failed", error=f"{type(error).__name__}: {error}", child_pid=None)
    raise
