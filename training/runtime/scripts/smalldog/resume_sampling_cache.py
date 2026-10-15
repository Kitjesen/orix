"""Apply the measured sampling fix at the next complete terrain checkpoint."""
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
state_path = validation / "r5_sampling_cache_transition_state.json"
canonical = validation / "r5_terrain_recovery_state.json"
old = json.loads(canonical.read_text())
run_root = base / "logs/rsl_rl/smalldog_v67_refined_gait_moe_cts"
source_run = run_root / "2026-10-15_11-08-37_v67_r5_terrain_recovery"
initial_latest = 10000
state = {"pipeline_pid": os.getpid(), "status": "running", "phase": "waiting_for_checkpoint",
         "task": old["task"], "num_envs": 4096, "expected_last_iteration": 12796,
         "wait_for_checkpoint_after": initial_latest, "initial_terrain_level": 0,
         "terrain_curriculum": True, "sampling_metadata_cache": True}
owns_training = False
environment = os.environ.copy()
environment.update(VK_ICD_FILENAMES="/root/autodl-tmp/vulkan/nvidia_egl_icd.json",
                   ACCEPT_EULA="Y", OMNI_KIT_ACCEPT_EULA="YES", PYTHONUNBUFFERED="1")


def save(**changes):
    state.update(changes)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    text = json.dumps(state, indent=2)
    state_path.write_text(text)
    if owns_training:
        canonical.write_text(text)


def stop_owned(pid, script, first_signal):
    proc = Path(f"/proc/{pid}")
    command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    assert script in command and (proc / "cwd").resolve() == base, command
    os.kill(pid, first_signal)
    for _ in range(30):
        try:
            if ") Z " in (proc / "stat").read_text():
                return
        except FileNotFoundError:
            return
        time.sleep(1)
    os.kill(pid, signal.SIGTERM)
    for _ in range(15):
        try:
            if ") Z " in (proc / "stat").read_text():
                return
        except FileNotFoundError:
            return
        time.sleep(1)
    raise RuntimeError(f"Owned process {pid} did not exit")


def run_step(phase, arguments, output=None):
    log = validation / f"r5_sampling_cache_{phase}.log"
    with log.open("w") as stream:
        child = subprocess.Popen(["./python.sh", *arguments, "--headless", "--device", "cuda:0"],
                                 cwd=base, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        save(status="running", phase=phase, child_pid=child.pid, log=str(log))
        result = child.wait()
    assert result == 0, f"{phase} failed: {log}"
    if output is not None:
        assert (base / output).is_file(), f"Missing {phase} report"


try:
    check = json.loads((validation / "r5_sampling_profile_cached.json").read_text())
    assert check["contact_and_reward_equivalence"]
    save()
    while True:
        assert Path(f"/proc/{old['child_pid']}/cmdline").exists(), "Source trainer exited"
        candidate = max(source_run.glob("model_*.pt"), key=lambda path: int(path.stem[6:]))
        if int(candidate.stem[6:]) > initial_latest:
            try:
                payload = torch.load(candidate, map_location="cpu", weights_only=False)
            except (EOFError, RuntimeError):
                time.sleep(1)
                continue
            assert payload["iter"] == int(candidate.stem[6:])
            assert "optimizer_state_dict" in payload and "optimizer_stu_enc_state_dict" in payload
            del payload
            break
        time.sleep(2)
    save(phase="applying_sampling_fix", source_checkpoint=str(candidate))
    stop_owned(old["pipeline_pid"], "scripts/smalldog/recover_terrain_training.py", signal.SIGTERM)
    stop_owned(old["child_pid"], "scripts/rsl_rl/train.py", signal.SIGINT)
    log = Path(old["log"]).read_text()
    last_logged = int(re.findall(r"Learning iteration\s+(\d+)/", log)[-1])
    starting = int(candidate.stem[6:])
    old.update(status="superseded", phase="sampling_cache_resume",
               preserved_checkpoint=str(candidate), replacement_state=str(state_path))
    (validation / "r5_terrain_before_sampling_cache.json").write_text(json.dumps(old, indent=2))
    owns_training = True
    save(starting_iteration=starting, additional_iterations=12797-starting,
         source_last_logged_iteration=last_logged,
         unsaved_iterations_not_transferred=max(0, last_logged-starting),
         curriculum_restore="Standard restart: level 0 and speed ceiling 0.35 m/s; policy and both optimizers restored")
    previous = set(run_root.glob("*"))
    run_step("training", [
        "scripts/rsl_rl/train.py", "--task", state["task"], "--num_envs", "4096",
        "--max_iterations", str(state["additional_iterations"]), "--logger", "tensorboard",
        "--run_name", "v67_r5_sampling_cache", "--resume", "--load_run", source_run.name,
        "--checkpoint", candidate.name,
    ])
    created = {path for path in run_root.glob("*") if path.is_dir()} - previous
    assert len(created) == 1
    run_dir = created.pop()
    final = run_dir / "model_12796.pt"
    assert final.is_file()
    save(run_dir=str(run_dir), final_checkpoint=str(final))
    for speed, label in ((0.3, "0p3"), (0.6, "0p6"), (1.0, "1p0")):
        output = f"validation/r5_terrain_final_{label}.json"
        run_step(f"evaluation_{label}", [
            "scripts/smalldog/evaluate_gait.py", "--refined", "--rough", "--max-terrain-level", "3",
            "--num-envs", "80", "--checkpoint", str(final), "--command-vx", str(speed),
            "--output", output,
        ], output)
    save(status="complete", phase="terrain_training_and_evaluation_complete", child_pid=None)
except Exception as error:
    save(status="failed", error=f"{type(error).__name__}: {error}")
    raise
