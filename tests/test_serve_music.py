"""The launcher's pure parts (device mapping, docker command edits, the cache workaround) and its orchestration
with an injected runner, so no test touches Docker or a chip."""
import hashlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "serve"))
import serve_music as sm   # noqa: E402

PRINTED = ("docker run --name tt-model-minimax-music3-p150 --user 1000:1000 --label a=b "
           "--device /dev/tenstorrent/0:/dev/tenstorrent/0 --ipc host --volume /home/u/.cache/huggingface:/hf "
           "--env HF_HOME=/hf --publish 20000:20000 --env HF_TOKEN --env HF_MODEL=MiniMaxAI/MiniMax-Music3 "
           "--env MESH_DEVICE=P150 tt-model/minimax-music3:abc python -m uvicorn --port 20000 app:app")


# ---- leased chips and device nodes ----------------------------------------------------------------------------

def make_sysfs(root, mapping):
    for bdf, node in mapping.items():
        d = root / bdf / "tenstorrent"
        d.mkdir(parents=True)
        (d / f"tenstorrent!{node}").mkdir()
    return root


def test_leased_chips_come_from_tt_visible_devices_in_order():
    assert sm.leased_chips({"TT_VISIBLE_DEVICES": "0000:03:00.0,0000:04:00.0"}) == ["0000:03:00.0", "0000:04:00.0"]
    assert sm.leased_chips({}) == []
    assert sm.leased_chips({"TT_VISIBLE_DEVICES": ""}) == []


def test_device_nodes_follow_the_pci_address_not_the_position(tmp_path):
    root = make_sysfs(tmp_path, {"0000:03:00.0": 2, "0000:04:00.0": 3})
    assert sm.device_nodes(["0000:04:00.0", "0000:03:00.0"], sysfs=root) == [3, 2]


def test_a_chip_with_no_device_node_is_an_error(tmp_path):
    with pytest.raises(sm.LauncherError, match="0000:09:00.0"):
        sm.device_nodes(["0000:09:00.0"], sysfs=tmp_path)


def test_rewrite_device_passes_only_the_given_nodes_as_devices_0_and_1():
    out = sm.rewrite_device(PRINTED, [2, 3])
    assert "--device /dev/tenstorrent/2:/dev/tenstorrent/0 --device /dev/tenstorrent/3:/dev/tenstorrent/1" in out
    assert out.count("--device") == 2 and "/dev/tenstorrent/0:/dev/tenstorrent/0" not in out


def test_rewrite_device_matches_whichever_node_tt_model_printed():
    printed = PRINTED.replace("/dev/tenstorrent/0:/dev/tenstorrent/0", "/dev/tenstorrent/2:/dev/tenstorrent/2")
    assert "--device /dev/tenstorrent/3:/dev/tenstorrent/0" in sm.rewrite_device(printed, [3])


@pytest.mark.parametrize("cmd", ["docker run --ipc host img", PRINTED + " --device /dev/tenstorrent/1:/dev/tenstorrent/1"])
def test_rewrite_device_requires_exactly_one_device_argument(cmd):
    with pytest.raises(sm.LauncherError, match="exactly one"):
        sm.rewrite_device(cmd, [0])


# ---- the model directory ----------------------------------------------------------------------------------------

def test_link_dirs_lists_each_directory_the_links_point_into_once(tmp_path):
    blobs = tmp_path / "hub" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "aa").write_text("x")
    (blobs / "bb").write_text("y")
    md = tmp_path / "model"
    (md / "language_model").mkdir(parents=True)
    os.symlink(blobs / "aa", md / "config.json")
    os.symlink(blobs / "bb", md / "language_model" / "w.safetensors")
    (md / "real.txt").write_text("not a link")
    assert sm.link_dirs(md) == [str(blobs)]


def test_add_model_dir_mounts_it_read_only_and_sets_hf_model_and_the_display_name(tmp_path):
    md = tmp_path / "model"
    md.mkdir()
    out = sm.add_model_dir(PRINTED, md, "fiona-crapple-merged", extra_mounts=["/data/blobs"])
    assert f"--volume {md}:{md}:ro" in out and "--volume /data/blobs:/data/blobs:ro" in out
    assert f"--env HF_MODEL={md}" in out and "--env HF_MODEL=MiniMaxAI/MiniMax-Music3" not in out
    assert "--env HF_MODEL_ID=fiona-crapple-merged" in out


def test_add_model_dir_requires_exactly_one_hf_model(tmp_path):
    with pytest.raises(sm.LauncherError, match="HF_MODEL"):
        sm.add_model_dir(PRINTED.replace("--env HF_MODEL=MiniMaxAI/MiniMax-Music3", ""), tmp_path, "x")


def test_detach_adds_dash_d_once():
    out = sm.detach(PRINTED)
    assert out.startswith("docker run -d ") and out.count(" -d ") == 1


# ---- the workaround for the tensor-cache load hang -----------------------------------------------------------------

CORE = ("def as_tensor(...):\n    def torch_to_ttnn(...):\n        pass\n\n"
        "    if cache_file_name is None:\n        return torch_to_ttnn(tensor, dtype, layout, device, memory_config, mesh_mapper)\n\n"
        "    def from_torch_and_dump(...):\n        pass\n")


def test_patch_core_changes_exactly_one_line_and_returns_the_hash_of_the_result():
    patched, sha = sm.patch_core(CORE)
    assert patched.count("if True:") == 1 and "if cache_file_name is None:" not in patched
    assert patched.replace("if True:  # " + patched.split("if True:  # ")[1].split("\n")[0], "if cache_file_name is None:") == CORE
    assert sha == hashlib.sha256(patched.encode()).hexdigest()


@pytest.mark.parametrize("text", ["nothing to edit\n", CORE + CORE])
def test_patch_core_refuses_when_the_line_is_not_there_exactly_once(text):
    with pytest.raises(sm.LauncherError, match="exactly once"):
        sm.patch_core(text)


@pytest.mark.parametrize("rc,timed_out,out,want", [
    (0, False, "PROBE-OK 0.4", "ok"),
    (0, False, "no marker here", "error"),
    (None, True, "", "hang"),
    (1, False, "Traceback ...", "error"),
])
def test_probe_verdict(rc, timed_out, out, want):
    assert sm.probe_verdict(rc, timed_out, out) == want


# ---- orchestration with an injected runner ---------------------------------------------------------------------------

class Runner:
    """Records every command and answers by prefix. `answers` maps a substring of the command to (rc, out, timed_out)."""
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    def __call__(self, argv, timeout=None):
        cmd = " ".join(argv) if isinstance(argv, (list, tuple)) else argv
        self.calls.append(cmd)
        for key, res in self.answers.items():
            if key in cmd:
                return res(cmd) if callable(res) else res
        return (0, "", False)


def make_launcher(tmp_path, probe, health_ok_after=1, stalled=False, runner_extra=None):
    md = tmp_path / "model"
    md.mkdir()
    sysfs = make_sysfs(tmp_path / "sys", {"0000:03:00.0": 2})
    answers = {
        "--print": (0, PRINTED, False),
        "cat /opt/tt-metal/ttnn/ttnn/operations/core.py": (0, CORE, False),
        "kv_probe.py": probe,
        "docker inspect": (0, "true", False),
        "docker logs": (0, "log line\n", False),
        **(runner_extra or {}),
    }
    r = Runner(answers)
    ticks = {"health": 0, "t": 0.0}

    def health():
        ticks["health"] += 1
        return ticks["health"] > health_ok_after

    def clock():
        ticks["t"] += 1.0
        return ticks["t"]
    cfg = sm.Config(model_dir=md, display_name="fiona-crapple-merged", env={"TT_VISIBLE_DEVICES": "0000:03:00.0"},
                    sysfs=sysfs, work_dir=tmp_path / "work", boot_deadline_s=50, stall_s=20, probe_timeout_s=60)
    return sm.Launcher(cfg, runner=r, health=health, clock=clock, sleep=lambda s: None), r


def docker_run_call(runner):
    return [c for c in runner.calls if c.startswith("docker run") and " -d " in c][0]


def test_probe_ok_boots_without_the_workaround(tmp_path):
    L, r = make_launcher(tmp_path, probe=(0, "PROBE-OK 0.3", False))
    L.start()
    run = docker_run_call(r)
    assert "core.py" not in run and L.workaround is None
    assert "--device /dev/tenstorrent/2:/dev/tenstorrent/0" in run


def test_probe_hang_selects_the_workaround_and_records_it(tmp_path):
    L, r = make_launcher(tmp_path, probe=(None, "", True))
    L.start()
    run = docker_run_call(r)
    assert ":/opt/tt-metal/ttnn/ttnn/operations/core.py:ro" in run
    assert L.workaround["reason"].startswith("probe hung") and len(L.workaround["sha256"]) == 64
    patched = Path(L.workaround["file"]).read_text()
    assert patched.count("if True:") == 1
    assert any(c.startswith("docker kill") and "probe" in c for c in r.calls)      # the hung probe container is killed


def test_probe_error_does_not_select_the_workaround_and_stops(tmp_path):
    L, r = make_launcher(tmp_path, probe=(1, "Traceback: no device", False))
    with pytest.raises(sm.LauncherError, match="probe failed"):
        L.start()
    assert not [c for c in r.calls if c.startswith("docker run") and " -d " in c]


def test_forced_workaround_skips_the_probe(tmp_path):
    L, r = make_launcher(tmp_path, probe=(0, "PROBE-OK", False))
    L.cfg.workaround = "on"
    L.start()
    assert not [c for c in r.calls if "kv_probe.py" in c] and L.workaround["reason"] == "forced by --workaround on"


def test_workaround_off_boots_even_when_the_probe_hangs(tmp_path):
    L, r = make_launcher(tmp_path, probe=(None, "", True))
    L.cfg.workaround = "off"
    L.start()
    assert "core.py" not in docker_run_call(r) and L.workaround is None


def test_cache_directories_are_created_before_docker_run(tmp_path):
    L, r = make_launcher(tmp_path, probe=(0, "PROBE-OK", False))
    L.start()
    for d in ("cache", "weights", "tensors"):
        assert (L.cfg.work_dir / d).is_dir()


def test_stall_ends_the_boot_with_the_log_tail(tmp_path):
    L, r = make_launcher(tmp_path, probe=(0, "PROBE-OK", False), health_ok_after=10_000)
    with pytest.raises(sm.LauncherError, match="stalled"):
        L.start()


def test_boot_deadline_is_enforced_even_when_the_log_keeps_moving(tmp_path):
    n = {"i": 0}

    def moving(cmd):
        n["i"] += 1
        return (0, "x" * n["i"], False)
    L, r = make_launcher(tmp_path, probe=(0, "PROBE-OK", False), health_ok_after=10_000, runner_extra={"docker logs": moving})
    with pytest.raises(sm.LauncherError, match="not healthy after"):
        L.start()


def test_an_exited_container_ends_the_boot_at_once(tmp_path):
    L, r = make_launcher(tmp_path, probe=(0, "PROBE-OK", False), health_ok_after=10_000,
                         runner_extra={"docker inspect": (0, "false", False)})
    with pytest.raises(sm.LauncherError, match="exited"):
        L.start()


def test_stop_saves_the_log_then_stops_then_removes_and_kills_when_stop_fails(tmp_path):
    L, r = make_launcher(tmp_path, probe=(0, "PROBE-OK", False), runner_extra={"docker stop": (1, "", False)})
    L.start()
    L.stop()
    order = [i for i, c in enumerate(r.calls) if c.startswith(("docker logs", "docker stop", "docker kill", "docker rm"))]
    kinds = [r.calls[i].split()[1] for i in order]
    assert kinds.index("logs") < kinds.index("stop") < kinds.index("kill") < kinds.index("rm")
    assert (L.cfg.work_dir / "container.log").exists()


def test_start_failure_still_stops_the_container(tmp_path):
    L, r = make_launcher(tmp_path, probe=(0, "PROBE-OK", False), health_ok_after=10_000)
    with pytest.raises(sm.LauncherError):
        L.start()
    assert any(c.startswith("docker stop") for c in r.calls)


def test_the_host_hf_token_is_not_passed_into_a_container_that_serves_a_local_directory(tmp_path):
    out = sm.add_model_dir(PRINTED, tmp_path, "x")
    assert "HF_TOKEN" not in out


def test_forced_workaround_really_mounts_the_patched_file(tmp_path):
    L, r = make_launcher(tmp_path, probe=(0, "PROBE-OK", False))
    L.cfg.workaround = "on"
    L.start()
    assert ":/opt/tt-metal/ttnn/ttnn/operations/core.py:ro" in docker_run_call(r)
    text = Path(L.workaround["file"]).read_text()
    assert text.count("if True:") == 1 and hashlib.sha256(text.encode()).hexdigest() == L.workaround["sha256"]


# ---- the server's published port ------------------------------------------------------------------------------------------

def test_publish_port_changes_only_the_host_side():
    out = sm.publish_port(PRINTED, 20001)
    assert "--publish 20001:20000" in out and "--publish 20000:20000" not in out


@pytest.mark.parametrize("cmd", ["docker run --ipc host img python x", PRINTED + " --publish 9:9"])
def test_publish_port_requires_exactly_one_publish_argument(cmd):
    with pytest.raises(sm.LauncherError, match="exactly one --publish"):
        sm.publish_port(cmd, 20001)


def test_the_launcher_publishes_its_configured_port(tmp_path):
    L, r = make_launcher(tmp_path, probe=(0, "PROBE-OK", False))
    L.cfg.port = 20005
    L.start()
    assert "--publish 20005:20000" in docker_run_call(r)


# ---- running under a service manager: a short PATH, a missing command -------------------------------------------------------

def test_a_missing_command_is_a_plain_result_not_a_traceback():
    rc, out, timed_out = sm.default_runner(["definitely-not-a-real-command-xyz", "--x"])
    assert rc == 127 and not timed_out and "command not found: definitely-not-a-real-command-xyz" in out


def test_ensure_path_adds_the_usual_tool_directories_once(tmp_path, monkeypatch):
    home = tmp_path
    for d in (".local/bin", ".tenstorrent-venv/bin"):
        (home / d).mkdir(parents=True)
    env = {"PATH": "/usr/bin:/bin"}
    sm.ensure_path(env, home=home)
    parts = env["PATH"].split(":")
    assert parts[:2] == ["/usr/bin", "/bin"]
    assert str(home / ".local/bin") in parts and str(home / ".tenstorrent-venv/bin") in parts
    before = env["PATH"]
    sm.ensure_path(env, home=home)
    assert env["PATH"] == before                                    # no duplicates on a second call


def test_ensure_path_skips_directories_that_do_not_exist(tmp_path):
    env = {"PATH": "/usr/bin"}
    sm.ensure_path(env, home=tmp_path)
    assert env["PATH"] == "/usr/bin"


def test_a_missing_tt_model_is_a_launcher_error_that_names_the_command(tmp_path):
    L, r = make_launcher(tmp_path, probe=(0, "PROBE-OK", False),
                         runner_extra={"--print": (127, "command not found: tt-model", False)})
    with pytest.raises(sm.LauncherError, match="tt-model"):
        L.start()
