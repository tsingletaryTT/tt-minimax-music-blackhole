#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Boot the MiniMax-Music3 TT container on the chips this process holds, with the merged weights, and wait for it.

    serve_music.py [--model-dir DIR | --weights-repo REPO] [--workaround auto|on|off] [--ui]

What it does that `tt-model serve` alone does not:
  * passes only the chips of the current lease (TT_VISIBLE_DEVICES, as gozer sets it) into the container, each under
    its own /dev/tenstorrent number, and refuses a docker command that does not have exactly one --device argument to replace;
  * serves a model directory (`HF_MODEL=<dir>`, mounted read-only with the blob directories its links point into);
  * keeps its own cache directories for this model directory and creates them first (a missing mount source is
    created by Docker as root, and the container then cannot write to it);
  * probes the image for a host problem: loading a cached tensor over about 32 MiB can hang, and with it
    the server's boot. With --workaround auto (the default) a hung probe selects a mounted copy of the image's
    `ttnn/operations/core.py` whose `as_tensor` never uses the flatbuffer cache. The edit is asserted to match once, the
    file is read-only, and its sha256 is recorded. `off` never patches; `on` always does;
  * waits for /v1/health with a boot deadline and a stall rule, saves the container log, and always stops the container.
With --ui it then runs the Gradio app in this process, so stopping the process (SIGTERM) stops the container.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

CATALOG_REPO = "jashansinghTT/minimax-music3-blackhole"
DEFAULT_WEIGHTS_REPO = "episod/minimax-music3-fiona-crapple"
CORE_PY = "/opt/tt-metal/ttnn/ttnn/operations/core.py"
CACHE_MOUNTS = {"/cache": "cache", "/weight-cache": "weights", "/tensor-cache": "tensors"}
PATCH_MARKER = "    if cache_file_name is None:\n        return torch_to_ttnn("
PATCH_NOTE = ("# LOCAL WORKAROUND (tt-minimax-music-blackhole): never use the flatbuffer cache; "
              "load_tensor_flatbuffer hangs above ~32 MiB on some hosts")
DEVICE_RE = re.compile(r"--device /dev/tenstorrent/\d+:/dev/tenstorrent/\d+")


class LauncherError(Exception):
    pass


# ---- pure helpers --------------------------------------------------------------------------------------------------

def leased_chips(env: dict) -> list[str]:
    """PCI addresses of the chips this process may use: gozer exports them in TT_VISIBLE_DEVICES."""
    raw = env.get("TT_VISIBLE_DEVICES", "")
    return [b for b in raw.split(",") if b]


def device_nodes(bdfs: list[str], sysfs=Path("/sys/bus/pci/devices")) -> list[int]:
    """The /dev/tenstorrent/<N> numbers for PCI addresses, read from sysfs (the number is not the position)."""
    nodes = []
    for bdf in bdfs:
        found = sorted(p.name.split("!")[-1] for p in (Path(sysfs) / bdf / "tenstorrent").glob("*")) \
            if (Path(sysfs) / bdf / "tenstorrent").is_dir() else []
        if not found:
            raise LauncherError(f"chip {bdf} has no /dev/tenstorrent node in sysfs")
        nodes.append(int(found[0]))
    return nodes


def rewrite_device(cmd: str, nodes: list[int]) -> str:
    """Replace the one --device argument `tt-model serve --print` wrote with one per leased node, under its own number."""
    if len(DEVICE_RE.findall(cmd)) != 1:
        raise LauncherError(f"expected exactly one --device argument in the docker command, found {len(DEVICE_RE.findall(cmd))}")
    # Each node keeps its own number inside the container. UMD's CHIP_IN_USE_<n>_PCIe lock is a file in /dev/shm, which
    # `--ipc host` shares between containers: renumbering two different chips to "0" made them wait on one lock.
    new = " ".join(f"--device /dev/tenstorrent/{n}:/dev/tenstorrent/{n}" for n in nodes)
    return DEVICE_RE.sub(lambda m: new, cmd, count=1)


def link_dirs(model_dir) -> list[str]:
    """Every directory a symlink under model_dir points into (the container needs each mounted at its own path)."""
    out = set()
    for root, _dirs, files in os.walk(model_dir):
        for f in files:
            p = Path(root) / f
            if p.is_symlink():
                out.add(str(Path(os.path.realpath(p)).parent))
    return sorted(out)


def add_model_dir(cmd: str, model_dir, display_name: str, extra_mounts=()) -> str:
    """Serve model_dir: mount it and the link targets read-only at their own paths, point HF_MODEL at it."""
    pat = re.compile(r"--env HF_MODEL=\S+")
    if len(pat.findall(cmd)) != 1:
        raise LauncherError("expected exactly one --env HF_MODEL=... in the docker command")
    cmd = re.sub(r"\s+--env HF_TOKEN(?=\s)", "", cmd)        # a local model directory needs no Hub access
    mounts = [str(model_dir), *extra_mounts]
    vols = " ".join(f"--volume {m}:{m}:ro" for m in dict.fromkeys(mounts))
    return pat.sub(lambda m: f"{vols} --env HF_MODEL={model_dir} --env HF_MODEL_ID={display_name}", cmd, count=1)


def with_cache_mounts(cmd: str, work_dir: Path) -> str:
    """Point the three cache mounts at work_dir/{cache,weights,tensors}, replacing tt-model's, so one model directory
    never reads another's cache (the weights cache holds a view of the model directory)."""
    for target in CACHE_MOUNTS:
        cmd = re.sub(rf"--volume \S+:{re.escape(target)}(?=\s)", "", cmd)
    cmd = re.sub(r"\s+--env (TT_METAL_CACHE|TT_DIT_CACHE_DIR|TT_CACHE_PATH)=\S+", "", cmd)
    vols = " ".join(f"--volume {work_dir / sub}:{target}" for target, sub in CACHE_MOUNTS.items())
    envs = (f"--env TT_METAL_CACHE=/cache --env TT_DIT_CACHE_DIR=/weight-cache --env TT_CACHE_PATH=/tensor-cache")
    return re.sub(r"(--env HF_HOME=\S+)", lambda m: f"{m.group(1)} {vols} {envs}", cmd, count=1) if "--env HF_HOME=" in cmd \
        else cmd.replace("docker run ", f"docker run {vols} {envs} ", 1)


def publish_port(cmd: str, port: int) -> str:
    """Publish the server on `port` on the host (inside the container it stays 20000)."""
    pat = re.compile(r"--publish \d+:20000")
    if len(re.findall(r"--publish \S+", cmd)) != 1 or len(pat.findall(cmd)) != 1:
        raise LauncherError("expected exactly one --publish argument (host:20000) in the docker command")
    return pat.sub(f"--publish {port}:20000", cmd, count=1)


def detach(cmd: str) -> str:
    return cmd.replace("docker run ", "docker run -d ", 1)


def patch_core(text: str) -> tuple[str, str]:
    """The image's core.py with `as_tensor` never using the flatbuffer cache, and the sha256 of the result."""
    if text.count(PATCH_MARKER) != 1:
        raise LauncherError(f"the cache line to edit must occur exactly once in core.py, found {text.count(PATCH_MARKER)}")
    patched = text.replace(PATCH_MARKER, f"    if True:  {PATCH_NOTE}\n        return torch_to_ttnn(", 1)
    return patched, hashlib.sha256(patched.encode()).hexdigest()


def probe_verdict(rc, timed_out: bool, out: str) -> str:
    if timed_out:
        return "hang"
    return "ok" if rc == 0 and "PROBE-OK" in out else "error"


def image_of(cmd: str) -> str:
    toks = shlex.split(cmd)
    return toks[toks.index("python") - 1]


# ---- the launcher ----------------------------------------------------------------------------------------------------

TOOL_DIRS = (".local/bin", ".tenstorrent-venv/bin")     # where tt-model and gozer usually live; a service's PATH lacks them


def ensure_path(env: dict, home: Path = Path.home()) -> None:
    """Append the usual tool directories to env["PATH"] when they exist and are not already there."""
    parts = [p for p in env.get("PATH", "").split(":") if p]
    for d in TOOL_DIRS:
        full = str(home / d)
        if (home / d).is_dir() and full not in parts:
            parts.append(full)
    env["PATH"] = ":".join(parts)


def default_runner(argv, timeout=None):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr, False
    except FileNotFoundError:
        return 127, f"command not found: {argv[0]} (PATH is {os.environ.get('PATH', '')})", False
    except subprocess.TimeoutExpired as e:
        return None, ((e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")), True


@dataclasses.dataclass
class Config:
    model_dir: Path
    display_name: str = "fiona-crapple-merged"
    env: dict = dataclasses.field(default_factory=lambda: dict(os.environ))
    sysfs: Path = Path("/sys/bus/pci/devices")
    work_dir: Path = Path.home() / ".cache" / "tt-minimax-music-blackhole"
    catalog_repo: str = CATALOG_REPO
    container_name: str = "tt-minimax-music-blackhole"
    workaround: str = "auto"                  # auto | on | off
    boot_deadline_s: float = 2400.0
    stall_s: float = 1200.0
    probe_timeout_s: float = 60.0
    port: int = 20000


class Launcher:
    def __init__(self, cfg: Config, runner: Callable = default_runner, health: Optional[Callable[[], bool]] = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep, log=print):
        self.cfg, self.run, self.clock, self.sleep, self.log = cfg, runner, clock, sleep, log
        self.health = health or self._http_health
        self.workaround: Optional[dict] = None

    def _http_health(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.cfg.port}/v1/health", timeout=3) as r:
                return r.status == 200
        except OSError:
            return False

    def _sh(self, cmd: str, timeout=None):
        return self.run(shlex.split(cmd), timeout)

    def _printed_command(self) -> str:
        for attempt in (0, 1):
            rc, out, _ = self.run(["tt-model", "serve", self.cfg.catalog_repo, "--local-only", "--print"], 120)
            lines = [l for l in out.splitlines() if l.startswith("docker run")]
            if lines:
                return lines[0]
            if attempt == 0:
                self.log(f"installing {self.cfg.catalog_repo} with `tt-model pull` (the container image, about 4 GB)")
                prc, pout, _ = self.run(["tt-model", "pull", self.cfg.catalog_repo], 3600)
                if prc != 0:
                    raise LauncherError(f"`tt-model pull {self.cfg.catalog_repo}` failed: {pout.strip()[-300:]}")
        raise LauncherError(f"`tt-model serve --print` gave no docker command: {out.strip()[-300:]}")

    def _probe(self, base_cmd: str, nodes: list[int]) -> str:
        cfg = self.cfg
        probe_dir = cfg.work_dir / "probe"
        (probe_dir / "cache").mkdir(parents=True, exist_ok=True)
        src = Path(__file__).with_name("kv_probe.py")
        if src.exists():
            (probe_dir / "kv_probe.py").write_text(src.read_text())
        name = f"{cfg.container_name}-probe"
        head = base_cmd[: base_cmd.index(image_of(base_cmd))]
        head = re.sub(r"--name \S+", f"--name {name}", head)
        head = re.sub(r"--publish \S+\s*", "", head)
        head = head.replace("docker run ", f"docker run --rm ", 1)
        cmd = f"{head}--volume {probe_dir}:/probe {image_of(base_cmd)} python /probe/kv_probe.py"
        rc, out, timed_out = self._sh(cmd, cfg.probe_timeout_s)
        verdict = probe_verdict(rc, timed_out, out)
        if verdict == "hang":
            self._sh(f"docker kill {name}")
        if verdict == "error":
            raise LauncherError(f"the cache probe failed (not a hang): {out.strip()[-400:]}")
        return verdict

    def _core_workaround(self, image: str, reason: str) -> tuple[str, dict]:
        rc, text, _ = self._sh(f"docker run --rm {image} cat {CORE_PY}", 120)
        if rc != 0:
            raise LauncherError(f"could not read core.py from the image: {text.strip()[-300:]}")
        patched, sha = patch_core(text)
        f = self.cfg.work_dir / "core.py.patched"
        f.write_text(patched)
        info = {"reason": reason, "sha256": sha, "file": str(f), "edit": "as_tensor skips the flatbuffer cache (one line)"}
        return f"--volume {f}:{CORE_PY}:ro", info

    def start(self) -> None:
        try:
            self._start()
        except BaseException:
            self.stop()
            raise

    def _start(self) -> None:
        cfg = self.cfg
        for sub in (*CACHE_MOUNTS.values(),):
            (cfg.work_dir / sub).mkdir(parents=True, exist_ok=True)       # before docker run: see the module docstring
        chips = leased_chips(cfg.env)
        nodes = device_nodes(chips, cfg.sysfs) if chips else [0]
        if not chips:
            self.log("TT_VISIBLE_DEVICES is not set: using /dev/tenstorrent/0 (run under `gozer run` to lease a chip)")
        base = self._printed_command()
        base = rewrite_device(base, nodes)
        image = image_of(base)
        extra = ""
        if cfg.workaround == "on":
            extra, self.workaround = self._core_workaround(image, "forced by --workaround on")
        elif cfg.workaround == "auto":
            self.log("probing the image for the cached-tensor load hang (up to a minute) ...")
            if self._probe(base, nodes) == "hang":
                extra, self.workaround = self._core_workaround(
                    image, f"probe hung: loading a 64 MiB cached tensor did not finish in {cfg.probe_timeout_s:.0f} s")
                self.log(f"workaround selected: {self.workaround['reason']}")
        cmd = add_model_dir(base, cfg.model_dir, cfg.display_name, extra_mounts=link_dirs(cfg.model_dir))
        cmd = with_cache_mounts(cmd, cfg.work_dir)
        cmd = publish_port(cmd, cfg.port)
        cmd = re.sub(r"--name \S+", f"--name {cfg.container_name}", cmd, count=1)
        if extra:
            cmd = cmd.replace("--ipc host", f"--ipc host {extra}", 1)
        rc, out, _ = self._sh(detach(cmd))
        if rc != 0:
            raise LauncherError(f"docker run failed: {out.strip()[-400:]}")
        self._wait_healthy()

    def _wait_healthy(self) -> None:
        cfg = self.cfg
        start = self.clock()
        last_change, last_size = start, -1
        while True:
            if self.health():
                self.log(f"healthy after {self.clock() - start:.0f} s")
                return
            rc, running, _ = self._sh(f"docker inspect -f {{{{.State.Running}}}} {cfg.container_name}")
            if rc != 0 or running.strip() != "true":
                raise LauncherError(f"the container exited during boot; log tail: {self._tail()}")
            now = self.clock()
            if now - start > cfg.boot_deadline_s:
                raise LauncherError(f"the server was not healthy after {cfg.boot_deadline_s:.0f} s; log tail: {self._tail()}")
            _, logs, _ = self._sh(f"docker logs {cfg.container_name}")
            if len(logs) != last_size:
                last_size, last_change = len(logs), now
            elif now - last_change > cfg.stall_s:
                raise LauncherError(f"the boot stalled: no new container log output for {cfg.stall_s:.0f} s; "
                                    f"log tail: {self._tail()}")
            self.sleep(10)

    def _tail(self) -> str:
        _, logs, _ = self._sh(f"docker logs --tail 8 {self.cfg.container_name}")
        return " | ".join(l[:160] for l in logs.strip().splitlines()[-4:])

    def stop(self) -> None:
        name = self.cfg.container_name
        try:
            self.cfg.work_dir.mkdir(parents=True, exist_ok=True)
            _, logs, _ = self._sh(f"docker logs {name}")
            (self.cfg.work_dir / "container.log").write_text(logs)
        except OSError:
            pass
        rc, _, _ = self._sh(f"docker stop -t 20 {name}")
        if rc != 0:
            self._sh(f"docker kill {name}")
        self._sh(f"docker rm {name}")


# ---- command line ------------------------------------------------------------------------------------------------------

def load_env_file(path: Path) -> None:
    """KEY=VALUE lines of a git-ignored .env.local, for machine-specific settings (MUSIC_MODEL_DIR and so on)."""
    if path.is_file():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def resolve_model_dir(args) -> Path:
    if args.model_dir:
        return Path(args.model_dir).expanduser()
    from huggingface_hub import snapshot_download
    ignore = ["flowmatching_vae.pth", "dav.pth", "qwen_7B/*", "assets/*", "figures/*", "scripts/*"]
    return Path(snapshot_download(args.weights_repo, ignore_patterns=ignore))


def main(argv=None) -> int:
    root = Path(__file__).resolve().parent.parent
    ensure_path(os.environ)
    load_env_file(root / ".env.local")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=os.environ.get("MUSIC_MODEL_DIR"))
    ap.add_argument("--weights-repo", default=os.environ.get("MUSIC_WEIGHTS_REPO", DEFAULT_WEIGHTS_REPO))
    ap.add_argument("--workaround", choices=("auto", "on", "off"), default=os.environ.get("MUSIC_WORKAROUND", "auto"))
    ap.add_argument("--name", default=os.environ.get("MUSIC_MODEL_NAME", "fiona-crapple-merged"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("MUSIC_SERVER_PORT", "20000")), help="host port for the server")
    ap.add_argument("--ui", action="store_true", help="run the Gradio app after the server is healthy")
    ap.add_argument("--ui-port", type=int, default=int(os.environ.get("MUSIC_UI_PORT", "7871")))
    args = ap.parse_args(argv)
    try:
        model_dir = resolve_model_dir(args)
        if not model_dir.is_dir():
            raise LauncherError(f"model directory {model_dir} does not exist")
        work = Path.home() / ".cache" / "tt-minimax-music-blackhole" / hashlib.sha256(str(model_dir).encode()).hexdigest()[:12]
        launcher = Launcher(Config(model_dir=model_dir, display_name=args.name, work_dir=work, workaround=args.workaround,
                                       port=args.port, container_name=f"tt-minimax-music-blackhole-{args.port}"))
        signal.signal(signal.SIGTERM, lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
        launcher.start()
    except LauncherError as e:
        print(f"serve_music: {e}", file=sys.stderr)
        return 2
    print(f"serving {args.name} on http://127.0.0.1:{launcher.cfg.port}"
          + (f" (workaround: {launcher.workaround['reason']})" if launcher.workaround else ""), flush=True)
    try:
        if args.ui:
            sys.path.insert(0, str(root / "demo"))
            import app
            app.launch(server_name=os.environ.get("MUSIC_UI_HOST", "0.0.0.0"), server_port=args.ui_port,
                       music_url=f"http://127.0.0.1:{launcher.cfg.port}")
        else:
            while True:
                time.sleep(30)
    except KeyboardInterrupt:
        pass
    finally:
        launcher.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
