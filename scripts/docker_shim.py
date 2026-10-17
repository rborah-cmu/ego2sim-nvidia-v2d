#!/usr/bin/env python3
"""Minimal `docker` stand-in for riddle (no Docker access).

Translates the V2D reconstruction wrappers' container calls into native runs:

    docker run [--rm] [-it] [--gpus X] [--runtime=nvidia] [--user U:G] [--ipc=host] [--shm-size N]
               [-e K=V]... [-v HOST:CONT[:ro]]... [-w DIR] IMAGE CMD ARGS...

becomes CMD ARGS... executed with ~/v2d/envs/<IMAGE>/.venv/bin first on PATH, every argument
(and -e value) that starts with a mounted container path rewritten to its host path. Small
.json/.yaml/.txt input files that mention mounted container paths are passed as rewritten
temporary copies. The
`--dev` mount of reconstruction/modules at /workspace is ignored: the venvs install those
modules editable from the pinned checkout already.

Also answers `docker version` and `docker image inspect IMAGE` (success iff the env exists).
`docker build` fails with a pointer to scripts/setup_recon_envs.sh.

Env vars: V2D_ENVS (default ~/v2d/envs), V2D_SHIM_DEBUG=1 prints the translated command.
"""
import os
import shlex
import sys
import tempfile
from pathlib import Path

ENVS = Path(os.environ.get("V2D_ENVS", Path.home() / "v2d" / "envs"))
TEXT_SUFFIXES = {".json", ".yaml", ".yml", ".txt"}
IGNORED_FLAGS = {"--rm", "-it", "-i", "-t", "-d", "--init", "--privileged"}
IGNORED_WITH_VALUE = {"--gpus", "--runtime", "--user", "-u", "--ipc", "--shm-size", "--network",
                      "--net", "--name", "--entrypoint", "-p", "--publish", "--ulimit", "--pid"}


def die(msg: str, code: int = 125) -> None:
    print(f"docker-shim: {msg}", file=sys.stderr)
    sys.exit(code)


def image_env(image: str) -> Path:
    name = image.split("/")[-1].split(":")[0]
    return ENVS / name / ".venv"


def split_opt(args: list[str], i: int) -> tuple[str, str, int]:
    """Return (flag, value, next_index) for `--flag value` or `--flag=value`."""
    a = args[i]
    if a.startswith("--") and "=" in a:
        flag, value = a.split("=", 1)
        return flag, value, i + 1
    if i + 1 >= len(args):
        die(f"missing value for {a}")
    return a, args[i + 1], i + 2


def cmd_run(args: list[str]) -> None:
    env = os.environ.copy()
    mounts: list[tuple[str, str]] = []
    workdir = None
    i = 0
    while i < len(args) and args[i].startswith("-"):
        a = args[i]
        flag = a.split("=", 1)[0]
        if a in IGNORED_FLAGS:
            i += 1
        elif flag in IGNORED_WITH_VALUE:
            _, _, i = split_opt(args, i)
        elif flag in ("-e", "--env"):
            _, kv, i = split_opt(args, i)
            if "=" in kv:
                k, v = kv.split("=", 1)
            else:
                k, v = kv, os.environ.get(kv, "")
            if k != "HOME":  # containers use HOME=/tmp; keep the real home natively
                env[k] = v
        elif flag in ("-v", "--volume"):
            _, spec, i = split_opt(args, i)
            parts = spec.split(":")
            if len(parts) < 2:
                die(f"anonymous volumes are not supported: {spec}")
            mounts.append((parts[0], parts[1]))
        elif flag in ("-w", "--workdir"):
            _, workdir, i = split_opt(args, i)
        else:
            die(f"unsupported docker run option {a}")
    if i >= len(args):
        die("no image given")
    image, command = args[i], args[i + 1:]
    venv = image_env(image)
    if not (venv / "bin" / "python").exists():
        die(f"no native env for image {image!r} (expected {venv}); build it with scripts/setup_recon_envs.sh")
    if not command:
        die(f"image {image!r} has no default command in the shim; pass one explicitly")

    # /workspace is the --dev mount of reconstruction/modules: the venv already has those sources.
    mounts = [(h, c.rstrip("/")) for h, c in mounts if c.rstrip("/") != "/workspace"]
    mounts.sort(key=lambda m: len(m[1]), reverse=True)

    def rewrite(value: str) -> str:
        for host, cont in mounts:
            if value == cont or value.startswith(cont + "/"):
                return host.rstrip("/") + value[len(cont):]
        return value

    def rewrite_file(path: str) -> str:
        # Wrappers sometimes write container paths into small config files (e.g. SAM2 prompt
        # JSON mask_path). Pass a rewritten temporary copy instead.
        p = Path(path)
        if p.suffix.lower() not in TEXT_SUFFIXES or not p.is_file() or p.stat().st_size > 50_000_000:
            return path
        text = p.read_text(errors="surrogateescape")
        new = text
        for host, cont in mounts:
            new = new.replace(cont + "/", host.rstrip("/") + "/")
        if new == text:
            return path
        tmp = Path(tempfile.mkdtemp(prefix="docker_shim_")) / p.name
        tmp.write_text(new, errors="surrogateescape")
        return str(tmp)

    command = [rewrite_file(rewrite(c)) for c in command]
    for k in list(env):
        if k not in os.environ or env[k] != os.environ[k]:
            env[k] = rewrite(env[k])
    if workdir is not None:
        workdir = rewrite(workdir)
        if workdir.startswith("/workspace") or not os.path.isdir(workdir):
            workdir = None
    env["PATH"] = f"{venv / 'bin'}:{env.get('PATH', '')}"
    env["VIRTUAL_ENV"] = str(venv)
    env.pop("PYTHONHOME", None)
    if command[0] in ("python", "python3"):
        command[0] = str(venv / "bin" / "python")
    if os.environ.get("V2D_SHIM_DEBUG"):
        print("docker-shim:", shlex.join(command), file=sys.stderr)
    if workdir:
        os.chdir(workdir)
    os.execvpe(command[0], command, env)


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        die("usage: docker run|version|image inspect ...", 1)
    sub = argv[0]
    if sub == "run":
        cmd_run(argv[1:])
    elif sub == "version":
        print("docker-shim (native uv envs under %s)" % ENVS)
    elif sub == "image" and len(argv) >= 3 and argv[1] == "inspect":
        ok = all((image_env(img) / "bin" / "python").exists() for img in argv[2:])
        print("[]" if not ok else "[{}]")
        sys.exit(0 if ok else 1)
    elif sub == "build":
        die("cannot build images on riddle; build the native env with scripts/setup_recon_envs.sh", 1)
    else:
        die(f"unsupported subcommand {sub!r}", 1)


if __name__ == "__main__":
    main()
