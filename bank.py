"""bank.py -- the Drive dataset bank: token caches parked on Google Drive.

Adapted from MultiCore's scripts/drive_cache.py (measured chunk sizes and
all). Building the 5-shard NeoX token cache costs ~58 minutes of tokenising;
pulling the finished file back off Drive costs ~90 seconds. So it is built
once, pushed, and every later instance downloads it instead. MultiCore2
deliberately shares MultiCore's bank folder: the cache file is immutable and
identical for both projects, and two copies on Drive would just be two
chances for them to drift.

`try_pull` is an optimisation and NEVER raises. A missing token, a missing
rclone, a Drive outage, a truncated transfer -- all return False and leave
the caller to tokenise locally. A Drive problem must cost minutes of CPU,
not the run. Only `push` raises, because a silent upload failure would leave
the whole fleet quietly rebuilding the cache forever.

Credentials, in priority order:
  1. RCLONE_DRIVE_TOKEN / RCLONE_DRIVE_TOKEN_B64 in the environment -- how
     instances get it (vast/launch.py base64s the raw token into the
     container because the JSON would be mangled by the --env string).
  2. No token but the machine's own rclone config already has a `gdrive:`
     remote -- how the local Windows box works.
"""
import base64
import json
import os
import shutil
import subprocess
import tempfile
import time

REMOTE = "gdrive"
REMOTE_FOLDER = "multicore-cache"
BANK_REMOTE = f"{REMOTE}:{REMOTE_FOLDER}"
# What vast/benchmark.py ranged-reads to measure Drive throughput at boot.
BANK_FILE = "fineweb100_gpt-neox-20b_5shards_u16.bin"

# 256M, NOT the 8M default. Drive cannot resume a resumable-upload session
# mid-file, so a single failed chunk restarts the entire transfer -- and at
# 8M a 7 GB cache is ~900 sequential chunks, which reliably lost the race.
# Measured: the default burned all three retries (21 GB moved, nothing
# committed) in 12 minutes; 256M committed in 2m23s at 64 MB/s first try.
UPLOAD_CHUNK = "256M"
# Drive will not serve parallel range reads the way S3 does, but rclone still
# gets a useful multiple of single-stream throughput out of 8.
PULL_STREAMS = "8"


def _token():
    """The rclone Drive OAuth blob, or None if the environment has none."""
    tok = os.environ.get("RCLONE_DRIVE_TOKEN")
    if tok:
        return tok
    b64 = os.environ.get("RCLONE_DRIVE_TOKEN_B64")
    if not b64:
        return None
    try:
        return base64.b64decode(b64).decode()
    except Exception:
        return None


def _have_local_remote():
    """True if this machine's own rclone config already knows `gdrive:`."""
    try:
        proc = subprocess.run(["rclone", "listremotes"], capture_output=True,
                              text=True, timeout=30)
        return proc.returncode == 0 and f"{REMOTE}:" in proc.stdout
    except Exception:
        return False


def _rclone_ready():
    """True once an rclone binary exists AND a bare `rclone` call can see the
    bank remote. Installs rclone if missing; if credentials only exist as an
    env token, materialises them into the default config so callers that
    shell out to bare rclone (vast/benchmark.py) work too."""
    if not shutil.which("rclone"):
        try:
            subprocess.run("curl -fsSL https://rclone.org/install.sh | bash",
                           shell=True, check=True, capture_output=True,
                           timeout=300)
        except Exception as e:
            print(f"[bank] rclone install failed ({type(e).__name__})",
                  flush=True)
            return False
        if not shutil.which("rclone"):
            return False
    if _have_local_remote():
        return True
    token = _token()
    if not token:
        return False
    # Append, never overwrite: the config may hold other remotes.
    proc = subprocess.run(["rclone", "config", "file"], capture_output=True,
                          text=True, timeout=30)
    path = proc.stdout.strip().splitlines()[-1] if proc.returncode == 0 else ""
    if not path:
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(f"\n[{REMOTE}]\ntype = drive\ntoken = {token}\n")
    return _have_local_remote()


def _conf(token):
    """A 0600 temp rclone.conf holding `token`. Caller unlinks it."""
    fd, path = tempfile.mkstemp(suffix=".conf")       # mkstemp is 0600
    with os.fdopen(fd, "w") as f:
        f.write(f"[{REMOTE}]\ntype = drive\ntoken = {token}\n")
    return path


def _conf_args():
    """rclone --config arguments: a temp conf from the env token when one
    exists, else nothing (the machine's own config). Returns (args, cleanup).
    """
    token = _token()
    if token:
        path = _conf(token)
        return ["--config", path], lambda: os.unlink(path)
    return [], lambda: None


def _remote_size(name, conf_args, folder=REMOTE_FOLDER):
    """Bytes of `name` on the remote, or None if absent/unreachable."""
    proc = subprocess.run(
        ["rclone", "lsjson", f"{REMOTE}:{folder}/{name}"] + conf_args,
        capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        return None
    try:
        items = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    return items[0].get("Size") if items else None


def try_pull(local_path, folder=REMOTE_FOLDER):
    """Fetch `local_path`'s basename from Drive. True iff it now exists.

    Downloads to a sibling .drivepull and renames only after the size matches
    what the remote reports, so a killed transfer can never leave a truncated
    file that the next run mistakes for a finished cache. (NOT .partial --
    the cache builder resumes an interrupted local build from exactly that
    path, and a half-downloaded file landing there would be read as "shard N
    is already written" and silently corrupt the cache.)
    """
    name = os.path.basename(local_path)
    if not (shutil.which("rclone") or _token()):
        return False
    if not _rclone_ready():
        return False
    conf_args, cleanup = _conf_args()
    tmp = local_path + ".drivepull"
    try:
        want = _remote_size(name, conf_args, folder)
        if not want:
            print(f"[bank] {name} not on Drive -- building locally",
                  flush=True)
            return False
        print(f"[bank] pulling {name} ({want / 1e9:.2f} GB)", flush=True)
        t0 = time.time()
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        proc = subprocess.run(
            ["rclone", "copyto", f"{REMOTE}:{folder}/{name}", tmp]
            + conf_args
            + ["--multi-thread-streams", PULL_STREAMS, "--stats", "30s", "-v"],
            timeout=7200)
        dt = time.time() - t0
        if proc.returncode != 0:
            print(f"[bank] pull failed (rclone {proc.returncode}) after "
                  f"{dt:.0f}s -- building locally", flush=True)
            return False
        got = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if got != want:
            print(f"[bank] size mismatch: got {got:,}, want {want:,} "
                  f"-- discarding, building locally", flush=True)
            return False
        os.replace(tmp, local_path)
        print(f"[bank] BANK_PULL_OK {name} in {dt:.0f}s "
              f"({got * 8 / dt / 1e6:.0f} Mbps)", flush=True)
        return True
    except Exception as e:
        print(f"[bank] pull errored ({type(e).__name__}: {e}) -- "
              f"building locally", flush=True)
        return False
    finally:
        cleanup()
        if os.path.exists(tmp):
            os.remove(tmp)


def push(local_path, folder=REMOTE_FOLDER):
    """Upload `local_path` to Drive and verify the committed size. Raises."""
    name = os.path.basename(local_path)
    if not _rclone_ready():
        raise RuntimeError("no rclone binary / credentials available")
    conf_args, cleanup = _conf_args()
    try:
        local = os.path.getsize(local_path)
        print(f"[bank] pushing {name} ({local / 1e9:.2f} GB)", flush=True)
        t0 = time.time()
        subprocess.run(
            ["rclone", "copy", local_path, f"{REMOTE}:{folder}/"]
            + conf_args
            + ["--drive-chunk-size", UPLOAD_CHUNK, "--stats", "30s", "-v"],
            check=True, timeout=7200)
        dt = time.time() - t0
        remote = _remote_size(name, conf_args, folder)
        if remote != local:
            raise RuntimeError(
                f"upload verify failed: local {local:,}, remote {remote!r}")
        print(f"[bank] BANK_PUSH_OK {name} in {dt:.0f}s "
              f"({local * 8 / dt / 1e6:.0f} Mbps)", flush=True)
    finally:
        cleanup()
