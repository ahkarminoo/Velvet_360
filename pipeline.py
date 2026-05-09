import subprocess
import os
import re
import math
import cv2
import time
import threading
import platform
from pathlib import Path
from typing import Callable, Optional

HUGIN_BIN = os.environ.get("HUGIN_BIN", "/usr/bin")


def _bin(name):
    suffix = ".exe" if platform.system() == "Windows" else ""
    return os.path.join(HUGIN_BIN, f"{name}{suffix}")


def run_hugin(cmd, cwd, can_fail=False, timeout=600, discard_stdout=False):
    """Run a Hugin command. Returns True on success. If can_fail=True, returns False instead of crashing."""
    tool = os.path.basename(cmd[0])
    stdout_target = subprocess.DEVNULL if discard_stdout else subprocess.PIPE
    try:
        result = subprocess.run(
            cmd, cwd=cwd, check=True,
            stdout=stdout_target, stderr=subprocess.PIPE,
            timeout=timeout
        )
        if not discard_stdout and result.stdout:
            out = result.stdout.decode('utf-8', errors='ignore').strip()
            if out:
                print(out[:300])
        return True
    except subprocess.TimeoutExpired:
        print(f"[FAIL] {tool} timed out after {timeout}s")
        return can_fail
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.decode('utf-8', errors='ignore').strip() if e.stderr else ''
        print(f"[FAIL] {tool} exited {e.returncode}")
        if stderr_msg:
            print(f"  {stderr_msg[:800]}")
        return can_fail
    except FileNotFoundError:
        print(f"[FAIL] {tool} not found — check HUGIN_BIN={HUGIN_BIN}")
        return False


def run_hugin_streaming(cmd, cwd, timeout=600):
    """Like run_hugin but streams stdout/stderr line-by-line so progress is visible during long runs."""
    tool = os.path.basename(cmd[0])
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
    except FileNotFoundError:
        print(f"[FAIL] {tool} not found — check HUGIN_BIN={HUGIN_BIN}")
        return False

    def _reader():
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(f"  {tool}: {line[:200]}", flush=True)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        print(f"[FAIL] {tool} timed out after {timeout}s")
        return False
    t.join(timeout=2)
    if proc.returncode != 0:
        print(f"[FAIL] {tool} exited {proc.returncode}")
        return False
    return True


def inject_angles_into_pto(pto_path: Path):
    """Set each image's yaw and pitch in the PTO from its el/az filename."""
    lines = pto_path.read_text(encoding='utf-8').splitlines()
    out = []
    for line in lines:
        if line.startswith('i ') and 'n"' in line:
            m = re.search(r'n"([^"]+)"', line)
            if m:
                stem = Path(m.group(1)).stem
                try:
                    el_part, az_part = stem.split('_az')
                    el = int(el_part.replace('el', '').replace('n', '-'))
                    az = int(az_part)
                    line = re.sub(r'(?<=\s)p-?\d+', f'p{el}', line)
                    line = re.sub(r'(?<=\s)y-?\d+', f'y{az}', line)
                except Exception:
                    pass
        out.append(line)
    pto_path.write_text('\n'.join(out), encoding='utf-8')


def count_control_points(pto_path: Path) -> int:
    """Count control point lines (starting with 'c ') in a PTO file."""
    try:
        return sum(1 for line in pto_path.read_text(encoding='utf-8').splitlines() if line.startswith('c '))
    except Exception:
        return 0


def _cleanup_intermediates(images_dir: Path):
    """Remove pano????.tif tiles, pano_final.tif, and project.pto to free disk."""
    for pat in ("pano*.tif", "project.pto"):
        for p in images_dir.glob(pat):
            try:
                p.unlink()
            except Exception:
                pass


def stitch_images(
    session_id: str,
    images_dir: Path,
    output_path: Path,
    fov: float = 75.0,
    set_stage: Optional[Callable[[str], None]] = None,
):
    """
    Uses the Hugin command-line toolchain to stitch ultra-wide images.
    Returns (success_boolean, output_path_or_error_message).

    set_stage: optional callback(stage_name) called at each pipeline step
               so the caller can track progress externally (e.g. write to disk).
    """
    try:
        return _stitch_inner(session_id, images_dir, output_path, fov, set_stage)
    finally:
        _cleanup_intermediates(images_dir)


def _stitch_inner(session_id, images_dir, output_path, fov, set_stage):
    start_time = time.time()

    def t() -> str:
        return f"{time.time() - start_time:.1f}s"

    def stage(name: str):
        print(f"[{t()}] STAGE: {name}")
        if set_stage:
            set_stage(name)

    imgs = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in ('.jpg', '.jpeg', '.png')
    )
    if len(imgs) < 2:
        return False, f"Not enough images to stitch. Found {len(imgs)}."

    img_paths = [str(p.absolute()) for p in imgs]
    pto_file = "project.pto"

    # Auto-correct FOV if images are portrait (phone held wrong way)
    w = h = 0
    sample = cv2.imread(str(imgs[0]))
    if sample is not None:
        h, w = sample.shape[:2]
        if w < h:
            fov = math.degrees(2 * math.atan(math.tan(math.radians(fov / 2)) * w / h))

    print(f"[START] session={session_id} images={len(imgs)} res={w}x{h} fov={fov:.1f}")

    # ── Step 1: Generate PTO ──────────────────────────────────────────
    stage("loading")
    cmd1 = [_bin("pto_gen"), "-o", pto_file, f"--fov={fov}"] + img_paths
    if not run_hugin(cmd1, cwd=str(images_dir), discard_stdout=True):
        return False, "pto_gen failed"
    inject_angles_into_pto(images_dir / pto_file)

    # ── Step 2: Feature matching ──────────────────────────────────────
    stage("matching")
    cmd2 = [_bin("cpfind"), "--fullscale", "--multirow", "-o", pto_file, pto_file]
    if not run_hugin(cmd2, cwd=str(images_dir), discard_stdout=True):
        return False, "cpfind failed — no features found. Ensure images have enough texture/detail."

    cp_count = count_control_points(images_dir / pto_file)
    print(f"[{t()}] cpfind --multirow: {cp_count} control points")

    # If multirow found nothing, retry using prealigned positions from filename injection
    if cp_count == 0:
        inject_angles_into_pto(images_dir / pto_file)
        cmd2_alt = [_bin("cpfind"), "--fullscale", "--multirow", "--prealigned", "-o", pto_file, pto_file]
        run_hugin(cmd2_alt, cwd=str(images_dir), can_fail=True, discard_stdout=True)
        cp_count = count_control_points(images_dir / pto_file)
        print(f"[{t()}] cpfind --prealigned: {cp_count} control points")

    has_real_cps = cp_count >= 20

    # ── Step 3: Clean control points (only when we have plenty) ──────
    stage("optimizing")
    if has_real_cps:
        cmd3 = [_bin("cpclean"), "--max-distance=10", "-o", pto_file, pto_file]
        run_hugin(cmd3, cwd=str(images_dir), can_fail=True, discard_stdout=True)
        cp_after = count_control_points(images_dir / pto_file)
        print(f"[{t()}] cpclean: {cp_count} -> {cp_after} CPs")
        if cp_after < max(10, cp_count // 3):
            print(f"[{t()}] cpclean too aggressive — restoring with cpfind")
            inject_angles_into_pto(images_dir / pto_file)
            run_hugin(cmd2, cwd=str(images_dir), can_fail=True, discard_stdout=True)
    else:
        print(f"[{t()}] WARN: only {cp_count} CPs — relying on filename angle injection")

    # ── Step 4 & 5: Line detection + optimiser ───────────────────────
    if has_real_cps:
        cmd4 = [_bin("linefind"), "-o", pto_file, pto_file]
        run_hugin(cmd4, cwd=str(images_dir), can_fail=True, discard_stdout=True)

        cmd5 = [_bin("autooptimiser"), "-a", "-m", "-l", "-s", "-p", "-o", pto_file, pto_file]
        if not run_hugin(cmd5, cwd=str(images_dir), can_fail=False, discard_stdout=True):
            print(f"[{t()}] autooptimiser -p failed — retrying without -p")
            cmd5b = [_bin("autooptimiser"), "-a", "-m", "-l", "-s", "-o", pto_file, pto_file]
            if not run_hugin(cmd5b, cwd=str(images_dir), discard_stdout=True):
                return False, "autooptimiser failed — not enough overlapping control points to solve geometry."
        print(f"[{t()}] autooptimiser OK (full)")
    else:
        # No real CPs — skip linefind, only do photometric (-m -l -s).
        # DO NOT use -a or -p: they would corrupt the angle-injected positions.
        cmd5 = [_bin("autooptimiser"), "-m", "-l", "-s", "-o", pto_file, pto_file]
        run_hugin(cmd5, cwd=str(images_dir), can_fail=True, discard_stdout=True)
        print(f"[{t()}] autooptimiser OK (photometric only — angle injection preserved)")

    # ── Step 6: Canvas ───────────────────────────────────────────────
    # Scale canvas with image count; env var can always override
    n_imgs = len(imgs)
    if n_imgs >= 20:
        default_canvas = "6000x3000"
    elif n_imgs >= 12:
        default_canvas = "4000x2000"
    else:
        default_canvas = "3000x1500"
    canvas = os.environ.get("STITCH_CANVAS", default_canvas)
    cmd6 = [_bin("pano_modify"), "--projection=2", f"--canvas={canvas}", "-o", pto_file, pto_file]
    if not run_hugin(cmd6, cwd=str(images_dir), discard_stdout=True):
        return False, "pano_modify failed"

    # ── Step 7a: Remap images with nona ──────────────────────────────
    stage("stitching")
    print(f"[{t()}] canvas={canvas} — remapping with nona...")
    cmd_nona = [_bin("nona"), "-m", "TIFF_m", "-o", "pano", pto_file]
    if not run_hugin_streaming(cmd_nona, cwd=str(images_dir), timeout=600):
        return False, "nona failed — could not remap images"
    remapped = sorted(images_dir.glob("pano????.tif"))
    tile_mb = sum(p.stat().st_size for p in remapped) // (1024 * 1024)
    print(f"[{t()}] nona OK — {len(remapped)} tiles, {tile_mb} MB")

    # ── Step 7b: Blend with enblend ──────────────────────────────────
    stage("saving")
    if not remapped:
        return False, "nona produced no remapped tiles (pano????.tif not found)"
    cmd_enblend = [
        _bin("enblend"),
        "-v",
        "--primary-seam-generator=nft",  # Nearest-Feature-Transform: orders of magnitude faster than annealing
        "--fine-mask",                   # skip annealing seam optimization (~3 min per pair on hobby vCPU)
        "--compression=LZW",
        "-o", "pano_final.tif",
    ] + [p.name for p in remapped]
    if not run_hugin_streaming(cmd_enblend, cwd=str(images_dir), timeout=600):
        return False, "enblend failed — see logs for details"

    output_files = list(images_dir.glob("pano_final.tif"))
    if not output_files:
        output_files = sorted(images_dir.glob("pano*.tif"), key=lambda p: p.stat().st_size, reverse=True)
    if not output_files:
        return False, "No output TIFF found after stitching"

    target_output = max(output_files, key=lambda p: p.stat().st_size)
    tiff_mb = target_output.stat().st_size // (1024 * 1024)
    print(f"[{t()}] enblend OK — {tiff_mb} MB TIFF")

    try:
        from PIL import Image
        img = Image.open(str(target_output))
        img = img.convert("RGB")
        img.save(str(output_path), "JPEG", quality=92)
        jpeg_kb = output_path.stat().st_size // 1024
        print(f"[DONE] {t()} — JPEG {jpeg_kb} KB -> {output_path.name}")
        return True, str(output_path)
    except Exception as e:
        return False, f"Failed to save final JPEG: {e}"
