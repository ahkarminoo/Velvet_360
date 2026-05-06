import subprocess
import os
import re
import math
import cv2
import time
import platform
from pathlib import Path
from typing import Callable, Optional

HUGIN_BIN = os.environ.get("HUGIN_BIN", "/usr/bin")


def _bin(name):
    suffix = ".exe" if platform.system() == "Windows" else ""
    return os.path.join(HUGIN_BIN, f"{name}{suffix}")


def run_hugin(cmd, cwd, can_fail=False, timeout=600, discard_stdout=False):
    """Run a Hugin command. Returns True on success. If can_fail=True, returns False instead of crashing."""
    print(f"Running: {' '.join(cmd)}")
    stdout_target = subprocess.DEVNULL if discard_stdout else subprocess.PIPE
    try:
        result = subprocess.run(
            cmd, cwd=cwd, check=True,
            stdout=stdout_target, stderr=subprocess.PIPE,
            timeout=timeout
        )
        if not discard_stdout and result.stdout:
            print(result.stdout.decode('utf-8', errors='ignore'))
        return True
    except subprocess.TimeoutExpired:
        print(f"[ERROR] {os.path.basename(cmd[0])} timed out after {timeout}s")
        return can_fail
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.decode('utf-8', errors='ignore') if e.stderr else ''
        stdout_msg = e.stdout.decode('utf-8', errors='ignore') if (e.stdout and not discard_stdout) else ''
        print(f"[ERROR] {os.path.basename(cmd[0])} exited {e.returncode}")
        if stderr_msg:
            print(f"  stderr: {stderr_msg[:2000]}")
        if stdout_msg:
            print(f"  stdout: {stdout_msg[:500]}")
        return can_fail
    except FileNotFoundError:
        print(f"[ERROR] Binary not found: {cmd[0]}. Check HUGIN_BIN={HUGIN_BIN}")
        return False


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
    def stage(name: str):
        print(f"[STAGE] {name}")
        if set_stage:
            set_stage(name)

    imgs = sorted(images_dir.glob("*.jpg"))
    if len(imgs) < 2:
        return False, f"Not enough images to stitch. Found {len(imgs)}."

    img_paths = [str(p.absolute()) for p in imgs]
    pto_file = "project.pto"

    # Auto-correct FOV if images are portrait (phone held wrong way)
    sample = cv2.imread(str(imgs[0]))
    if sample is not None:
        h, w = sample.shape[:2]
        if w < h:
            fov = math.degrees(2 * math.atan(math.tan(math.radians(fov / 2)) * w / h))
            print(f"Portrait images detected ({w}x{h}) — adjusted FOV to {fov:.1f}°")

    print(f"--- Starting Hugin Stitching Pipeline for {session_id} ({len(imgs)} images) ---")
    start_time = time.time()

    # ── Step 1: Generate PTO ──────────────────────────────────────────
    stage("loading")
    cmd1 = [_bin("pto_gen"), "-o", pto_file, f"--fov={fov}"] + img_paths
    if not run_hugin(cmd1, cwd=str(images_dir)):
        return False, "pto_gen failed"

    inject_angles_into_pto(images_dir / pto_file)

    # ── Step 2: Feature matching ──────────────────────────────────────
    stage("matching")
    cmd2 = [_bin("cpfind"), "--fullscale", "--multirow", "-o", pto_file, pto_file]
    if not run_hugin(cmd2, cwd=str(images_dir)):
        return False, "cpfind failed — no features found. Ensure images have enough texture/detail."

    # ── Step 3: Clean control points (smart — never fatal) ────────────
    stage("optimizing")
    cp_before = count_control_points(images_dir / pto_file)
    print(f"Control points found: {cp_before}")
    if cp_before >= 20:
        cmd3 = [_bin("cpclean"), "--max-distance=10", "-o", pto_file, pto_file]
        run_hugin(cmd3, cwd=str(images_dir), can_fail=True)
        cp_after = count_control_points(images_dir / pto_file)
        print(f"Control points after cpclean: {cp_after}")
        if cp_after < max(10, cp_before // 3):
            print(f"[WARN] cpclean too aggressive ({cp_before} -> {cp_after}). Restoring with cpfind.")
            inject_angles_into_pto(images_dir / pto_file)
            run_hugin(cmd2, cwd=str(images_dir), can_fail=True)
    else:
        print(f"[WARN] Only {cp_before} control points — skipping cpclean.")

    # ── Step 4: Line detection (optional) ────────────────────────────
    cmd4 = [_bin("linefind"), "-o", pto_file, pto_file]
    run_hugin(cmd4, cwd=str(images_dir), can_fail=True)

    # ── Step 5: Optimize camera parameters ───────────────────────────
    cmd5 = [_bin("autooptimiser"), "-a", "-m", "-l", "-s", "-p", "-o", pto_file, pto_file]
    if not run_hugin(cmd5, cwd=str(images_dir), can_fail=False):
        print("[WARN] Full autooptimiser failed — retrying without position flag (-p).")
        cmd5b = [_bin("autooptimiser"), "-a", "-m", "-l", "-s", "-o", pto_file, pto_file]
        if not run_hugin(cmd5b, cwd=str(images_dir)):
            return False, "autooptimiser failed — not enough overlapping control points to solve geometry."

    # ── Step 6: Canvas ───────────────────────────────────────────────
    canvas = os.environ.get("STITCH_CANVAS", "2000x1000")
    cmd6 = [_bin("pano_modify"), f"--canvas={canvas}", "-o", pto_file, pto_file]
    if not run_hugin(cmd6, cwd=str(images_dir)):
        return False, "pano_modify failed"

    # ── Step 7a: Remap images with nona ──────────────────────────────
    stage("stitching")
    print(f"[INFO] Canvas: {canvas} — remapping images with nona...")
    cmd_nona = [_bin("nona"), "-m", "TIFF_m", "-o", "pano", pto_file]
    if not run_hugin(cmd_nona, cwd=str(images_dir), timeout=600, discard_stdout=True):
        return False, "nona failed — could not remap images"

    # ── Step 7b: Blend with enblend ──────────────────────────────────
    stage("saving")
    remapped = sorted(images_dir.glob("pano????.tif"))
    if not remapped:
        return False, "nona produced no remapped tiles (pano????.tif not found)"
    print(f"[INFO] Blending {len(remapped)} tiles with enblend...")
    cmd_enblend = [
        _bin("enblend"),
        "--compression=LZW",
        "-o", "pano_final.tif",
    ] + [p.name for p in remapped]
    if not run_hugin(cmd_enblend, cwd=str(images_dir), timeout=600, discard_stdout=True):
        return False, "enblend failed — see logs for details"

    print(f"--- Hugin Pipeline completed in {time.time() - start_time:.1f} seconds ---")

    output_files = list(images_dir.glob("pano_final.tif"))
    if not output_files:
        # fallback: grab any large pano tif
        output_files = sorted(images_dir.glob("pano*.tif"), key=lambda p: p.stat().st_size, reverse=True)
    if not output_files:
        return False, "No output TIFF found after stitching"

    target_output = max(output_files, key=lambda p: p.stat().st_size)
    print(f"Output: {target_output} ({target_output.stat().st_size // 1024} KB)")

    try:
        from PIL import Image
        img = Image.open(str(target_output))
        img = img.convert("RGB")
        img.save(str(output_path), "JPEG", quality=92)
        return True, str(output_path)
    except Exception as e:
        return False, f"Failed to save final JPEG: {e}"
