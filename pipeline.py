import subprocess
import os
import re
import math
import cv2
import time
import platform
from pathlib import Path

HUGIN_BIN = os.environ.get("HUGIN_BIN", "/usr/bin")

def _bin(name):
    suffix = ".exe" if platform.system() == "Windows" else ""
    return os.path.join(HUGIN_BIN, f"{name}{suffix}")

def run_hugin(cmd, cwd):
    print(f"Running: {' '.join(cmd)}")
    try:
        # Run with check=True to raise exception on failure
        result = subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {cmd[0]}:")
        print(e.stderr.decode('utf-8', errors='ignore'))
        return False
    except FileNotFoundError:
        print(f"Could not find {cmd[0]}. Make sure Hugin is installed at {HUGIN_BIN}.")
        return False

def inject_angles_into_pto(pto_path: Path):
    """Set each image's yaw and pitch in the PTO from its el/az filename."""
    lines = pto_path.read_text(encoding='utf-8').splitlines()
    out = []
    for line in lines:
        if line.startswith('i ') and 'n"' in line:
            m = re.search(r'n"([^"]+)"', line)
            if m:
                stem = Path(m.group(1)).stem  # e.g. "el50_az060" or "eln45_az030"
                try:
                    el_part, az_part = stem.split('_az')
                    el = int(el_part.replace('el', '').replace('n', '-'))
                    az = int(az_part)
                    # Replace standalone p (pitch=elevation) and y (yaw=azimuth) tokens
                    line = re.sub(r'(?<=\s)p-?\d+', f'p{el}', line)
                    line = re.sub(r'(?<=\s)y-?\d+', f'y{az}', line)
                except Exception:
                    pass
        out.append(line)
    pto_path.write_text('\n'.join(out), encoding='utf-8')


def stitch_images(session_id: str, images_dir: Path, output_path: Path, fov: float = 75.0):
    """
    Uses the Hugin command-line toolchain to stitch ultra-wide images.
    Returns (success_boolean, output_path_or_error_message).
    """
    imgs = sorted(images_dir.glob("*.jpg"))
    if len(imgs) < 2:
        return False, f"Not enough images to stitch. Found {len(imgs)}."
        
    # Full absolute paths for Hugin
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
    
    # 1. Generate PTO project
    cmd1 = [_bin("pto_gen"), "-o", pto_file, f"--fov={fov}"] + img_paths
    if not run_hugin(cmd1, cwd=str(images_dir)): return False, "pto_gen failed"

    # 1b. Inject known angles from filenames so cpfind has correct starting positions
    inject_angles_into_pto(images_dir / pto_file)

    # 2. Find control points at full resolution for sharper seams
    cmd2 = [_bin("cpfind"), "--multirow", "--fullscale", "-o", pto_file, pto_file]
    if not run_hugin(cmd2, cwd=str(images_dir)): return False, "cpfind failed"

    # 3. Clean bad control points
    cmd3 = [_bin("cpclean"), "-o", pto_file, pto_file]
    if not run_hugin(cmd3, cwd=str(images_dir)): return False, "cpclean failed"

    # 4. Find vertical/horizontal lines
    cmd4 = [_bin("linefind"), "-o", pto_file, pto_file]
    if not run_hugin(cmd4, cwd=str(images_dir)): return False, "linefind failed"

    # 5. Optimize camera parameters
    cmd5 = [_bin("autooptimiser"), "-a", "-m", "-l", "-s", "-p", "-o", pto_file, pto_file]
    if not run_hugin(cmd5, cwd=str(images_dir)): return False, "autooptimiser failed"

    # 6. Calculate optimal canvas (no auto-crop — it can produce an empty mask)
    cmd6 = [_bin("pano_modify"), "--canvas=AUTO", "-o", pto_file, pto_file]
    if not run_hugin(cmd6, cwd=str(images_dir)): return False, "pano_modify failed"

    # 7. Execute stitching (nona + enblend)
    cmd7 = [_bin("hugin_executor"), "--stitching", "--prefix=pano", pto_file]
    if not run_hugin(cmd7, cwd=str(images_dir)): return False, "hugin_executor failed"
    
    print(f"--- Hugin Pipeline completed in {time.time() - start_time:.1f} seconds ---")
    
    # Locate output file. hugin_executor typically generates prefix.tif or prefix.jpg
    # It might append _equirectangular or similar. Let's find it.
    output_files = list(images_dir.glob("pano*.tif")) + list(images_dir.glob("pano*.jpg"))
    if not output_files:
        return False, "Failed to locate Hugin output file (pano*.tif or pano*.jpg)"
        
    # Take the largest file if multiple exist (to avoid taking a small preview)
    target_output = max(output_files, key=lambda p: p.stat().st_size)
    
    try:
        # Convert TIF to JPG for the web viewer if needed, and move to final output path
        img = cv2.imread(str(target_output))
        if img is None:
            return False, f"OpenCV failed to read Hugin output: {target_output}"
            
        cv2.imwrite(str(output_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        return True, str(output_path)
    except Exception as e:
        return False, f"Failed to save final JPEG: {str(e)}"
