"""Render a stage built by build_scene.py.

    python3 render.py <scene.blend> <out folder> [first-last | frame,frame,...] [percent]

Writes <out>/frame_0001.png ... with Blender's Cycles (no graphics card needed). With every frame rendered and
ffmpeg installed, also <out>/../<scene>.mp4 (H.264, the scene's frame rate). In Blender itself: Render > Render
Animation does the same (frames go next to the .blend, in frames/).
"""
import os
import subprocess
import sys
import time

import bpy


def frames(arg, scene):
    if not arg:
        return list(range(scene.frame_start, scene.frame_end + 1))
    if "-" in arg:
        a, b = arg.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(v) for v in arg.split(",")]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    blend, out = os.path.abspath(argv[0]), os.path.abspath(argv[1])
    bpy.ops.wm.open_mainfile(filepath=blend)
    scene = bpy.context.scene
    todo = frames(argv[2] if len(argv) > 2 else "", scene)
    if len(argv) > 3:
        scene.render.resolution_percentage = int(argv[3])
    os.makedirs(out, exist_ok=True)
    started = time.time()
    for n, f in enumerate(todo, 1):
        scene.frame_set(f)
        scene.render.filepath = os.path.join(out, "frame_%04d.png" % f)
        bpy.ops.render.render(write_still=True)
        if n == 1 or n % 10 == 0 or n == len(todo):
            print("frame %d (%d of %d), %.1f s per frame" % (f, n, len(todo), (time.time() - started) / n), flush=True)
    whole = todo == list(range(scene.frame_start, scene.frame_end + 1))
    if whole:
        mp4 = os.path.join(os.path.dirname(out), os.path.splitext(os.path.basename(blend))[0] + ".mp4")
        try:
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(scene.render.fps), "-start_number",
                            str(scene.frame_start), "-i", os.path.join(out, "frame_%04d.png"), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "slow", mp4], check=True)
            print("video:", mp4)
        except (OSError, subprocess.CalledProcessError) as e:
            print("frames are ready; making the video failed:", e)


if __name__ == "__main__":
    main()
