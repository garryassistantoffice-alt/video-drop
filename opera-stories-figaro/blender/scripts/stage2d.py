"""The same painted-cutout stage as build_scene.py makes in Blender, drawn directly in 2D (about a second a frame
instead of about twenty), for rendering whole episodes.

    python3 stage2d.py <build folder> <out folder> [first-last] [--check]

With an orthographic camera nothing in the stage needs real 3D: every card stays parallel to the picture, so
- the camera's few degrees of turn only slide each card sideways in proportion to its depth (parallax);
- the window light's shadows are each card's own outline, moved right and down in proportion to the gap between the
  cards, softened by the light's size, darkening what lies behind by `shadow`;
- bones bend a card the way Blender's armature does (each bone turns and moves its region; the pull fades out over
  `soft` pixels; a child bone follows its parent).
It reads the same layers.json and animation as build_scene.py and keyframes are eased the same way, so a frame here
and the same frame from Blender should match (compare them with --check, which also writes side-by-side pictures
when Blender's frames are in <out>/../<scene>_frames).
"""
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np

W_OUT, H_OUT = 1080, 1920


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def keyed(keys, t, size=None):
    """A keyframed value at time t: eased between keys (Blender's Bezier keys with auto-clamped handles ease in and
    out at every key, which a smooth step between neighbours follows closely)."""
    if not keys:
        return np.zeros(size) if size else 0.0
    if t <= keys[0][0]:
        return np.asarray(keys[0][1], float) if size else float(keys[0][1])
    if t >= keys[-1][0]:
        return np.asarray(keys[-1][1], float) if size else float(keys[-1][1])
    for (ta, va), (tb, vb) in zip(keys, keys[1:]):
        if ta <= t <= tb:
            s = float(smooth((t - ta) / max(tb - ta, 1e-6)))
            va = np.asarray(va, float) if size else float(va)
            vb = np.asarray(vb, float) if size else float(vb)
            return va + (vb - va) * s
    return np.asarray(keys[-1][1], float) if size else float(keys[-1][1])


def region_weight(shape, poly, soft, box):
    """Weight of one bone's region over a box of the picture: 1 inside, fading to 0 over `soft` pixels outside."""
    x0, y0, x1, y1 = box
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    pts = np.round(np.array(poly, np.float64) - [x0, y0]).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)
    dist = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
    return (1.0 - smooth(dist / max(soft, 1e-6))).astype(np.float32)


class Card:
    def __init__(self, build, layer, w, h):
        self.name = layer["name"]
        self.depth = float(layer["depth"])
        rgba = cv2.imread(os.path.join(build, layer["file"]), cv2.IMREAD_UNCHANGED)
        if rgba.shape[2] == 3:
            rgba = np.dstack([rgba, np.full(rgba.shape[:2], 255, np.uint8)])
        self.rgba = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255.0
        caster = os.path.join(build, self.name + "_caster.png")
        self.caster = (cv2.imread(caster, cv2.IMREAD_UNCHANGED)[..., 3].astype(np.float32) / 255.0
                       if os.path.exists(caster) else self.rgba[..., 3].copy())
        self.rig = layer.get("rig")
        if self.rig:
            ys, xs = np.nonzero(self.rgba[..., 3] > 0.004)
            pad = 40
            self.box = (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad), min(w, int(xs.max()) + pad),
                        min(h, int(ys.max()) + pad))
            self.weights = {}
            left = np.ones((self.box[3] - self.box[1], self.box[2] - self.box[0]), np.float32)
            for region in self.rig["weights"]:
                wgt = region_weight((h, w), region["region"], region.get("soft", 15), self.box) * left
                self.weights[region["bone"]] = self.weights.get(region["bone"], 0) + wgt
                left = left - wgt
            root = self.rig["bones"][0]["name"]
            self.weights[root] = self.weights.get(root, 0) + left
            self.bones = {b["name"]: b for b in self.rig["bones"]}

    def bone_matrices(self, t, anim):
        """Each bone's 2D transform (2x3, picture pixels) at time t, parents first."""
        mats = {}
        for b in self.rig["bones"]:
            keys = anim.get(b["name"], {})
            deg = keyed(keys.get("turn", []), t)
            dx, dy = keyed(keys.get("shift", []), t, 2) if keys.get("shift") else (0.0, 0.0)
            hx, hy = b["head"]
            th = math.radians(deg)   # anticlockwise as seen by the camera: picture y points down
            c, s = math.cos(th), math.sin(th)
            # turn about the bone's head, then shift (pixels, x right, y down)
            local = np.array([[c, s, hx - c * hx - s * hy + dx], [-s, c, hy + s * hx - c * hy + dy], [0, 0, 1.0]])
            parent = b.get("parent")
            mats[b["name"]] = (mats[parent] @ local) if parent else local
        return mats

    def bent(self, t, anim):
        """The card's RGBA and its shadow outline after the bones, as full-picture arrays."""
        if not self.rig or not anim:
            return self.rgba, self.caster
        x0, y0, x1, y1 = self.box
        mats = self.bone_matrices(t, anim)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        # where each pixel of the card goes (linear blend of the bones, as Blender's armature does)...
        fx = np.zeros_like(xx)
        fy = np.zeros_like(yy)
        for name, wgt in self.weights.items():
            m = mats[name]
            fx += wgt * (m[0, 0] * xx + m[0, 1] * yy + m[0, 2])
            fy += wgt * (m[1, 0] * xx + m[1, 1] * yy + m[1, 2])
        dx, dy = (fx - xx).astype(np.float32), (fy - yy).astype(np.float32)
        # ...and where each output pixel comes from: the movement turned around by a few fixed-point steps
        sx, sy = xx.copy(), yy.copy()
        for _ in range(6):
            ddx = cv2.remap(dx, sx - x0, sy - y0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            ddy = cv2.remap(dy, sx - x0, sy - y0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            sx, sy = (xx - ddx).astype(np.float32), (yy - ddy).astype(np.float32)
        rgba = self.rgba.copy()
        rgba[y0:y1, x0:x1] = cv2.remap(self.rgba, sx, sy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        caster = self.caster.copy()
        caster[y0:y1, x0:x1] = cv2.remap(self.caster, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return np.clip(rgba, 0, 1), np.clip(caster, 0, 1)


class Stage:
    def __init__(self, build):
        self.info = json.load(open(os.path.join(build, "layers.json")))
        self.w, self.h = self.info["size"]
        self.cards = [Card(build, l, self.w, self.h) for l in self.info["layers"]]
        look = self.info.get("light", {})
        self.shadow = float(look.get("shadow", 0.35))
        self.offset = look.get("offset", [0.10, 0.08])
        self.soft = math.radians(float(look.get("softness_degrees", 6)))
        anim = self.info.get("animation", {})
        self.fps = int(anim.get("fps", 30))
        self.frames = max(1, int(round(float(anim.get("seconds", 6.0)) * self.fps)))
        self.camera = anim.get("camera", [[0, [0.5, 0.5, 1.05], 0]])
        self.bones = anim.get("bones", {})
        self.pivot = float(self.info.get("stage", {}).get("pivot_depth", 1.6))

    def frame(self, n):
        """Frame n (1-based, like Blender's) as RGB uint8, 1080 x 1920."""
        t = (n - 1) / self.fps
        cx, cy, _ = keyed([[k[0], k[1]] for k in self.camera], t, 3)
        # Blender eases the camera's orthographic size (the visible height), not the zoom itself
        vis_h = keyed([[k[0], self.h / k[1][2]] for k in self.camera], t)
        turn = math.radians(keyed([[k[0], k[2]] for k in self.camera], t))
        bent = [c.bent(t, self.bones.get(c.name)) for c in self.cards]
        # the visible window (picture pixels), exactly as the orthographic camera frames it
        vis_w = vis_h * W_OUT / H_OUT
        scale = W_OUT / vis_w
        left = cx * self.w - vis_w / 2
        top = cy * self.h - vis_h / 2
        out = np.zeros((H_OUT, W_OUT, 3), np.float32)
        for i, card in enumerate(self.cards):
            rgba, _ = bent[i]
            # light reaching this card: the cards in front, each moved by its gap and softened by the light's size
            lit = np.ones((self.h, self.w), np.float32)
            for j in range(i + 1, len(self.cards)):
                gap = (self.cards[j].depth - card.depth) * 100.0
                if gap <= 0:
                    continue
                m = np.float32([[1, 0, gap * self.offset[0]], [0, 1, gap * self.offset[1]]])
                shade = cv2.warpAffine(bent[j][1], m, (self.w, self.h), flags=cv2.INTER_LINEAR, borderValue=0)
                sigma = max(0.5, gap * math.tan(self.soft) / 2.0)
                shade = cv2.GaussianBlur(shade, (0, 0), sigma)
                lit *= 1.0 - shade
            colour = rgba[..., :3] * ((1.0 - self.shadow) + self.shadow * lit)[..., None]
            # the camera's turn slides each card by its depth: cards behind the pivot move the way the camera moves,
            # cards in front of it the other way
            slide = (self.pivot - card.depth) * 100.0 * math.sin(turn)
            # Blender treats a picture pixel as the square [x, x+1]; OpenCV puts its centre at x: half a pixel apart
            m = np.float32([[scale, 0, -(left - slide) * scale + 0.5 * scale - 0.5],
                            [0, scale, -top * scale + 0.5 * scale - 0.5]])
            layer = cv2.warpAffine(np.dstack([colour, rgba[..., 3]]), m, (W_OUT, H_OUT), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE if i == 0 else cv2.BORDER_CONSTANT, borderValue=0)
            a = layer[..., 3:4]
            out = out * (1 - a) + layer[..., :3] * a
        return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    build, out = argv[0], argv[1]
    stage = Stage(build)
    first, last = 1, stage.frames
    if len(argv) > 2:
        a, b = argv[2].split("-") if "-" in argv[2] else (argv[2], argv[2])
        first, last = int(a), int(b)
    os.makedirs(out, exist_ok=True)
    import time
    started = time.time()
    for n in range(first, last + 1):
        cv2.imwrite(os.path.join(out, "frame_%04d.png" % n), cv2.cvtColor(stage.frame(n), cv2.COLOR_RGB2BGR))
    took = (time.time() - started) / (last - first + 1)
    print("frames %d-%d, %.2f s per frame" % (first, last, took))
    if first == 1 and last == stage.frames:
        mp4 = out.rstrip("/") + ".mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(stage.fps), "-i",
                        os.path.join(out, "frame_%04d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
                        "-preset", "slow", mp4], check=True)
        print("video:", mp4)


if __name__ == "__main__":
    main()
