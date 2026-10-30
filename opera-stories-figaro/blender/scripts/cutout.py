"""Layers for a painted-cutout stage, made from one existing painting (nothing is repainted).

    python3 cutout.py scenes/<scene>.json [out folder]

The scene file names the painting and its layers, back to front. Every layer is the painting itself, cut along an
outline; nothing moves out of place, so the layers line up pixel for pixel. What a layer hides is rebuilt on the layer
behind it, but only where it can ever be seen:
- behind a moving part (a hand that turns, a head that nods), the whole area the part covers;
- behind a still layer, a band along its edge ("band" pixels), which the camera's slight turn can uncover.
The rest of what a layer hides is never seen, so the painting's own pixels stay there.

Writes <out>/<layer>.png (the back layer as RGB, the others as RGBA) and <out>/layers.json (outlines in picture pixels,
used by build_scene.py in Blender), plus <out>/check_<layer>.jpg pictures for checking the cuts by eye.

Outlines: "poly" is traced by hand on a grid (short1/grid.py). With "snap" pixels > 0 the edge is refined
inside a band of that width around the outline (OpenCV GrabCut), which follows hair, lace and sleeves better than a
hand-traced line; with 0 the outline is used as it is (right for skin against skin, where colours cannot tell the edge).
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def polygon_mask(shape, poly):
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(np.array(poly, np.float64)).astype(np.int32)], 255)
    return mask


def snap(image, mask, band):
    """Refine a hand-traced outline: GrabCut decides only inside a band of +-band pixels around the line."""
    if band <= 0:
        return mask
    k = np.ones((2 * band + 1, 2 * band + 1), np.uint8)
    inner, outer = cv2.erode(mask, k), cv2.dilate(mask, k)
    gc = np.full(mask.shape, cv2.GC_BGD, np.uint8)
    gc[outer > 0] = cv2.GC_PR_BGD
    gc[mask > 0] = cv2.GC_PR_FGD
    gc[inner > 0] = cv2.GC_FGD
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(image, gc, None, bg, fg, 5, cv2.GC_INIT_WITH_MASK)
    out = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # keep the largest piece and fill small holes (GrabCut leaves specks in lace)
    n, lab, st, _ = cv2.connectedComponentsWithStats(out, 8)
    if n > 1:
        out = np.where(lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])), 255, 0).astype(np.uint8)
    holes = cv2.bitwise_not(out)
    n, lab, st, _ = cv2.connectedComponentsWithStats(holes, 4)
    h, w = out.shape
    for i in range(1, n):
        x, y, ww, hh, area = st[i]
        if area < 2500 and x > 0 and y > 0 and x + ww < w and y + hh < h:
            out[lab == i] = 255
    return out


def tidy(mask, holes=3000):
    """The largest piece of a mask, its holes up to `holes` pixels filled (a sash or an apron fold the model took
    for something else; a real gap between an arm and a body is larger)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        mask = np.where(lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])), 255, 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(mask), 4)
    h, w = mask.shape
    for i in range(1, n):
        x, y, ww, hh, area = st[i]
        if area < holes and x > 0 and y > 0 and x + ww < w and y + hh < h:
            mask[lab == i] = 255
    return mask


def soft_edge(mask, px):
    """A cut edge as soft as the painting's own edges: about px pixels wide, centred on the line."""
    if px <= 0:
        return mask
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), px / 2.0), 0, 255).astype(np.uint8)


def rebuild(image, region):
    """Paint the region from its surroundings. Two passes: Telea fills it, then the fill is blended with a smoother
    Navier-Stokes fill so long thin regions (a finger, a band along an edge) do not streak."""
    if not region.any():
        return image.copy()
    telea = cv2.inpaint(image, region, 9, cv2.INPAINT_TELEA)
    ns = cv2.inpaint(image, region, 9, cv2.INPAINT_NS)
    mix = cv2.addWeighted(telea, 0.5, ns, 0.5, 0)
    out = image.copy()
    out[region > 0] = mix[region > 0]
    return out


def moving_area(shape, layer, mask):
    """The part of a moving layer that moves: its "moving" polygon, or the whole layer."""
    if not layer.get("moving"):
        return mask
    return cv2.bitwise_and(mask, polygon_mask(shape, layer["moving"]))


def band_inside(mask, width):
    """The part of a mask within `width` pixels of its edge (what a slight turn of the camera can uncover)."""
    k = np.ones((2 * width + 1, 2 * width + 1), np.uint8)
    return cv2.subtract(mask, cv2.erode(mask, k))


def make(spec_path, out):
    spec = json.load(open(spec_path))
    root = os.path.dirname(os.path.abspath(spec_path))
    painting = os.path.normpath(os.path.join(root, spec["painting"]))
    image = cv2.imread(painting, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit("cannot read the painting %s" % painting)
    h, w = image.shape[:2]
    if [w, h] != spec["size"]:
        raise SystemExit("the painting is %dx%d, the scene expects %dx%d" % (w, h, spec["size"][0], spec["size"][1]))
    os.makedirs(out, exist_ok=True)
    layers = spec["layers"]
    masks = {}
    sam = None
    for layer in layers[1:]:
        if "sam" in layer:
            # Segment Anything from a box and a few points; a traced "poly" then only limits it (where a forearm is
            # cut across, say), widened by "limit" pixels so the model's own edge is kept
            if sam is None:
                import segment
                sam = segment.Sam()
            q = layer["sam"]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cut, _ = sam.cut(rgb, q.get("box"), [tuple(p) for p in q.get("in", [])], [tuple(p) for p in q.get("out", [])])
            m = cut.astype(np.uint8) * 255
            if "poly" in layer:
                k = 2 * int(layer.get("limit", 6)) + 1
                m = cv2.bitwise_and(m, cv2.dilate(polygon_mask(image.shape, layer["poly"]), np.ones((k, k), np.uint8)))
            m = tidy(m, int(layer.get("holes", 3000)))
        else:
            m = snap(image, polygon_mask(image.shape, layer["poly"]), int(layer.get("snap", 0)))
        for name in layer.get("plus", []):   # other outlines that belong to this layer (a table, a prop)
            m = cv2.bitwise_or(m, polygon_mask(image.shape, spec["extra"][name]))
        if "sam" in layer or layer.get("plus"):
            m = tidy(m, int(layer.get("holes", 3000)))
        masks[layer["name"]] = m
    # a part on top of a figure (a hand on Susanna) belongs to that figure too: the figure's own cut must cover
    # it, or the room would show through where the part moves away
    for layer in layers[1:]:
        over = layer.get("over")
        if over and over in masks:
            masks[over] = tidy(cv2.bitwise_or(masks[over], cv2.dilate(masks[layer["name"]], np.ones((7, 7), np.uint8))), 200)
    # what each layer shows: the painting, rebuilt where a layer in front of it moves or has an edge
    info = {"painting": os.path.basename(painting), "size": [w, h], "layers": []}
    for i, layer in enumerate(layers):
        colour = image.copy()
        hide = np.zeros((h, w), np.uint8)
        for front in layers[i + 1:]:
            # a front layer hides the layer it sits on ("over"; the back layer unless named)
            if front.get("over", layers[0]["name"]) != layer["name"]:
                continue
            fm = masks[front["name"]]
            if front.get("moves"):
                # the area the moving part covers, a little wider for its soft edge; "moving" limits it to the part
                # that really moves (a forearm whose far end stays still needs nothing rebuilt under that end)
                moving = moving_area(image.shape, front, fm)
                hide = cv2.bitwise_or(hide, cv2.dilate(moving, np.ones((5, 5), np.uint8)))
            else:
                hide = cv2.bitwise_or(hide, band_inside(fm, int(front.get("band", 12))))
        if i > 0:   # a layer never needs what lies outside its own outline
            own = cv2.dilate(masks[layer["name"]], np.ones((9, 9), np.uint8))
            hide = cv2.bitwise_and(hide, own)
        colour = rebuild(colour, hide)
        rgb = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
        if i == 0:
            Image.fromarray(rgb).save(os.path.join(out, layer["name"] + ".png"))
            alpha = None
        else:
            alpha = soft_edge(masks[layer["name"]], float(layer.get("edge", 1.5)))
            Image.fromarray(np.dstack([rgb, alpha])).save(os.path.join(out, layer["name"] + ".png"))
            # the card's thickness (build_scene.py darkens it): the same cut 2 pixels smaller, so from the front
            # it hides completely behind the card and never shows through the soft edge
            inner = soft_edge(cv2.erode(masks[layer["name"]], np.ones((5, 5), np.uint8)), 1.0)
            Image.fromarray(np.dstack([rgb, inner])).save(os.path.join(out, layer["name"] + "_edge.png"))
            if layer.get("moves"):
                # the shadow of a moving part: only what moves casts it, so the still end of the part (a forearm cut
                # across) throws no shadow line onto the layer below
                caster = soft_edge(moving_area(image.shape, layer, masks[layer["name"]]), 3.0)
                Image.fromarray(np.dstack([rgb, caster])).save(os.path.join(out, layer["name"] + "_caster.png"))
        check(rgb, alpha, hide, os.path.join(out, "check_" + layer["name"] + ".jpg"))
        entry = {k: v for k, v in layer.items() if k not in ("plus",)}
        entry["file"] = layer["name"] + ".png"
        if alpha is not None:
            cnts, _ = cv2.findContours((alpha > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            entry["outline"] = [c[:, 0, :].tolist() for c in cnts if cv2.contourArea(c) > 50]
        entry["rebuilt_pixels"] = int((hide > 0).sum())
        info["layers"].append(entry)
    for key in ("camera", "light", "animation", "stage", "render"):
        if key in spec:
            info[key] = spec[key]
    json.dump(info, open(os.path.join(out, "layers.json"), "w"), indent=1)
    print("layers written to", out, ":", ", ".join("%s (%d rebuilt px)" % (e["name"], e["rebuilt_pixels"]) for e in info["layers"]))


def check(rgb, alpha, rebuilt, path):
    """For the eye: the layer on grey (rebuilt pixels tinted blue) beside its edge on a dark copy."""
    view = rgb.astype(np.float32)
    if rebuilt is not None and rebuilt.any():
        tint = np.array([90, 140, 255], np.float32)
        view[rebuilt > 0] = view[rebuilt > 0] * 0.6 + tint * 0.4
    if alpha is not None:
        a = alpha.astype(np.float32)[..., None] / 255.0
        view = view * a + 128 * (1 - a)
    im = Image.fromarray(np.clip(view, 0, 255).astype(np.uint8))
    im.thumbnail((700, 1244))
    im.save(path, quality=85)


if __name__ == "__main__":
    spec_path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(spec_path)), "..", "build",
                                                              os.path.splitext(os.path.basename(spec_path))[0])
    make(spec_path, os.path.normpath(out))
