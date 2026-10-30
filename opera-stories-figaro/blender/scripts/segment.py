"""Cut a person, a hand or a prop out of a painting with Segment Anything (Meta's SAM, ViT-B), from a box and a few
clicks, instead of tracing every curl by hand.

    python3 segment.py <painting> <out mask.png> --box x0 y0 x1 y1 [--in x,y ...] [--out x,y ...]

--box: a rectangle around the thing, in picture pixels; --in: points on it; --out: points that are not part of it
(the next person, the wall between two curls). Writes a black-and-white mask the size of the painting and a
<mask>_check.jpg with the outline drawn on the painting. The painting's embedding is cached next to the models
(<models>/cache/), so further cuts of the same painting take a second.

The model files (vit_b-encoder-quant.onnx, vit_b-decoder-quant.onnx, about 110 MB, from the rembg project's GitHub
releases) are looked for in GARRY_SAM_MODELS or ./models. Needs: numpy, opencv-python, onnxruntime, Pillow.
"""
import argparse
import hashlib
import os

import cv2
import numpy as np

SIZE = 1024
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


def models_dir():
    return os.environ.get("GARRY_SAM_MODELS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))


class Sam:
    def __init__(self, folder=None):
        import onnxruntime as ort
        folder = folder or models_dir()
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.folder = folder
        self.encoder = ort.InferenceSession(os.path.join(folder, "vit_b-encoder-quant.onnx"), opts,
                                            providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(os.path.join(folder, "vit_b-decoder-quant.onnx"), opts,
                                            providers=["CPUExecutionProvider"])

    def embed(self, rgb):
        """The painting seen by SAM: longest side 1024, normalised, padded. Cached by the picture's content."""
        key = hashlib.sha1(rgb.tobytes()).hexdigest()[:16]
        cache = os.path.join(self.folder, "cache", key + ".npy")
        h, w = rgb.shape[:2]
        scale = SIZE / max(h, w)
        if os.path.exists(cache):
            return np.load(cache), scale, (h, w)
        small = cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        x = (small.astype(np.float32) - MEAN) / STD
        pad = np.zeros((SIZE, SIZE, 3), np.float32)
        pad[:x.shape[0], :x.shape[1]] = x
        emb = self.encoder.run(None, {"x": pad.transpose(2, 0, 1)[None]})[0]
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, emb)
        return emb, scale, (h, w)

    def cut(self, rgb, box=None, inside=(), outside=()):
        emb, scale, (h, w) = self.embed(rgb)
        coords, labels = [], []
        for p in inside:
            coords.append(p); labels.append(1)
        for p in outside:
            coords.append(p); labels.append(0)
        if box is not None:
            coords += [box[:2], box[2:]]; labels += [2, 3]
        else:
            coords.append((0, 0)); labels.append(-1)   # no box: SAM's padding point
        pts = np.array(coords, np.float32)[None] * scale
        _, scores, low = self.decoder.run(None, {
            "image_embeddings": emb, "point_coords": pts, "point_labels": np.array(labels, np.float32)[None],
            "mask_input": np.zeros((1, 1, 256, 256), np.float32), "has_mask_input": np.zeros(1, np.float32),
            "orig_im_size": np.array([h, w], np.float32)})
        best = int(np.argmax(scores[0]))
        # this model file's own full-size "masks" output is misplaced (tested 2026-10-30), so the mask is enlarged
        # here from its 256x256 one, which covers the padded 1024x1024 input: up, cut the padding off, to full size
        up = cv2.resize(low[0, best], (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        up = up[:int(round(h * scale)), :int(round(w * scale))]
        full = cv2.resize(up, (w, h), interpolation=cv2.INTER_LINEAR)
        return full > 0.0, float(scores[0, best])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("painting")
    ap.add_argument("out")
    ap.add_argument("--box", nargs=4, type=float)
    ap.add_argument("--in", dest="inside", nargs="*", default=[])
    ap.add_argument("--out", dest="outside", nargs="*", default=[])
    a = ap.parse_args()
    rgb = cv2.cvtColor(cv2.imread(a.painting), cv2.COLOR_BGR2RGB)
    pt = lambda s: tuple(float(v) for v in s.split(","))
    mask, score = Sam().cut(rgb, a.box, [pt(p) for p in a.inside], [pt(p) for p in a.outside])
    cv2.imwrite(a.out, (mask * 255).astype(np.uint8))
    view = (rgb * 0.45).astype(np.uint8)
    view[mask] = rgb[mask]
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(view, cnts, -1, (255, 40, 160), 2)
    if a.box:
        x0, y0, x1, y1 = [int(v) for v in a.box]
        cv2.rectangle(view, (x0, y0), (x1, y1), (255, 230, 0), 1)
    cv2.imwrite(os.path.splitext(a.out)[0] + "_check.jpg", cv2.cvtColor(view, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("mask %s: %d px, model's confidence %.2f" % (a.out, int(mask.sum()), score))


if __name__ == "__main__":
    main()
