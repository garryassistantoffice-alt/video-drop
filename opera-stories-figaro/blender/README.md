# Susanna's finger: the first painted cutout stage

**This folder**
- `Susanna_finger.blend`: the stage, ready to open in Blender 4.5 (the paintings are packed inside).
- `renders/Susanna_finger.mp4`: the 7 seconds rendered by Blender (Cycles, 1080 x 1920).
- `renders/Susanna_finger_fast2d.mp4`: the same 7 seconds from `scripts/stage2d.py` (the fast renderer).
- `layers/susanna_finger/`: the layers the stage is made of (the room, the door, Susanna, her hand), each one
  the painting itself cut along its outline; `*_edge.png` give the cards thickness, `hand_caster.png` is the
  moving part's shadow.
- `paintings/3.png`: your new picture 3, untouched. `scenes/susanna_finger.json`: every decision for this stage.
- `scripts/`: the tools that made it (also in the garry-forge repository, `studio/figaro/blender/`).

The paintings stay exactly as they are. Each one becomes a small theatre: the room at the back, the people as painted
cards in front of it, and only the parts that must move (a hand, a head) cut out and given bones. Nothing is
repainted or redrawn; what a moving part uncovers is rebuilt from the painting around it.

Made 2026-10-30 for Alex's first experiment: Susanna's raised finger (episode 1, the new picture 3).

## What is tested and what is not

Tested here (Claude's Linux machine, Blender 4.5.9 LTS as a Python module, Cycles on the CPU):
- making the layers (`cutout.py`), building the scene (`build_scene.py`), rendering (`render.py`);
- the Susanna scene rendered in full, every frame looked at in close-up (hand, finger, shadows, edges).

Not tested yet:
- opening the `.blend` in Blender on your Mac (it is a normal Blender 4.5 file; you are the first to open it there);
- Eevee (Blender's fast preview renderer): Eevee needs a graphics card, which Claude's machine does not have;
- running the scripts from Blender's Scripting workspace instead of the command line.

## Open and play the scene (Blender 4.5 LTS on the Mac)

1. Download this folder from GitHub and keep it together.
2. In Blender: **File > Open**, choose `Susanna_finger.blend`. The paintings are packed inside the file.
3. Hover the mouse over the big 3D view and press **Numpad 0** (no number pad: **View > Cameras > Active Camera**)
   to look through the camera.
4. Press **Space** to play the 7 seconds; press Space again to stop.
5. To see the finished picture: top right of the 3D view, the four little spheres: click the last one
   (**Rendered**). On a MacBook this is slow; the full video is already in `renders/`.

## Change the gesture

1. In the **Outliner** (the list at the top right) click **hand rig (pose these bones)**.
2. Press **Ctrl + Tab**: Pose Mode. Three bones appear on the arm: `forearm` (stays still), `wrist`, `finger`.
3. Click a bone, press **R**, move the mouse: it turns in the picture's plane. Click to keep, right-click to cancel.
4. To store it at the current moment of the timeline: press **I** and choose **Rotation**.
5. The timeline at the bottom shows the stored moments (small diamonds). Drag a diamond to change the timing;
   select one and press **X** to delete it.

The skin bends instead of folding like paper because each bone moves its part fully and the pull fades out over a
few pixels around it.

## Change the camera

- **Camera turn (rotate Z for depth)**: the camera's arm. Turning it a few degrees round its Z axis shows the depth
  between the cards (the room slides behind Susanna). More than about 3 degrees uncovers painting that was never
  rebuilt.
- **Camera > Orthographic Scale** (camera properties, the green camera icon): smaller = closer.

## How a scene is made (what Claude does)

```
paintings/3.png  +  scenes/susanna_finger.json
        |  cutout.py       the layers: each one the painting itself, cut, with what it hides rebuilt
        v
build/susanna_finger/*.png + layers.json
        |  build_scene.py  cards at their depths, light, bones, camera, animation
        v
blend/susanna_finger.blend
        |  render.py       frames and the video (Cycles)
        v
renders/susanna_finger.mp4
```

The scene file (`scenes/<name>.json`) holds everything that is decided by hand:
- `layers`, back to front: `room` (the back), then cards with a `depth` (1 = 100 picture pixels in front of the
  back wall). A card is cut either with Segment Anything (`"sam"`: a box around it and a few points on it and off
  it) or along a traced outline (`"poly"`); `"plus"` adds traced pieces that belong to it (Susanna's table).
- A moving part names the card it sits on (`"over": "susanna"`) and `"moves": true`; `"moving"` is the area that
  really moves (the far end of the forearm stays still, so nothing is rebuilt there).
- `"rig"`: bones (head, tail in picture pixels, parent) and the area each bone moves (`region`, fading out over
  `soft` pixels).
- `animation`: for each bone, `turn` (degrees; positive = anticlockwise as you look at it) and `shift` (pixels)
  at given seconds; the camera's framing (centre and zoom) and turn (degrees) at given seconds.
- `light`: `shadow` (how much a full shadow darkens, 0.4 = 40 %), `offset` (how far shadows fall per unit of depth:
  right, down; the light comes from the window side), `softness_degrees`.

The light is balanced so that every lit pixel shows exactly the painting's colour; only the shadows the cards cast
on each other are new.

Commands (on a computer with Python, numpy, opencv-python, Pillow, onnxruntime and Blender 4.5 or the `bpy` module):

```
python3 segment.py paintings/Q.png masks/figaro.png --box 370 440 790 1420 --in 560,540 600,720 --out 700,300
python3 cutout.py scenes/susanna_finger.json build/susanna_finger
python3 build_scene.py build/susanna_finger blend/susanna_finger.blend
python3 render.py blend/susanna_finger.blend renders/susanna_finger_frames
```

Segment Anything (Meta's SAM, ViT-B, the ONNX files published with the rembg project, about 110 MB) is looked for
in `models/` or the folder named by `GARRY_SAM_MODELS`. Its full-size mask output is misplaced in this export, so
`segment.py` enlarges its 256 x 256 mask itself (checked against Susanna's face box).

## The rest of the paintings

ChatGPT's six batches hold no new paintings: every one of them is one of the 46 already in the episodes (matched
pixel for pixel, at a smaller size). Their masks are first rough passes (some figures without heads, one almost
empty), so the cuts are made again from the full-size originals with Segment Anything. For each painting:

1. Read its ANIMATION_NOTES and the episode's words for that shot; decide what moves (as few parts as possible).
2. Cut each figure (box + clicks), check the outline on the painting, add clicks where it strays.
3. Moving parts: their own card only when they move far (a hand, a fan, a letter); heads, brows, hair and
   breathing bend on the figure's own card with bones, no cutting.
4. `cutout.py`, `build_scene.py`, a few test frames at half size, look at every edge, then the full render.

## Rendering time

- Blender's Cycles on Claude's machine: about 20 seconds a frame at 1080 x 1920 (12 samples, denoised). A one-minute
  episode is 1,650 frames: about 9 hours.
- `stage2d.py` draws the same stage without Blender: the camera is orthographic, so nothing needs real 3D (cards
  slide sideways by their depth when the camera turns, shadows are each card's outline moved by the gap, bones bend
  the card as Blender's armature does). About 0.6 seconds a frame: a one-minute episode in about 17 minutes. Checked
  against Blender's own frames of the Susanna scene: they match within about one colour level out of 255.
- Eevee, Blender's fast renderer, gives the same picture (one frame checked here with a software driver, which is
  slower than Cycles). With a good graphics card it is the quick way to render inside Blender itself.

So: build and adjust a stage in Blender, render the episodes with `stage2d.py`; Blender stays the place to look at
a stage, try a gesture or check a frame.

```
python3 stage2d.py build/susanna_finger renders/susanna_finger_2d          # all frames + the .mp4
python3 stage2d.py build/susanna_finger renders/check 70-70                # one frame
```
