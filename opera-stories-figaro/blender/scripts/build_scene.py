"""Build a painted-cutout stage in Blender from the layers cutout.py made, rig its moving parts and animate them.

Headless (how Claude builds the scenes):
    python3 build_scene.py <build folder> <out.blend>                  (with the bpy module, Blender 4.5)
    blender -b -P build_scene.py -- <build folder> <out.blend>         (with Blender itself)
In Blender's Scripting workspace: open this file, set BUILD below to the build folder, press Run Script.

What it makes (1 Blender unit = 100 picture pixels, so the stage keeps the painting's exact proportions):
- one card per layer, at its depth in front of the back wall, cut along its outline (the picture's own alpha);
  a darker copy just behind each card gives it a little thickness;
- the light: a sun from the window side that casts the cards' shadows, plus the painting's own colour as an even
  glow, balanced so a lit pixel shows exactly the painting's colour and a shadow darkens it by `shadow` (0.35 = 35 %);
- moving parts (a hand, a head) as a fine mesh bent by an armature: each bone moves its region fully and fades out
  over `soft` pixels, so skin bends instead of hinging like paper;
- an orthographic camera that glides between framings and turns a few degrees around the actor, so the cards show
  their depth;
- keyframes from the scene's "animation" (seconds and degrees; positive = anticlockwise as seen by the camera).
"""
import json
import math
import os
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Quaternion, Vector

BUILD = ""   # set this when running inside Blender's Scripting workspace (the folder with layers.json)
PX = 0.01    # Blender units per picture pixel


def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    build = argv[0] if argv else BUILD
    out = argv[1] if len(argv) > 1 else os.path.join(build, "scene.blend")
    if not build or not os.path.exists(os.path.join(build, "layers.json")):
        raise SystemExit("no layers.json in %r: run cutout.py first, then give its folder" % build)
    return os.path.abspath(build), os.path.abspath(out)


class Stage:
    def __init__(self, info):
        self.w, self.h = info["size"]
        self.info = info

    def world(self, x, y, depth):
        """Picture pixel (x, y) on a card `depth` units in front of the back wall (the camera looks along +Y)."""
        return Vector(((x - self.w / 2) * PX, -depth, (self.h / 2 - y) * PX))


def material(name, image, shadow, alpha=True, tint=None):
    """The painting's colour, lit: diffuse (takes the sun and its shadows) + an emission of the same colour that
    stands in for soft room light. With the sun at strength `shadow` and the glow at `1 - shadow`, a lit pixel is
    exactly the painting's colour and a pixel in full shadow is darker by `shadow` (0.35 = 35 %)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = image
    tex.interpolation = "Linear"
    tex.extension = "CLIP"
    colour = tex.outputs["Color"]
    if tint is not None:   # the thickness card: the painting darkened, like the cut edge of painted board
        dark = nt.nodes.new("ShaderNodeHueSaturation")
        dark.inputs["Value"].default_value = tint
        nt.links.new(colour, dark.inputs["Color"])
        colour = dark.outputs["Color"]
    diffuse = nt.nodes.new("ShaderNodeBsdfDiffuse")
    nt.links.new(colour, diffuse.inputs["Color"])
    glow = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(colour, glow.inputs["Color"])
    glow.inputs["Strength"].default_value = 1.0 - shadow
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(diffuse.outputs["BSDF"], add.inputs[0])
    nt.links.new(glow.outputs["Emission"], add.inputs[1])
    surface = add.outputs["Shader"]
    if alpha:
        clear = nt.nodes.new("ShaderNodeBsdfTransparent")
        cut = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(tex.outputs["Alpha"], cut.inputs["Fac"])
        nt.links.new(clear.outputs["BSDF"], cut.inputs[1])
        nt.links.new(surface, cut.inputs[2])
        surface = cut.outputs["Shader"]
    nt.links.new(surface, out.inputs["Surface"])
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"   # Eevee in Blender 4.2+: alpha cut-outs with shadows
    if hasattr(mat, "cycles") and hasattr(mat.cycles, "emission_sampling"):
        mat.cycles.emission_sampling = "NONE"   # the glow is the painting's colour, not a lamp: never sampled as light
    return mat


def card_mesh(name, stage, layer, alpha, cell):
    """A mesh that follows the layer's outline: grid cells (cell pixels) wherever the picture's alpha shows,
    UVs pinned to the picture, so the card lines up with the painting pixel for pixel."""
    w, h = stage.w, stage.h
    depth = layer["depth"]
    if alpha is None:   # the back wall: one rectangle
        cols, rows, x0, y0, x1, y1 = 1, 1, 0, 0, w, h
        keep = [[True]]
    else:
        ys, xs = np.nonzero(alpha)
        y0, y1 = max(0, int(ys.min()) - cell), min(h, int(ys.max()) + cell + 1)
        x0, x1 = max(0, int(xs.min()) - cell), min(w, int(xs.max()) + cell + 1)
        cols, rows = max(1, math.ceil((x1 - x0) / cell)), max(1, math.ceil((y1 - y0) / cell))
        occupied = np.zeros((rows, cols), bool)
        occupied[(ys - y0) // cell, (xs - x0) // cell] = True
        keep = occupied.copy()   # one cell of margin so the soft edge is never cut
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                keep |= np.roll(np.roll(occupied, dr, 0), dc, 1)
    bm = bmesh.new()
    uv = bm.loops.layers.uv.new("UVMap")
    verts = {}

    def place(i, j):
        if alpha is None:
            return x0 + j * (x1 - x0) / cols, y0 + i * (y1 - y0) / rows
        return min(x1, x0 + j * cell), min(y1, y0 + i * cell)

    def vert(i, j):
        if (i, j) not in verts:
            verts[(i, j)] = bm.verts.new(stage.world(*place(i, j), depth))
        return verts[(i, j)]
    for i in range(rows):
        for j in range(cols):
            if keep[i][j]:
                # anticlockwise as seen from the camera (-Y), so the face looks at the camera
                corners = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
                f = bm.faces.new([vert(a, b) for a, b in corners])
                for loop, (a, b) in zip(f.loops, corners):
                    x, y = place(a, b)
                    loop[uv].uv = (x / w, 1 - y / h)
    bm.normal_update()
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


def alpha_rows(path, w, h):
    """Which pixels of a layer show (alpha > 0), top row first, read with Blender's own image loader."""
    img = bpy.data.images.load(path, check_existing=True)
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)   # RGBA floats, bottom row first
    return img, px.reshape(h, w, 4)[::-1, :, 3] > 0.004


def smooth(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def region_weight(x, y, poly, soft):
    """1 inside the region, fading to 0 over `soft` pixels outside it (distance to the polygon's edges)."""
    inside = False
    n = len(poly)
    for k in range(n):
        (xa, ya), (xb, yb) = poly[k], poly[(k + 1) % n]
        if (ya > y) != (yb > y) and x < xa + (y - ya) * (xb - xa) / (yb - ya):
            inside = not inside
    if inside:
        return 1.0
    best = 1e9
    for k in range(n):
        (xa, ya), (xb, yb) = poly[k], poly[(k + 1) % n]
        dx, dy = xb - xa, yb - ya
        t = max(0.0, min(1.0, ((x - xa) * dx + (y - ya) * dy) / max(dx * dx + dy * dy, 1e-9)))
        best = min(best, math.hypot(x - xa - t * dx, y - ya - t * dy))
    return 1.0 - smooth(best / max(soft, 1e-6))


def rig(stage, layer, obj):
    """An armature for a moving part: bones in the picture plane, weights from the scene's regions."""
    spec = layer["rig"]
    depth = layer["depth"] + 0.001
    arm_data = bpy.data.armatures.new(layer["name"] + " rig")
    arm = bpy.data.objects.new(layer["name"] + " rig (pose these bones)", arm_data)
    bpy.context.scene.collection.objects.link(arm)
    arm.show_in_front = True
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for b in spec["bones"]:
        eb = arm_data.edit_bones.new(b["name"])
        eb.head = stage.world(b["head"][0], b["head"][1], depth)
        eb.tail = stage.world(b["tail"][0], b["tail"][1], depth)
        eb.roll = 0.0
        if b.get("parent"):
            eb.parent = arm_data.edit_bones[b["parent"]]
    bpy.ops.object.mode_set(mode="OBJECT")
    # weights: listed regions from the tip of the chain down; what a region leaves goes to the next, the rest to
    # the first bone (the root, which stays still)
    root = spec["bones"][0]["name"]
    groups = {b["name"]: obj.vertex_groups.new(name=b["name"]) for b in spec["bones"]}
    me = obj.data
    for v in me.vertices:
        x = v.co.x / PX + stage.w / 2
        y = stage.h / 2 - v.co.z / PX
        left = 1.0
        for region in spec["weights"]:
            wgt = region_weight(x, y, region["region"], region.get("soft", 15)) * left
            if wgt > 0:
                groups[region["bone"]].add([v.index], wgt, "REPLACE")
            left -= wgt
        if left > 0:
            groups[root].add([v.index], left, "REPLACE")
    mod = obj.modifiers.new("Bones bend the card", "ARMATURE")
    mod.object = arm
    obj.parent = arm
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    return arm


def turn(arm, bone, degrees):
    """A bone's rotation in the picture plane: positive = anticlockwise as seen by the camera."""
    rest = arm.data.bones[bone].matrix_local.to_3x3()
    axis = rest.inverted() @ Vector((0.0, -1.0, 0.0))
    return Quaternion(axis.normalized(), math.radians(degrees))


def move(arm, bone, dx_px, dy_px):
    """A bone's shift in picture pixels (x right, y down), in the bone's own axes."""
    rest = arm.data.bones[bone].matrix_local.to_3x3()
    return rest.inverted() @ Vector((dx_px * PX, 0.0, -dy_px * PX))


def animate(stage, info, rigs, cam_pivot, cam):
    anim = info.get("animation", {})
    fps = int(anim.get("fps", 30))
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = max(1, int(round(float(anim.get("seconds", 6.0)) * fps)))

    def frame(t):
        return 1 + t * fps
    for layer_name, bones in anim.get("bones", {}).items():
        arm = rigs[layer_name]
        for bone, keys in bones.items():
            pb = arm.pose.bones[bone]
            for t, deg in keys.get("turn", []):
                pb.rotation_quaternion = turn(arm, bone, deg)
                pb.keyframe_insert("rotation_quaternion", frame=frame(t))
            for t, (dx, dy) in keys.get("shift", []):
                pb.location = move(arm, bone, dx, dy)
                pb.keyframe_insert("location", frame=frame(t))
    pivot_depth = float(info.get("stage", {}).get("pivot_depth", 1.6))
    for t, (cx, cy, zoom), deg in anim.get("camera", [[0, [0.5, 0.5, 1.05], 0]]):
        # the pivot sits on the actor's card; the camera turns around it
        cam_pivot.location = Vector(((cx - 0.5) * stage.w * PX, -pivot_depth, (0.5 - cy) * stage.h * PX))
        cam_pivot.rotation_euler = (0.0, 0.0, math.radians(deg))
        cam.data.ortho_scale = stage.h * PX / zoom
        cam_pivot.keyframe_insert("location", frame=frame(t))
        cam_pivot.keyframe_insert("rotation_euler", frame=frame(t))
        cam.data.keyframe_insert("ortho_scale", frame=frame(t))


def build(build_dir, out):
    info = json.load(open(os.path.join(build_dir, "layers.json")))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    stage = Stage(info)
    look = info.get("light", {})
    shadow = float(look.get("shadow", 0.35))
    rigs = {}
    stage_col = bpy.data.collections.new("Stage (painted cards)")
    scene.collection.children.link(stage_col)
    for layer in info["layers"]:
        path = os.path.join(build_dir, layer["file"])
        is_back = "outline" not in layer
        if is_back:
            img = bpy.data.images.load(path, check_existing=True)
            alpha = None
        else:
            img, alpha = alpha_rows(path, stage.w, stage.h)
        img.colorspace_settings.name = "sRGB"
        img.alpha_mode = "STRAIGHT"
        cell = int(layer.get("cell", 3 if layer.get("rig") else 8))
        me = card_mesh(layer["name"], stage, layer, alpha, cell)
        obj = bpy.data.objects.new(layer["name"], me)
        stage_col.objects.link(obj)
        me.materials.append(material(layer["name"] + " paint", img, shadow, alpha=not is_back))
        edge_path = os.path.join(build_dir, layer["name"] + "_edge.png")
        if not is_back and float(layer.get("thickness", 3)) > 0 and os.path.exists(edge_path):
            # thickness: a darker copy of the card just behind it, cut 2 pixels smaller so it never shows through
            # the card's soft edge from the front; it shows when the camera turns, and in the shadows
            edge_img = bpy.data.images.load(edge_path, check_existing=True)
            edge_img.alpha_mode = "STRAIGHT"
            back = bpy.data.objects.new(layer["name"] + " edge", me.copy())
            back.data.materials.clear()
            back.data.materials.append(material(layer["name"] + " edge", edge_img, shadow, tint=0.35))
            back.location.y += float(layer.get("thickness", 3)) * PX
            back.visible_shadow = False   # the card itself casts the shadow
            stage_col.objects.link(back)
            back.parent = obj
        caster_path = os.path.join(build_dir, layer["name"] + "_caster.png")
        if os.path.exists(caster_path):
            # the moving part's own shadow: an unseen copy cut to the part that moves; the card itself casts none
            caster_img = bpy.data.images.load(caster_path, check_existing=True)
            caster_img.alpha_mode = "STRAIGHT"
            caster = bpy.data.objects.new(layer["name"] + " shadow", me.copy())
            caster.data.materials.clear()
            caster.data.materials.append(material(layer["name"] + " shadow", caster_img, shadow))
            caster.visible_camera = False
            caster.visible_diffuse = caster.visible_glossy = caster.visible_transmission = False
            caster.location.y += 0.002   # a hair behind the card, so the card never shades itself
            stage_col.objects.link(caster)
            caster.parent = obj
            obj.visible_shadow = False
        if layer.get("rig"):
            rigs[layer["name"]] = rig(stage, layer, obj)
            for child in obj.children:
                mod = child.modifiers.new("Bones bend the card", "ARMATURE")
                mod.object = rigs[layer["name"]]
                for vg in obj.vertex_groups:
                    child.vertex_groups.new(name=vg.name)
                for v in child.data.vertices:
                    for g in obj.data.vertices[v.index].groups:
                        child.vertex_groups[obj.vertex_groups[g.group].name].add([v.index], g.weight, "REPLACE")
    # the sun: from the window side, a little above and in front; shadows fall `offset` of the gap between cards
    ox, oy = look.get("offset", [0.10, 0.08])   # shadow shift per unit of depth: right, down
    direction = Vector((ox, 1.0, -oy)).normalized()
    sun_data = bpy.data.lights.new("Window light", "SUN")
    sun_data.energy = shadow * math.pi / direction.y   # with the glow, a lit pixel = the painting's colour
    sun_data.angle = math.radians(float(look.get("softness_degrees", 6)))
    sun_data.color = (1.0, 1.0, 1.0)
    sun = bpy.data.objects.new("Window light", sun_data)
    sun.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(sun)
    world = bpy.data.worlds.new("Dark theatre")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
    scene.world = world
    # camera: orthographic, on an arm that turns around the actor's card
    cam_pivot = bpy.data.objects.new("Camera turn (rotate Z for depth)", None)
    scene.collection.objects.link(cam_pivot)
    cam_data = bpy.data.cameras.new("Orthographic camera")
    cam_data.type = "ORTHO"
    cam_data.clip_start, cam_data.clip_end = 0.1, 200.0
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    cam.parent = cam_pivot
    cam.location = (0.0, -40.0, 0.0)
    cam.rotation_euler = (math.radians(90), 0.0, 0.0)
    scene.camera = cam
    animate(stage, info, rigs, cam_pivot, cam)
    # render: Cycles (works without a graphics card), direct light only so colours stay the painting's
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = int(info.get("render", {}).get("samples", 32))
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 16
    scene.cycles.caustics_reflective = scene.cycles.caustics_refractive = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.render.resolution_x, scene.render.resolution_y = 1080, 1920
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = "//frames/frame_"
    scene.cycles.filter_width = 1.2   # a little crisper than the default 1.5: the brushwork stays sharp
    scene.cycles.use_adaptive_sampling = True
    scene.render.use_persistent_data = True   # keep the cards loaded between frames
    os.makedirs(os.path.dirname(out), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out, relative_remap=True)
    print("saved", out, "| frames", scene.frame_start, "-", scene.frame_end, "| rigs:", ", ".join(rigs) or "none")


if __name__ == "__main__":
    build(*args())
