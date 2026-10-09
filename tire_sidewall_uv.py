from __future__ import annotations
# ==============================================================
#   TIRE SIDEWALL UV  v1.1
#   by Alfredo Mendez
#
#   Detects tire sidewall faces, projects a planar disc UV
#   exactly along the wheel axis, sized to the tire diameter,
#   with live texture preview for UE5 export.
#
#   v1.1: UV Mode — Same UV (default, channel 0) or Separate UV
#         Park Rest of Tire option for single-UV workflows
#
#   Compatible with Blender 4.3 and above
#   Panel: N-bar -> "Tire UV" tab
# ==============================================================

import bpy
import bmesh
import math
import numpy as np
from mathutils import Vector
from bpy_extras.io_utils import ImportHelper


bl_info = {
    "name": "Tire Sidewall UV",
    "author": "Alfredo Mendez",
    "version": (1, 1),
    "blender": (4, 3, 0),
    "location": "View3D > Sidebar > Tire UV",
    "description": "Planar disc UV for tire sidewalls with texture preview, ready for UE5.",
    "category": "UV",
}

FRAME_KEY      = "tsuv_frame"        # [cx,cy,cz, ax,ay,az, r_in, r_out]
COUNT_KEY      = "tsuv_count"
PREVIEW_KEY    = "tsuv_preview_on"
SIDE_ATTR      = "tsuv_sidewall"     # face int: 0 none, 1 outer, 2 inner
ORIG_MAT_ATTR  = "tsuv_orig_mat"     # face int: original material index
PREVIEW_MAT    = "TSUV_Sidewall_Preview"
PARK_UV        = (0.002, 0.002)      # corner of the texture (transparent area)


# ──────────────────────────────────────────────────────────────
#  GEOMETRY
# ──────────────────────────────────────────────────────────────

def world_verts_np(obj):
    me = obj.data
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    mw = np.array(obj.matrix_world)
    return co @ mw[:3, :3].T + mw[:3, 3]


def fit_circle_2d(x, y):
    """Algebraic least-squares circle fit (Kasa)."""
    A = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = math.sqrt(max(c + cx * cx + cy * cy, 0.0))
    return float(cx), float(cy), r


def plane_basis(axis):
    """e1/e2 in the wheel plane; e2 follows world up so lettering stays upright."""
    up = Vector((0, 0, 1))
    if abs(axis.dot(up)) > 0.9:
        up = Vector((1, 0, 0))
    e2 = (up - axis * up.dot(axis)).normalized()
    e1 = e2.cross(axis).normalized()
    return e1, e2


def compute_frame(obj, flip=False):
    pts = world_verts_np(obj)
    if len(pts) < 16:
        return None

    mean = pts.mean(axis=0)
    cov = np.cov((pts - mean).T)
    vals, vecs = np.linalg.eigh(cov)
    # Axis = the eigenvector whose eigenvalue differs from the other two
    if (vals[1] - vals[0]) > (vals[2] - vals[1]):
        axis = Vector(vecs[:, 0])
    else:
        axis = Vector(vecs[:, 2])
    axis.normalize()

    e1, e2 = plane_basis(axis)
    rel = pts - mean
    x = rel @ np.array(e1)
    y = rel @ np.array(e2)
    z = rel @ np.array(axis)

    # Rough center, then refine with the bead-seat ring (innermost vertices)
    cx0 = (x.min() + x.max()) / 2
    cy0 = (y.min() + y.max()) / 2
    r0 = np.hypot(x - cx0, y - cy0)
    r_lo = np.percentile(r0, 3)
    band = r0 <= r_lo + (r0.max() - r_lo) * 0.05
    if band.sum() >= 8:
        cx, cy, _ = fit_circle_2d(x[band], y[band])
    else:
        cx, cy = cx0, cy0

    zc = (z.min() + z.max()) / 2
    center = Vector(mean) + e1 * cx + e2 * cy + axis * zc

    r = np.hypot(x - cx, y - cy)
    r_in, r_out = float(r.min()), float(r.max())

    # Outward = away from car center (world origin after Car Prep)
    if center.dot(axis) < 0:
        axis = -axis
    if flip:
        axis = -axis

    return center, axis, r_in, r_out


def store_frame(obj, center, axis, r_in, r_out):
    obj[FRAME_KEY] = [center.x, center.y, center.z, axis.x, axis.y, axis.z, r_in, r_out]


def load_frame(obj):
    f = obj.get(FRAME_KEY)
    if f is None or len(f) != 8:
        return None
    return Vector(f[0:3]), Vector(f[3:6]).normalized(), f[6], f[7]


def ensure_face_int_attr(me, name):
    attr = me.attributes.get(name)
    if attr is None or attr.domain != 'FACE' or attr.data_type != 'INT':
        if attr is not None:
            me.attributes.remove(attr)
        attr = me.attributes.new(name, 'INT', 'FACE')
    return attr


def detect_sidewall(obj, props):
    frame = load_frame(obj)
    if frame is None:
        return 0
    center, axis, r_in, r_out = frame
    me = obj.data
    ensure_face_int_attr(me, SIDE_ATTR)

    mw = obj.matrix_world
    nmat = mw.to_3x3().inverted_safe().transposed()
    thr = props.normal_threshold
    min_r = props.min_radius * r_out

    bm = bmesh.new()
    bm.from_mesh(me)
    lay = bm.faces.layers.int.get(SIDE_ATTR)
    count = 0
    for f in bm.faces:
        n = (nmat @ f.normal).normalized()
        d = n.dot(axis)
        c = mw @ f.calc_center_median()
        rel = c - center
        radial = (rel - axis * rel.dot(axis)).length
        side = 0
        if radial >= min_r:
            if d > thr and props.side in {'OUTER', 'BOTH'}:
                side = 1
            elif d < -thr and props.side in {'INNER', 'BOTH'}:
                side = 2
        f[lay] = side
        if side:
            count += 1
    bm.to_mesh(me)
    bm.free()
    me.update()
    obj[COUNT_KEY] = count
    return count


def image_aspect_scale(img):
    if img is None or img.size[0] == 0 or img.size[1] == 0:
        return 1.0, 1.0
    w, h = img.size[0], img.size[1]
    m = min(w, h)
    return m / w, m / h


# ──────────────────────────────────────────────────────────────
#  UV TARGET
# ──────────────────────────────────────────────────────────────

def target_uv_name(me, props, create=False):
    """
    SAME     -> first UV map of the tire (UE channel 0); created as 'UVMap' if none
    SEPARATE -> named UV map; created if missing
    Returns the UV map name, or None if it doesn't exist and create=False.
    """
    if props.uv_mode == 'SAME':
        if len(me.uv_layers) == 0:
            if not create:
                return None
            me.uv_layers.new(name="UVMap")
        return me.uv_layers[0].name

    name = props.uv_name.strip() or "UV_Sidewall"
    if me.uv_layers.get(name) is None:
        if not create or len(me.uv_layers) >= 8:
            return None
        me.uv_layers.new(name=name)
    return name


def apply_uv(obj, props):
    frame = load_frame(obj)
    me = obj.data
    if frame is None or me.attributes.get(SIDE_ATTR) is None:
        return False
    center, axis, r_in, r_out = frame
    diameter = 2.0 * r_out
    if diameter <= 0:
        return False

    uv_name = target_uv_name(me, props, create=True)
    if uv_name is None:
        return False

    park = props.uv_mode == 'SAME' and props.park_rest

    su, sv = image_aspect_scale(props.image)
    cos_a, sin_a = math.cos(props.rotation), math.sin(props.rotation)
    s = props.scale

    bases = {
        1: plane_basis(axis),    # outer: viewed from outside
        2: plane_basis(-axis),   # inner: viewed from inside
    }

    mw = obj.matrix_world
    bm = bmesh.new()
    bm.from_mesh(me)
    uv_lay = bm.loops.layers.uv.get(uv_name)
    side_lay = bm.faces.layers.int.get(SIDE_ATTR)

    for f in bm.faces:
        side = f[side_lay]
        if not side:
            if park:
                for loop in f.loops:
                    loop[uv_lay].uv = PARK_UV
            continue
        e1, e2 = bases[side]
        for loop in f.loops:
            rel = (mw @ loop.vert.co) - center
            x = rel.dot(e1) / diameter
            y = rel.dot(e2) / diameter
            xr = x * cos_a - y * sin_a
            yr = x * sin_a + y * cos_a
            u = 0.5 + xr * s * su + props.offset_u
            v = 0.5 + yr * s * sv + props.offset_v
            loop[uv_lay].uv = (u, v)

    bm.to_mesh(me)
    bm.free()
    me.update()
    return True


# ──────────────────────────────────────────────────────────────
#  PREVIEW MATERIAL
# ──────────────────────────────────────────────────────────────

def build_preview_material(img, uv_name):
    mat = bpy.data.materials.get(PREVIEW_MAT) or bpy.data.materials.new(PREVIEW_MAT)
    try:
        mat.use_nodes = True
    except Exception:
        pass
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    nodes.clear()

    out  = nodes.new('ShaderNodeOutputMaterial');  out.location  = (600, 0)
    bsdf = nodes.new('ShaderNodeBsdfPrincipled');  bsdf.location = (300, 0)
    mix  = nodes.new('ShaderNodeMix');             mix.location  = (80, 0)
    mix.data_type = 'RGBA'
    tex  = nodes.new('ShaderNodeTexImage');        tex.location  = (-260, 0)
    tex.image = img
    tex.extension = 'CLIP'
    uvn  = nodes.new('ShaderNodeUVMap');           uvn.location  = (-480, 0)
    uvn.uv_map = uv_name

    mix.inputs[6].default_value = (0.02, 0.02, 0.02, 1.0)   # A: rubber base
    bsdf.inputs['Roughness'].default_value = 0.85

    links.new(uvn.outputs['UV'], tex.inputs['Vector'])
    links.new(tex.outputs['Alpha'], mix.inputs[0])           # Factor
    links.new(tex.outputs['Color'], mix.inputs[7])           # B: texture
    links.new(mix.outputs[2], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs[0], out.inputs['Surface'])
    return mat


def preview_on(obj, props):
    me = obj.data
    side_attr = me.attributes.get(SIDE_ATTR)
    uv_name = target_uv_name(me, props, create=False)
    if side_attr is None or uv_name is None:
        return False
    n = len(me.polygons)
    sides = np.empty(n, dtype=np.int32)
    side_attr.data.foreach_get("value", sides)

    mat = build_preview_material(props.image, uv_name)

    if not obj.get(PREVIEW_KEY):
        idx_now = np.empty(n, dtype=np.int32)
        me.polygons.foreach_get("material_index", idx_now)
        orig = ensure_face_int_attr(me, ORIG_MAT_ATTR)
        orig.data.foreach_set("value", idx_now)

    slot = me.materials.find(mat.name)
    if slot == -1:
        if len(me.materials) == 0:
            me.materials.append(None)   # keep slot 0 for the original look
        me.materials.append(mat)
        slot = len(me.materials) - 1

    orig_idx = np.empty(n, dtype=np.int32)
    me.attributes[ORIG_MAT_ATTR].data.foreach_get("value", orig_idx)
    new_idx = np.where(sides > 0, slot, orig_idx).astype(np.int32)
    me.polygons.foreach_set("material_index", new_idx)
    me.update()
    obj[PREVIEW_KEY] = True
    return True


def preview_off(obj):
    me = obj.data
    if not obj.get(PREVIEW_KEY):
        return False
    orig = me.attributes.get(ORIG_MAT_ATTR)
    if orig is not None:
        n = len(me.polygons)
        idx = np.empty(n, dtype=np.int32)
        orig.data.foreach_get("value", idx)
        me.polygons.foreach_set("material_index", idx)
        me.attributes.remove(orig)
    slot = me.materials.find(PREVIEW_MAT)
    if slot != -1:
        me.materials.pop(index=slot)
    me.update()
    obj[PREVIEW_KEY] = False
    return True


# ──────────────────────────────────────────────────────────────
#  PROPERTIES (live update)
# ──────────────────────────────────────────────────────────────

def _active_ready(context):
    obj = context.active_object
    if obj and obj.type == 'MESH' and obj.mode == 'OBJECT' and FRAME_KEY in obj:
        return obj
    return None


def _live_update(self, context):
    if not self.auto_update:
        return
    obj = _active_ready(context)
    if obj:
        apply_uv(obj, self)


def _image_update(self, context):
    mat = bpy.data.materials.get(PREVIEW_MAT)
    if mat and mat.node_tree:
        for nd in mat.node_tree.nodes:
            if nd.type == 'TEX_IMAGE':
                nd.image = self.image
    _live_update(self, context)


class TSUVProperties(bpy.types.PropertyGroup):
    image: bpy.props.PointerProperty(
        name="Texture", type=bpy.types.Image, update=_image_update,
        description="Sidewall texture (lettering / logo disc)",
    )
    uv_mode: bpy.props.EnumProperty(
        name="UV Mode",
        items=[
            ('SAME',     "Same UV",     "Write the disc into the tire's main UV map (UE channel 0)"),
            ('SEPARATE', "Separate UV", "Write the disc into its own UV map (extra UE channel)"),
        ],
        default='SAME',
    )
    uv_name: bpy.props.StringProperty(
        name="UV Layer", default="UV_Sidewall",
        description="Name of the separate UV map that receives the sidewall projection",
    )
    park_rest: bpy.props.BoolProperty(
        name="Park Rest of Tire", default=True, update=_live_update,
        description=(
            "Move the UVs of all non-sidewall faces to the texture corner (transparent), "
            "so the lettering only shows on the sidewall. Turn off if the rest of the tire "
            "uses its own textures on this UV"
        ),
    )
    side: bpy.props.EnumProperty(
        name="Side",
        items=[
            ('OUTER', "Outer", "Outboard sidewall only"),
            ('INNER', "Inner", "Inboard sidewall only"),
            ('BOTH',  "Both",  "Both sidewalls (each reads correctly from its side)"),
        ],
        default='OUTER',
    )
    flip_side: bpy.props.BoolProperty(
        name="Flip Outer/Inner", default=False,
        description="Use if the outer side was detected on the wrong face",
    )
    normal_threshold: bpy.props.FloatProperty(
        name="Normal Threshold", default=0.6, min=0.1, max=0.99,
        description="How parallel to the wheel axis a face normal must be to count as sidewall",
    )
    min_radius: bpy.props.FloatProperty(
        name="Min Radius", default=0.0, min=0.0, max=0.95, subtype='FACTOR',
        description="Ignore faces closer to the axis than this fraction of the tire radius",
    )
    rotation: bpy.props.FloatProperty(
        name="Rotation", default=0.0, subtype='ANGLE', update=_live_update,
    )
    scale: bpy.props.FloatProperty(
        name="Scale", default=1.0, min=0.1, max=5.0, update=_live_update,
        description="1.0 = tire outer diameter fills the 0-1 UV square",
    )
    offset_u: bpy.props.FloatProperty(name="Offset U", default=0.0, step=0.1, update=_live_update)
    offset_v: bpy.props.FloatProperty(name="Offset V", default=0.0, step=0.1, update=_live_update)
    auto_update: bpy.props.BoolProperty(name="Live Update", default=True)


# ──────────────────────────────────────────────────────────────
#  OPERATORS
# ──────────────────────────────────────────────────────────────

def _ensure_object_mode(context):
    if context.active_object and context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


class TSUV_OT_load_texture(bpy.types.Operator, ImportHelper):
    bl_idname = "tsuv.load_texture"
    bl_label = "Load Texture"
    bl_description = "Load the sidewall texture from disk"

    filter_glob: bpy.props.StringProperty(
        default="*.png;*.jpg;*.jpeg;*.tga;*.tif;*.tiff;*.exr;*.bmp",
        options={'HIDDEN'},
    )

    def execute(self, context):
        try:
            img = bpy.data.images.load(self.filepath, check_existing=True)
        except RuntimeError as e:
            self.report({'ERROR'}, f"Could not load image: {e}")
            return {'CANCELLED'}
        context.scene.tsuv_props.image = img
        self.report({'INFO'}, f"Loaded {img.name}  ({img.size[0]}x{img.size[1]})")
        return {'FINISHED'}


class TSUV_OT_detect_and_map(bpy.types.Operator):
    bl_idname = "tsuv.detect_and_map"
    bl_label = "Detect & Map"
    bl_description = "Find wheel axis/center, detect sidewall faces and project the planar disc UV"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        props = context.scene.tsuv_props
        obj = context.active_object
        _ensure_object_mode(context)

        was_preview = bool(obj.get(PREVIEW_KEY))
        if was_preview:
            preview_off(obj)

        frame = compute_frame(obj, flip=props.flip_side)
        if frame is None:
            self.report({'ERROR'}, "Not enough geometry to analyze.")
            return {'CANCELLED'}
        store_frame(obj, *frame)

        count = detect_sidewall(obj, props)
        if count == 0:
            self.report({'WARNING'}, "No sidewall faces found. Lower Normal Threshold or check Side.")
            return {'CANCELLED'}

        if not apply_uv(obj, props):
            self.report({'ERROR'}, "Could not write UVs (max 8 UV maps?).")
            return {'CANCELLED'}

        if was_preview:
            preview_on(obj, props)

        d_cm = frame[3] * 200
        self.report({'INFO'}, f"{count} sidewall faces mapped | Tire Ø {d_cm:.1f} cm")
        return {'FINISHED'}


class TSUV_OT_select_faces(bpy.types.Operator):
    bl_idname = "tsuv.select_faces"
    bl_label = "Select Detected Faces"
    bl_description = "Show the detected sidewall faces in Edit Mode"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and obj.data.attributes.get(SIDE_ATTR) is not None

    def execute(self, context):
        obj = context.active_object
        _ensure_object_mode(context)
        me = obj.data
        bm = bmesh.new()
        bm.from_mesh(me)
        lay = bm.faces.layers.int.get(SIDE_ATTR)
        for v in bm.verts:
            v.select = False
        for e in bm.edges:
            e.select = False
        for f in bm.faces:
            f.select = bool(f[lay])
        bm.select_flush(True)
        bm.to_mesh(me)
        bm.free()
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='FACE')
        return {'FINISHED'}


class TSUV_OT_preview_on(bpy.types.Operator):
    bl_idname = "tsuv.preview_on"
    bl_label = "Preview On"
    bl_description = "Assign a temporary preview material to the sidewall faces using the sidewall UV"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and FRAME_KEY in obj

    def execute(self, context):
        props = context.scene.tsuv_props
        if props.image is None:
            self.report({'WARNING'}, "Load or pick a texture first.")
            return {'CANCELLED'}
        _ensure_object_mode(context)
        obj = context.active_object
        if not preview_on(obj, props):
            self.report({'ERROR'}, "Run Detect & Map first.")
            return {'CANCELLED'}
        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D' and space.shading.type in {'SOLID', 'WIREFRAME'}:
                        space.shading.type = 'MATERIAL'
        return {'FINISHED'}


class TSUV_OT_preview_off(bpy.types.Operator):
    bl_idname = "tsuv.preview_off"
    bl_label = "Preview Off"
    bl_description = "Restore original materials (UVs are kept). Do this before exporting"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and obj.get(PREVIEW_KEY)

    def execute(self, context):
        _ensure_object_mode(context)
        preview_off(context.active_object)
        return {'FINISHED'}


# ──────────────────────────────────────────────────────────────
#  PANEL
# ──────────────────────────────────────────────────────────────

class VIEW3D_PT_tire_sidewall_uv(bpy.types.Panel):
    bl_label = "Tire Sidewall UV"
    bl_idname = "VIEW3D_PT_tire_sidewall_uv"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tire UV"

    def draw(self, context):
        layout = self.layout
        props = context.scene.tsuv_props
        obj = context.active_object
        is_mesh = obj is not None and obj.type == 'MESH'

        # ── Target ─────────────────────────────────────────────
        box = layout.box()
        if is_mesh:
            box.label(text=f"Tire: {obj.name}", icon='MESH_CIRCLE')
        else:
            box.alert = True
            box.label(text="Select the tire mesh", icon='ERROR')

        # ── 1. Texture ─────────────────────────────────────────
        layout.label(text="1. Texture:", icon='TEXTURE')
        row = layout.row(align=True)
        row.prop(props, "image", text="")
        row.operator("tsuv.load_texture", text="", icon='FILEBROWSER')
        if props.image and props.image.size[0] > 0:
            w, h = props.image.size
            layout.label(text=f"  {w} x {h} px", icon='IMAGE_DATA')

        layout.separator()

        # ── 2. Detection ───────────────────────────────────────
        layout.label(text="2. Detect & Map:", icon='VIEWZOOM')
        col = layout.column(align=True)
        col.prop(props, "side", expand=True)
        col = layout.column(align=True)
        col.prop(props, "normal_threshold")
        col.prop(props, "min_radius")
        col.prop(props, "flip_side")

        # UV target
        uvbox = layout.box()
        uvbox.label(text="UV Target:", icon='GROUP_UVS')
        uvbox.row().prop(props, "uv_mode", expand=True)
        if props.uv_mode == 'SAME':
            uvbox.prop(props, "park_rest")
        else:
            uvbox.prop(props, "uv_name")

        row = layout.row()
        row.scale_y = 1.8
        row.enabled = is_mesh
        row.operator("tsuv.detect_and_map", text="▶  Detect & Map", icon='UV')
        layout.operator("tsuv.select_faces", icon='RESTRICT_SELECT_OFF')

        # Info
        if is_mesh and FRAME_KEY in obj:
            frame = load_frame(obj)
            info = layout.box()
            d_m = frame[3] * 2
            info.label(text=f"Tire Ø: {d_m * 100:.1f} cm  |  Faces: {obj.get(COUNT_KEY, 0)}", icon='INFO')
            if props.image and props.image.size[0] > 0 and d_m > 0:
                px_cm = min(props.image.size) * props.scale / (d_m * 100)
                info.label(text=f"Density: {px_cm:.1f} px/cm")
            uv_name = target_uv_name(obj.data, props, create=False)
            if uv_name:
                uv_idx = obj.data.uv_layers.find(uv_name)
                info.label(text=f"UV: {uv_name}  →  UE Channel {uv_idx}", icon='EXPORT')

        layout.separator()

        # ── 3. Adjust ──────────────────────────────────────────
        layout.label(text="3. Adjust:", icon='MODIFIER')
        col = layout.column(align=True)
        col.prop(props, "rotation")
        col.prop(props, "scale")
        row = col.row(align=True)
        row.prop(props, "offset_u", text="U")
        row.prop(props, "offset_v", text="V")
        layout.prop(props, "auto_update")

        layout.separator()

        # ── 4. Preview ─────────────────────────────────────────
        layout.label(text="4. Preview:", icon='HIDE_OFF')
        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator("tsuv.preview_on", icon='SHADING_TEXTURE')
        row.operator("tsuv.preview_off", icon='X')
        if is_mesh and obj.get(PREVIEW_KEY):
            warn = layout.box()
            warn.alert = True
            warn.label(text="Preview active — turn off before export", icon='ERROR')

        # ── Signature ──────────────────────────────────────────
        layout.separator()
        sig = layout.column()
        sig.enabled = False
        sig.label(text="Tire Sidewall UV by Alfredo Mendez", icon='AUTO')


# ──────────────────────────────────────────────────────────────
#  REGISTRATION
# ──────────────────────────────────────────────────────────────

classes = (
    TSUVProperties,
    TSUV_OT_load_texture,
    TSUV_OT_detect_and_map,
    TSUV_OT_select_faces,
    TSUV_OT_preview_on,
    TSUV_OT_preview_off,
    VIEW3D_PT_tire_sidewall_uv,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.tsuv_props = bpy.props.PointerProperty(type=TSUVProperties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.tsuv_props


if __name__ == "__main__":
    register()
