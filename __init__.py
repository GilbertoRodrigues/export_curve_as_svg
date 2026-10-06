import math
import os
import random
import re
from contextlib import contextmanager
from types import SimpleNamespace
from xml.sax.saxutils import escape

import bpy
from mathutils import Matrix, Vector
from bpy_extras.io_utils import ExportHelper
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    StringProperty,
)

# --- 1. Math & Color Helpers ---

def linear_to_srgb_single(c):
    """Applies the standard sRGB transfer function to a single linear color channel."""
    if c < 0.0031308:
        return c * 12.92
    else:
        return 1.055 * (c ** (1.0 / 2.4)) - 0.055

def rgb_to_hex(color_tuple):
    """Converts a Blender RGB tuple (Linear) into a standard SVG Hex string (sRGB)."""
    try:
        r = max(0.0, min(1.0, color_tuple[0]))
        g = max(0.0, min(1.0, color_tuple[1]))
        b = max(0.0, min(1.0, color_tuple[2]))
        
        r_srgb = linear_to_srgb_single(r)
        g_srgb = linear_to_srgb_single(g)
        b_srgb = linear_to_srgb_single(b)
        
        return f"#{int(r_srgb*255):02x}{int(g_srgb*255):02x}{int(b_srgb*255):02x}"
    except:
        return "#000000"

def get_material_color(obj):
    """Retrieve the visual color of an object."""
    mat = obj.active_material
    
    if not mat and obj.data.materials:
        mat = obj.data.materials[0]

    if not mat:
        return None 

    if mat.use_nodes and mat.node_tree:
        for node in mat.node_tree.nodes:
            if node.type in ['BSDF_PRINCIPLED', 'BSDF_DIFFUSE']:
                if len(node.inputs) > 0 and node.inputs[0].type == 'RGBA':
                    return node.inputs[0].default_value[:3]

    return mat.diffuse_color[:3]

def resolve_color(obj, source_type, user_color):
    """Determines color based on the user's selection."""
    if source_type == 'CUSTOM':
        return rgb_to_hex(user_color)
    elif source_type == 'OBJECT':
        return rgb_to_hex(obj.color[:3])
    elif source_type == 'MATERIAL':
        raw_color = get_material_color(obj)
        if raw_color:
            return rgb_to_hex(raw_color)
        else:
            return "#000000"
    elif source_type == 'RANDOM':
        rand_rgb = (random.random(), random.random(), random.random())
        return rgb_to_hex(rand_rgb)
    return "#000000"

# --- 2. Geometry Helpers ---

def get_view3d_area(context):
    """Find the current viewport, or the largest one in the current window."""
    area = context.area
    if area is None or area.type != 'VIEW_3D':
        screen = context.screen
        if screen is None:
            return None
        area = max(
            (area for area in screen.areas if area.type == 'VIEW_3D'),
            key=lambda area: area.width * area.height,
            default=None,
        )
    return area


def get_view_orientation_matrix(context):
    """Copy the viewport's world-to-view rotation before opening a dialog."""
    area = get_view3d_area(context)
    if area is None:
        return None

    region_3d = None
    if area == context.area:
        region_3d = context.region_data
    if region_3d is None:
        region_3d = area.spaces.active.region_3d
    if region_3d is None:
        return None

    matrix = region_3d.view_matrix.to_3x3()
    if abs(matrix.determinant()) < 0.000001:
        return None
    return matrix.to_4x4()


def get_camera_projection(context):
    """Copy the evaluated camera and render frame into an export projection."""
    camera = context.scene.camera
    if camera is None:
        raise ValueError("Set an active camera in the scene before exporting")
    camera = camera.evaluated_get(context.evaluated_depsgraph_get())
    if camera.data.type not in {'PERSP', 'ORTHO'}:
        raise ValueError("Camera projection supports perspective and orthographic cameras")
    render = context.scene.render
    width = render.resolution_x / 100.0
    height = render.resolution_y * render.pixel_aspect_y / render.pixel_aspect_x / 100.0
    frame = camera.data.view_frame(scene=context.scene)
    perspective = camera.data.type == 'PERSP'
    if perspective:
        frame = [v / -v.z for v in frame]
    left, right = min(v.x for v in frame), max(v.x for v in frame)
    bottom, top = min(v.y for v in frame), max(v.y for v in frame)
    return SimpleNamespace(
        matrix=camera.matrix_world.normalized().inverted(),
        perspective=perspective, width=width, height=height,
        left=left, bottom=bottom,
        x_scale=width / (right - left), y_scale=height / (top - bottom),
    )


def camera_project_spline(spline, matrix, camera, tolerance):
    """Project curves, bounding perspective flattening error in SVG units."""
    local = camera.matrix @ matrix

    def project(co):
        depth = -co.z if camera.perspective else 1.0
        return Vector(((co.x / depth - camera.left) * camera.x_scale,
                       (co.y / depth - camera.bottom) * camera.y_scale, 0.0))

    if not camera.perspective:
        affine = Matrix(((camera.x_scale, 0, 0, -camera.left * camera.x_scale),
                         (0, camera.y_scale, 0, -camera.bottom * camera.y_scale),
                         (0, 0, 1, 0), (0, 0, 0, 1)))
        return spline, affine @ local

    points = spline.bezier_points if spline.type == 'BEZIER' else spline.points
    coords = []

    def flatten(controls, depth=0):
        # Positive rational Bezier weights keep the projected curve inside
        # its projected control hull. Subdivide in 3D before perspective divide.
        if max(co.z for co in controls) >= -1e-6:
            if min(co.z for co in controls) >= -1e-6 or depth >= 20:
                raise ValueError("spline crosses or lies behind the camera plane")
        else:
            projected = [project(co) for co in controls]
            start, end = projected[0], projected[-1]
            edge = end - start
            length_squared = edge.length_squared
            error = max((co - (start + edge * max(0.0, min(
                1.0, (co - start).dot(edge) / length_squared
            )))).length if length_squared else (co - start).length
                        for co in projected[1:-1])
            if error <= tolerance:
                if not coords:
                    coords.append(start)
                coords.append(end)
                return
            if depth >= 20:
                raise ValueError("perspective approximation exceeded its subdivision limit")
        a, b, c, d = controls
        ab, bc, cd = (a + b) / 2, (b + c) / 2, (c + d) / 2
        abc, bcd = (ab + bc) / 2, (bc + cd) / 2
        middle = (abc + bcd) / 2
        flatten((a, ab, abc, middle), depth + 1)
        flatten((middle, bcd, cd, d), depth + 1)

    if spline.type == 'POLY':
        local_points = [local @ p.co.to_3d() for p in points]
        if any(co.z >= -1e-6 for co in local_points):
            raise ValueError("spline crosses or lies behind the camera plane")
        coords = [project(co) for co in local_points]
    else:
        count = len(points) if spline.use_cyclic_u else len(points) - 1
        for i in range(count):
            a, b = points[i], points[(i + 1) % len(points)]
            flatten(tuple(local @ co for co in (
                a.co, a.handle_right, b.handle_left, b.co)))
        if spline.use_cyclic_u:
            coords.pop()  # The closing endpoint is supplied by SVG's Z.
    return SimpleNamespace(
        type='POLY', use_cyclic_u=spline.use_cyclic_u,
        points=[SimpleNamespace(co=co) for co in coords],
        character_index=getattr(spline, 'character_index', None),
    ), Matrix.Identity(4)


def get_2d_coords(vector, axis_mode):
    """Projects a 3D vector onto the selected 2D plane."""
    if axis_mode == 'TOP':
        return vector.x, vector.y
    elif axis_mode == 'FRONT':
        return vector.x, vector.z 
    elif axis_mode == 'SIDE':
        return vector.y, vector.z
    return vector.x, vector.y

def get_bezier_path_d(spline, matrix, axis_mode):
    """Generates the SVG 'd' attribute string for a Bezier spline."""
    if len(spline.bezier_points) < 2:
        return ""
    
    points = spline.bezier_points
    
    v_start = matrix @ points[0].co
    sx, sy = get_2d_coords(v_start, axis_mode)
    d = [f"M {sx:.4f},{sy:.4f}"]

    for i in range(1, len(points)):
        p0 = points[i - 1]
        p1 = points[i]
        
        h1x, h1y = get_2d_coords(matrix @ p0.handle_right, axis_mode)
        h2x, h2y = get_2d_coords(matrix @ p1.handle_left, axis_mode)
        dx, dy = get_2d_coords(matrix @ p1.co, axis_mode)
        
        d.append(f"C {h1x:.4f},{h1y:.4f} {h2x:.4f},{h2y:.4f} {dx:.4f},{dy:.4f}")

    if spline.use_cyclic_u:
        p0 = points[-1]
        p1 = points[0]
        h1x, h1y = get_2d_coords(matrix @ p0.handle_right, axis_mode)
        h2x, h2y = get_2d_coords(matrix @ p1.handle_left, axis_mode)
        dx, dy = get_2d_coords(matrix @ p1.co, axis_mode)
        d.append(f"C {h1x:.4f},{h1y:.4f} {h2x:.4f},{h2y:.4f} {dx:.4f},{dy:.4f}")
        d.append("Z")
        
    return " ".join(d)

def get_poly_path_d(spline, matrix, axis_mode):
    """Generates the SVG 'd' attribute string for a Poly (Linear) spline."""
    if len(spline.points) < 2:
        return ""
        
    points = spline.points
    
    v_start = matrix @ points[0].co
    sx, sy = get_2d_coords(v_start, axis_mode)
    d = [f"M {sx:.4f},{sy:.4f}"]

    for i in range(1, len(points)):
        dx, dy = get_2d_coords(matrix @ points[i].co, axis_mode)
        d.append(f"L {dx:.4f},{dy:.4f}")

    if spline.use_cyclic_u:
        d.append("Z")
        
    return " ".join(d)


def get_bezier_axis_bounds(p0, p1, p2, p3):
    """Return the minimum and maximum of one cubic Bezier coordinate."""
    d0 = p1 - p0
    d1 = p2 - p1
    d2 = p3 - p2
    a = d0 - 2.0 * d1 + d2
    b = 2.0 * (d1 - d0)
    c = d0

    if a == 0.0:
        roots = (-c / b,) if b != 0.0 else ()
    else:
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            roots = ()
        else:
            q = -0.5 * (b + math.copysign(math.sqrt(discriminant), b))
            roots = (q / a, c / q) if q != 0.0 else (0.0,)

    values = [p0, p3]
    for t in roots:
        if 0.0 < t < 1.0:
            u = 1.0 - t
            values.append(
                u ** 3 * p0 + 3.0 * u * u * t * p1
                + 3.0 * u * t * t * p2 + t ** 3 * p3
            )

    return min(values), max(values)


def get_spline_bounds(spline, matrix, axis_mode):
    """Return bounds of an exported spline in its projection plane."""

    def project(co):
        # Match the precision of the coordinates written to the SVG paths.
        return tuple(
            round(value, 4)
            for value in get_2d_coords(matrix @ co, axis_mode)
        )

    if spline.type == 'POLY':
        coords = [project(point.co) for point in spline.points]
    else:
        points = spline.bezier_points
        segment_count = (
            len(points) if spline.use_cyclic_u else len(points) - 1
        )
        coords = []
        for i in range(segment_count):
            start = points[i]
            end = points[(i + 1) % len(points)]
            controls = (
                project(start.co),
                project(start.handle_right),
                project(end.handle_left),
                project(end.co),
            )
            x_bounds = get_bezier_axis_bounds(*(co[0] for co in controls))
            y_bounds = get_bezier_axis_bounds(*(co[1] for co in controls))
            coords.extend(zip(x_bounds, y_bounds))

    return (
        min(co[0] for co in coords),
        min(co[1] for co in coords),
        max(co[0] for co in coords),
        max(co[1] for co in coords),
    )


# --- 3. Main Export Logic ---

def sample_nurbs_data(data):
    """Consume temporary curve data and copy Blender's native evaluated paths."""
    scene = obj = nodes = None
    try:
        scene = bpy.data.scenes.new("SVG NURBS Temporary")
        obj = bpy.data.objects.new("SVG NURBS Temporary", data)
        scene.collection.objects.link(obj)
        nodes = bpy.data.node_groups.new("SVG NURBS Sampling", 'GeometryNodeTree')
        nodes.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
        nodes.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
        source = nodes.nodes.new('NodeGroupInput')
        output = nodes.nodes.new('NodeGroupOutput')
        resample = nodes.nodes.new('GeometryNodeResampleCurve')
        if resample.inputs.get('Mode') is not None:
            resample.inputs['Mode'].default_value = 'Evaluated'
        else:
            resample.mode = 'EVALUATED'
        nodes.links.new(source.outputs['Geometry'], resample.inputs['Curve'])
        nodes.links.new(resample.outputs['Curve'], output.inputs['Geometry'])
        obj.modifiers.new("SVG NURBS Sampling", 'NODES').node_group = nodes
        with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
            graph = bpy.context.evaluated_depsgraph_get()
            geometry = obj.evaluated_get(graph).evaluated_geometry()
            if geometry.curves is None:
                raise ValueError("NURBS evaluation produced no curves")
            return get_evaluated_splines(geometry.curves)
    finally:
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)
        if nodes is not None:
            bpy.data.node_groups.remove(nodes)
        if scene is not None:
            bpy.data.scenes.remove(scene)
        if isinstance(data, bpy.types.Curve):
            bpy.data.curves.remove(data)
        else:
            bpy.data.hair_curves.remove(data)


def nurbs_error(count, order, cyclic, mode, resolution):
    if count < order or order < 2:
        return "NURBS needs at least as many control points as its order"
    if resolution < 1:
        return "NURBS resolution must be positive"
    if mode in {2, 3} and ((mode == 2 and count <= order)
                          or (cyclic and count % (order - 1))):
        return "NURBS control-point count is incompatible with its Bezier knot mode"
    return None


def get_legacy_nurbs_spline(spline):
    """Evaluate a single unmodified legacy NURBS spline without fill or bevel."""
    if spline.point_count_v > 1:
        raise ValueError("NURBS surfaces are not supported")
    mode = int(spline.use_endpoint_u) + 2 * int(spline.use_bezier_u)
    error = nurbs_error(len(spline.points), spline.order_u, spline.use_cyclic_u,
                        mode, spline.resolution_u)
    if error:
        raise ValueError(error)
    data = bpy.data.curves.new("SVG NURBS Data", 'CURVE')
    try:
        data.dimensions = '3D'
        target = data.splines.new('NURBS')
        target.points.add(len(spline.points) - 1)
        for a, b in zip(target.points, spline.points):
            a.co = b.co
        for name in ('order_u', 'use_cyclic_u', 'use_endpoint_u',
                     'use_bezier_u', 'resolution_u'):
            setattr(target, name, getattr(spline, name))
    except Exception:
        bpy.data.curves.remove(data)
        raise
    sampled = sample_nurbs_data(data)
    if not sampled:
        raise ValueError("NURBS evaluation produced no paths")
    return sampled[0]


def copy_stroke_curve_data(curves):
    """Copy drawing attributes into Curves data for native NURBS evaluation."""
    if isinstance(curves, bpy.types.Curves):
        return curves.copy()
    data = bpy.data.hair_curves.new("SVG NURBS Drawing")
    try:
        data.add_curves([curve.points_length for curve in curves.curves])
        fields = {'FLOAT': 'value', 'INT': 'value', 'INT8': 'value',
                  'BOOLEAN': 'value', 'FLOAT_VECTOR': 'vector'}
        for attribute in curves.attributes:
            field = fields.get(attribute.data_type)
            if field is None:
                continue
            target = data.attributes.get(attribute.name)
            if target is None:
                target = data.attributes.new(attribute.name, attribute.data_type, attribute.domain)
            for a, b in zip(target.data, attribute.data):
                setattr(a, field, getattr(b, field))
        return data
    except Exception:
        bpy.data.hair_curves.remove(data)
        raise


def get_evaluated_splines(curves):
    """Adapt original or evaluated Curves data without changing its geometry."""
    attributes = curves.attributes
    curve_types = attributes.get('curve_type')
    cyclic = attributes.get('cyclic')
    handles_left = attributes.get('handle_left')
    handles_right = attributes.get('handle_right')
    type_names = ('CATMULL_ROM', 'POLY', 'BEZIER', 'NURBS')
    splines = []
    sampled_nurbs = None

    for i, curve in enumerate(curves.curves):
        curve_type = curve_types.data[i].value if curve_types else 0
        spline = SimpleNamespace(
            type=type_names[curve_type],
            use_cyclic_u=cyclic.data[i].value if cyclic else False,
            points=range(curve.points_length),
            bezier_points=[],
        )
        if spline.type == 'NURBS':
            def value(name, default):
                attr = attributes.get(name)
                return attr.data[i].value if attr is not None else default
            error = nurbs_error(curve.points_length, value('nurbs_order', 4),
                                spline.use_cyclic_u, value('knots_mode', 0),
                                value('resolution', 12))
            if error is None:
                try:
                    if sampled_nurbs is None:
                        sampled_nurbs = sample_nurbs_data(copy_stroke_curve_data(curves))
                    spline = sampled_nurbs[i]
                except (ValueError, RuntimeError, IndexError) as exc:
                    error = str(exc)
            if error is not None:
                spline.nurbs_error = error
            splines.append(spline)
            continue
        if spline.type in {'BEZIER', 'POLY', 'CATMULL_ROM'}:
            points = []
            start = curve.first_point_index
            for j in range(start, start + curve.points_length):
                co = curves.position_data[j].vector.copy()
                point = SimpleNamespace(co=co)
                if spline.type == 'BEZIER':
                    point.handle_left = (
                        handles_left.data[j].vector.copy()
                        if handles_left else co.copy()
                    )
                    point.handle_right = (
                        handles_right.data[j].vector.copy()
                        if handles_right else co.copy()
                    )
                points.append(point)
            if spline.type == 'CATMULL_ROM':
                # Blender uses uniform Catmull-Rom with duplicated open ends.
                # The equivalent cubic Bezier handles are one sixth of the
                # vector between each point's neighbors.
                for j, point in enumerate(points):
                    if len(points) == 2:
                        offset = (points[1].co - points[0].co) / 6.0
                        point.handle_left = point.co - offset
                        point.handle_right = point.co + offset
                    else:
                        previous = points[j - 1] if (
                            j > 0 or spline.use_cyclic_u
                        ) else point
                        following = points[(j + 1) % len(points)] if (
                            j + 1 < len(points) or spline.use_cyclic_u
                        ) else point
                        offset = (following.co - previous.co) / 6.0
                        point.handle_left = point.co - offset
                        point.handle_right = point.co + offset
                if len(points) == 2 and spline.use_cyclic_u:
                    # A two-point loop evaluates each direction separately.
                    points[0].handle_left = points[0].handle_right.copy()
                    points[1].handle_right = points[1].handle_left.copy()
                spline.type = 'BEZIER'
            if spline.type == 'BEZIER':
                spline.bezier_points = points
            else:
                spline.points = points
        splines.append(spline)

    return splines


def get_text_splines(obj, depsgraph):
    """Copy text outlines, then release Blender's temporary curve data."""
    try:
        outline = obj.to_curve(depsgraph, apply_modifiers=False)
        if outline is None:
            return []
        splines = []
        for source in outline.splines:
            spline = SimpleNamespace(
                type=source.type,
                use_cyclic_u=source.use_cyclic_u,
                character_index=source.character_index,
                points=[
                    SimpleNamespace(co=point.co.copy())
                    for point in source.points
                ],
                bezier_points=[
                    SimpleNamespace(
                        co=point.co.copy(),
                        handle_left=point.handle_left.copy(),
                        handle_right=point.handle_right.copy(),
                    )
                    for point in source.bezier_points
                ],
            )
            splines.append(spline)
        return splines
    finally:
        obj.to_curve_clear()


def get_surface_splines(obj):
    """Copy the original surface's tessellated wireframe, then release it."""
    try:
        mesh = obj.to_mesh()
        return get_mesh_splines(mesh) if mesh is not None else []
    finally:
        obj.to_mesh_clear()


def get_mesh_splines(mesh):
    """Partition wireframe edges into maximal chains and closed loops."""
    adjacency = [[] for _ in mesh.vertices]
    edges = [tuple(edge.vertices) for edge in mesh.edges]
    for index, (a, b) in enumerate(edges):
        adjacency[a].append(index)
        adjacency[b].append(index)
    used = set()
    splines = []

    def walk(start, edge_index):
        vertices = [start]
        current = start
        while edge_index not in used:
            used.add(edge_index)
            a, b = edges[edge_index]
            current = b if current == a else a
            vertices.append(current)
            if current == start or len(adjacency[current]) != 2:
                break
            edge_index = next(i for i in adjacency[current] if i not in used)
        cyclic = len(vertices) > 2 and vertices[-1] == start
        if cyclic:
            vertices.pop()
        splines.append(SimpleNamespace(
            type='POLY', use_cyclic_u=cyclic,
            points=[SimpleNamespace(co=mesh.vertices[i].co.copy()) for i in vertices],
        ))

    # Split at endpoints and branch vertices; never choose an arbitrary turn.
    for vertex, incident in enumerate(adjacency):
        if len(incident) != 2:
            for edge_index in incident:
                if edge_index not in used:
                    walk(vertex, edge_index)
    # Every remaining component is a degree-two cycle.
    for index, (a, b) in enumerate(edges):
        if index not in used:
            walk(a, index)
    return splines


def get_grease_pencil_splines(data, obj, depsgraph):
    """Copy visible current-frame stroke centerlines into object-space curves."""
    splines = []
    for layer in data.layers:
        node = layer
        visible = layer.opacity > 0.0
        while node is not None:
            visible = visible and not node.hide
            node = node.parent_group
        if not visible:
            continue
        frame = layer.current_frame()
        if frame is None or frame.drawing is None:
            continue
        drawing = frame.drawing
        offsets = []
        offset = 0
        for stroke in drawing.strokes:
            count = len(stroke.points)
            offsets.append(SimpleNamespace(first_point_index=offset, points_length=count))
            offset += count
        positions = drawing.attributes.get('position')
        if positions is None:
            continue
        adapted = get_evaluated_splines(SimpleNamespace(
            attributes=drawing.attributes, curves=offsets,
            position_data=positions.data,
        ))
        transform = layer.matrix_local.copy()
        if layer.parent is not None:
            parent = layer.parent.evaluated_get(depsgraph)
            parent_world = parent.matrix_world.copy()
            if parent.type == 'ARMATURE' and layer.parent_bone:
                bone = parent.pose.bones.get(layer.parent_bone)
                if bone is not None:
                    parent_world = parent_world @ bone.matrix
            transform = (obj.matrix_world.inverted_safe() @ parent_world
                         @ layer.matrix_parent_inverse @ transform)
        hidden = drawing.attributes.get('hide_stroke')
        fill_ids = drawing.attributes.get('fill_id')
        material_indices = drawing.attributes.get('material_index')
        for index, spline in enumerate(adapted):
            has_fill = fill_ids is not None and fill_ids.data[index].value != 0
            if fill_ids is None:
                # Blender 5.0 stores filled contours in Grease Pencil materials.
                material_index = (material_indices.data[index].value
                                  if material_indices is not None else 0)
                if 0 <= material_index < len(obj.material_slots):
                    material = obj.material_slots[material_index].material
                    settings = material.grease_pencil if material is not None else None
                    has_fill = settings is not None and settings.show_fill
            # hide_stroke suppresses the outline, not a visible fill contour.
            if hidden is not None and hidden.data[index].value and not has_fill:
                continue
            if has_fill:
                spline.use_cyclic_u = True
            if spline.type in {'BEZIER', 'POLY'}:
                points = spline.bezier_points if spline.type == 'BEZIER' else spline.points
                for point in points:
                    point.co = transform @ point.co
                    if spline.type == 'BEZIER':
                        point.handle_left = transform @ point.handle_left
                        point.handle_right = transform @ point.handle_right
            splines.append(spline)
    return splines


@contextmanager
def evaluated_export_objects(context, objects, apply_modifiers):
    """Prepare conversion and evaluation, copying disabled objects as needed."""
    temporary_objects = []
    temporary_collection = None
    try:
        needs_evaluation = [
            obj for obj in objects
            if (
                obj.type in {'FONT', 'CURVES', 'GREASEPENCIL'}
                or (apply_modifiers
                    and any(mod.show_viewport for mod in obj.modifiers))
            )
        ]
        evaluated = {}
        depsgraph = None
        if needs_evaluation:
            depsgraph = context.evaluated_depsgraph_get()
            sources = {}
            for obj in needs_evaluation:
                source = obj
                if not obj.evaluated_get(depsgraph).is_evaluated:
                    # Hidden or excluded objects may be absent from the graph.
                    if temporary_collection is None:
                        temporary_collection = bpy.data.collections.new(
                            "SVG Export Temporary"
                        )
                        context.scene.collection.children.link(
                            temporary_collection
                        )
                    source = obj.copy()
                    temporary_objects.append(source)
                    source.hide_viewport = False
                    temporary_collection.objects.link(source)
                sources[obj] = source

            if temporary_objects:
                context.view_layer.update()
                depsgraph = context.evaluated_depsgraph_get()
            evaluated = {
                obj: source.evaluated_get(depsgraph)
                for obj, source in sources.items()
            }
        yield evaluated, depsgraph
    finally:
        for obj in temporary_objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        if temporary_collection is not None:
            bpy.data.collections.remove(temporary_collection)


def get_non_curve_components(geometry, allow_grease_pencil=False, allow_mesh=False):
    """Describe evaluated components which cannot be written as SVG curves."""
    components = [
        label for name, label in (
            ('mesh', 'mesh'),
            ('pointcloud', 'point cloud'),
            ('volume', 'volume'),
            ('grease_pencil', 'Grease Pencil'),
        )
        if getattr(geometry, name) is not None
        and not (allow_grease_pencil and name == 'grease_pencil')
        and not (allow_mesh and name == 'mesh')
    ]
    instances = geometry.instances_pointcloud()
    if instances is not None and len(instances.points):
        components.append('instances (add a Realize Instances node)')
    return components


def get_export_curves(context, scope):
    """Collect Curve, Curves, Text, Grease Pencil, Surface, and Mesh objects from the requested scope."""
    if scope == 'VISIBLE':
        area = get_view3d_area(context)
        viewport = area.spaces.active if area is not None else None
        objects = (
            obj for obj in context.view_layer.objects
            if obj.visible_get(view_layer=context.view_layer, viewport=viewport)
        )
    elif scope == 'ACTIVE':
        active = context.view_layer.objects.active
        objects = (active,) if active is not None else ()
    elif scope == 'COLLECTION':
        collection = context.view_layer.active_layer_collection.collection
        objects = collection.all_objects
    elif scope == 'ALL':
        objects = context.scene.objects
    else:
        objects = context.selected_objects

    return [obj for obj in objects if obj.type in {'CURVE', 'CURVES', 'FONT', 'GREASEPENCIL', 'MESH', 'SURFACE'}]


def unique_svg_id(name, used_ids):
    """Return a valid, unique XML identifier for a group or path."""
    base = re.sub(r'[^A-Za-z0-9_.-]', '_', name)
    if not base or not (base[0].isalpha() or base[0] == '_'):
        base = 'id_' + base
    identifier = base
    suffix = 2
    while identifier in used_ids:
        identifier = f'{base}_{suffix}'
        suffix += 1
    used_ids.add(identifier)
    return identifier


def get_object_collection_paths(root, scene_collection):
    """Assign each object to its first collection path in the export subtree."""
    root_path = () if root == scene_collection else (root,)
    pending = [(root, root_path)]
    visited = set()
    paths = {}
    while pending:
        collection, path = pending.pop()
        if collection in visited:
            continue
        visited.add(collection)
        for obj in collection.objects:
            paths.setdefault(obj, path)
        for child in reversed(tuple(collection.children)):
            pending.append((child, (*path, child)))
    return paths


def merge_svg_bounds(a, b):
    if a is None:
        return b
    return (min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3]))


def group_svg_objects(svg_objects, grouping, used_ids, collection_paths,
                      center_data=None):
    """Nest exported paths without adding geometry from parents or collections."""
    def center_attribute(bounds):
        if center_data is None or bounds is None:
            return ''
        x = -(bounds[0] + bounds[2]) / 2
        y = -(bounds[1] + bounds[3]) / 2
        centered = (bounds[0] + x, bounds[1] + y,
                    bounds[2] + x, bounds[3] + y)
        center_data['bounds'] = merge_svg_bounds(center_data['bounds'], centered)
        return f' transform="translate({x:.6f}, {y:.6f})"'

    if grouping == 'OBJECT':
        content = []
        for obj, identifier, paths in svg_objects:
            transform = center_attribute(
                center_data['objects'].get(obj) if center_data else None
            )
            content.append(f'  <g id="{identifier}"{transform}>')
            content.extend(paths)
            content.append('  </g>')
        return content

    use_collections = grouping in {'COLLECTION', 'COLLECTION_HIERARCHY'}
    use_parents = grouping in {'HIERARCHY', 'COLLECTION_HIERARCHY'}

    def new_group(name):
        return SimpleNamespace(
            name=name, identifier=None, paths=[], children={}, bounds=None,
        )

    root = new_group('')
    for obj, identifier, paths in svg_objects:
        if not paths:
            continue
        ancestry = []
        if use_collections:
            ancestry.extend(
                ('COLLECTION', collection)
                for collection in collection_paths.get(obj, ())
            )
        if use_parents:
            parents = []
            parent = obj.parent
            visited = {obj}
            while parent is not None and parent not in visited:
                parents.append(parent)
                visited.add(parent)
                parent = parent.parent
            ancestry.extend(('OBJECT', parent) for parent in reversed(parents))
        ancestry.append(('OBJECT', obj))

        group = root
        for key in ancestry:
            if key not in group.children:
                group.children[key] = new_group(key[1].name)
            group = group.children[key]
            if center_data is not None:
                group.bounds = merge_svg_bounds(group.bounds, center_data['objects'][obj])
        group.identifier = identifier
        group.paths = paths

    content = []
    pending = [(group, 1) for group in reversed(tuple(root.children.values()))]
    while pending:
        group, depth = pending.pop()
        indent = '  ' * depth
        if group is None:
            content.append(f'{indent}</g>')
            continue
        identifier = group.identifier or unique_svg_id(group.name, used_ids)
        transform = center_attribute(group.bounds) if depth == 1 else ''
        content.append(f'{indent}<g id="{identifier}"{transform}>')
        content.append(f'{indent}  <title>{escape(group.name)}</title>')
        content.extend(f'{indent}  {path.lstrip()}' for path in group.paths)
        pending.append((None, depth))
        pending.extend(
            (child, depth + 1)
            for child in reversed(tuple(group.children.values()))
        )
    return content


def write_svg(context, filepath, props, warnings=None):
    """Iterate curve objects in the export scope and write the SVG file."""
    if warnings is None:
        warnings = []
    scale_factor = props.global_scale
    stroke_width_val = props.stroke_width
    stroke_style_attributes = ''
    for prop, attribute, default in (
        ('stroke_cap', 'stroke-linecap', 'BUTT'),
        ('stroke_join', 'stroke-linejoin', 'MITER'),
    ):
        value = getattr(props, prop, default)
        if value != default:
            stroke_style_attributes += f' {attribute}="{value.lower()}"'
    axis_mode = props.projection_axis
    center_mode = getattr(props, 'center_mode',
                          'ALL' if getattr(props, 'center_svg', True) else 'NONE')
    center_svg = center_mode != 'NONE'
    fixed_page = (getattr(props, 'page_size', 'CONTENT') != 'CONTENT'
                  and axis_mode != 'CAMERA')
    center_data = {'objects': {}, 'bounds': None} if center_mode == 'GROUPS' else None
    
    fill_mode = props.fill_settings
    fill_source = props.fill_color_source
    fill_user_col = props.user_fill_color
    
    stroke_source = props.stroke_color_source
    stroke_user_col = props.user_stroke_color

    if hasattr(props, '_export_curves'):
        curves = props._export_curves[props.export_scope]
    else:
        curves = get_export_curves(context, props.export_scope)

    if not curves:
        return {'NO_OBJECTS'}

    type_options = {
        'CURVE': 'include_curve', 'CURVES': 'include_curves',
        'FONT': 'include_text', 'GREASEPENCIL': 'include_grease_pencil',
        'MESH': 'include_mesh', 'SURFACE': 'include_surface',
    }
    curves = [obj for obj in curves
              if getattr(props, type_options[obj.type], True)]
    if not curves:
        return {'NO_INCLUDED_OBJECTS'}


    camera = None
    if axis_mode == 'CAMERA':
        try:
            camera = get_camera_projection(context)
        except (ValueError, RuntimeError) as error:
            warnings.append(str(error))
            return {'NO_CAMERA'}
        center_svg = False
        center_data = None
        axis_mode = 'TOP'

    view_matrix = None
    if axis_mode == 'VIEW':
        if hasattr(props, '_view_matrix'):
            view_matrix = props._view_matrix
        else:
            view_matrix = get_view_orientation_matrix(context)
        if view_matrix is None:
            return {'NO_VIEW'}
        axis_mode = 'TOP'

    svg_objects = []
    used_ids = set()
    bounds = None
    path_count = 0

    with evaluated_export_objects(
        context, curves, props.apply_modifiers
    ) as (evaluated_objects, depsgraph):
        for obj in curves:
            evaluated = evaluated_objects.get(obj)
            matrix = evaluated.matrix_world if evaluated else obj.matrix_world
            if (
                props.apply_modifiers
                and any(mod.show_viewport for mod in obj.modifiers)
            ):
                try:
                    geometry = evaluated.evaluated_geometry()
                except (RuntimeError, TypeError) as error:
                    warnings.append(
                        f"Skipped '{obj.name}': could not evaluate modifiers "
                        f"({error})"
                    )
                    continue
                is_grease_pencil = obj.type == 'GREASEPENCIL'
                non_curve_components = get_non_curve_components(
                    geometry, allow_grease_pencil=is_grease_pencil,
                    allow_mesh=obj.type in {'MESH', 'SURFACE'}
                )
                if non_curve_components:
                    warnings.append(
                        f"'{obj.name}': skipped non-curve modifier output "
                        f"({', '.join(non_curve_components)})"
                    )
                evaluated_curves = geometry.curves
                evaluated_strokes = geometry.grease_pencil if is_grease_pencil else None
                evaluated_mesh = geometry.mesh if obj.type in {'MESH', 'SURFACE'} else None
                if all(part is None for part in (
                    evaluated_curves, evaluated_strokes, evaluated_mesh
                )):
                    if not non_curve_components:
                        warnings.append(
                            f"Skipped '{obj.name}': modifier result is empty "
                            "(no splines)"
                        )
                    continue
                splines = (get_evaluated_splines(evaluated_curves)
                           if evaluated_curves is not None else [])
                if evaluated_strokes is not None:
                    splines.extend(get_grease_pencil_splines(
                        evaluated_strokes, evaluated, depsgraph
                    ))
                if evaluated_mesh is not None:
                    splines.extend(get_mesh_splines(evaluated_mesh))
            elif obj.type == 'SURFACE':
                try:
                    splines = get_surface_splines(obj)
                except (RuntimeError, ValueError) as error:
                    warnings.append(f"Skipped surface '{obj.name}': {error}")
                    continue
            elif obj.type == 'MESH':
                splines = get_mesh_splines(obj.data)
            elif obj.type == 'GREASEPENCIL':
                splines = get_grease_pencil_splines(obj.data, evaluated, depsgraph)
                if not props.apply_modifiers and any(
                    mod.type == 'LINEART' and mod.show_viewport for mod in obj.modifiers
                ):
                    warnings.append(
                        f"'{obj.name}': enable Apply Modifiers to export generated Line Art"
                    )
            elif obj.type == 'FONT':
                try:
                    splines = get_text_splines(evaluated, depsgraph)
                except (RuntimeError, TypeError) as error:
                    warnings.append(
                        f"Skipped text '{obj.name}': could not convert outlines "
                        f"({error})"
                    )
                    continue
            elif obj.type == 'CURVES':
                splines = get_evaluated_splines(obj.data)
            else:
                splines = obj.data.splines

            if not len(splines):
                warnings.append(f"Skipped empty object '{obj.name}': no splines")
                continue
            if view_matrix is not None:
                matrix = view_matrix @ matrix
            safe_obj_name = unique_svg_id(obj.name, used_ids)
            object_paths = []

            obj_fill_hex = resolve_color(obj, fill_source, fill_user_col)
            obj_stroke_hex = resolve_color(obj, stroke_source, stroke_user_col)
            empty_splines = []
            unsupported_types = set()
            text_paths = {}

            for i, spline in enumerate(splines):
                if spline.type == 'NURBS':
                    try:
                        if hasattr(spline, 'nurbs_error'):
                            raise ValueError(spline.nurbs_error)
                        spline = get_legacy_nurbs_spline(spline)
                    except (ValueError, RuntimeError) as error:
                        warnings.append(f"'{obj.name}', spline {i + 1}: skipped ({error})")
                        continue
                points = (
                    spline.bezier_points if spline.type == 'BEZIER'
                    else spline.points
                )
                if len(points) < 2:
                    empty_splines.append(i + 1)
                    continue
                spline_matrix = matrix
                if camera is not None and spline.type in {'BEZIER', 'POLY'}:
                    try:
                        spline, spline_matrix = camera_project_spline(
                            spline, matrix, camera,
                            getattr(props, 'camera_tolerance', 0.25) / scale_factor,
                        )
                    except ValueError as error:
                        warnings.append(f"'{obj.name}', spline {i + 1}: skipped ({error})")
                        continue
                if spline.type == 'BEZIER':
                    path_d = get_bezier_path_d(spline, spline_matrix, axis_mode)
                elif spline.type == 'POLY':
                    path_d = get_poly_path_d(spline, spline_matrix, axis_mode)
                else:
                    unsupported_types.add(spline.type)
                    continue

                path_count += 1
                if center_svg or fixed_page:
                    spline_bounds = get_spline_bounds(
                        spline, matrix, axis_mode
                    )
                    if center_data is not None:
                        center_data['objects'][obj] = merge_svg_bounds(
                            center_data['objects'].get(obj), spline_bounds
                        )
                    if bounds is None:
                        bounds = spline_bounds
                    else:
                        bounds = (
                            min(bounds[0], spline_bounds[0]),
                            min(bounds[1], spline_bounds[1]),
                            max(bounds[2], spline_bounds[2]),
                            max(bounds[3], spline_bounds[3]),
                        )

                final_fill = "none"
                should_fill = False
                is_cyclic = spline.use_cyclic_u

                if fill_mode == 'ALL':
                    should_fill = True
                elif fill_mode == 'CLOSED' and is_cyclic:
                    should_fill = True
                elif fill_mode == 'OBJECT':
                    is_2d = (getattr(obj.data, 'dimensions', None) == '2D')
                    has_fill = (getattr(obj.data, 'fill_mode', 'NONE') != 'NONE')
                    if is_2d and has_fill and is_cyclic:
                        should_fill = True
                
                if should_fill:
                    final_fill = obj_fill_hex
                
                final_stroke = obj_stroke_hex
                spline_id = unique_svg_id(f'{safe_obj_name}_{i}', used_ids)
                
                if obj.type == 'FONT':
                    # Keep each letter's contours together so counters (holes)
                    # are filled correctly. Evaluated geometry has no character
                    # indices; nonzero winding also handles overlapping letters.
                    key = (getattr(spline, 'character_index', None), final_fill)
                    if key not in text_paths:
                        text_paths[key] = (spline_id, [])
                    text_paths[key][1].append(path_d)
                else:
                    object_paths.append(
                        f'    <path id="{spline_id}" d="{path_d}" stroke="{final_stroke}" stroke-width="{stroke_width_val}" fill="{final_fill}" />'
                    )

            for (_, final_fill), (path_id, outlines) in text_paths.items():
                path_d = ' '.join(outlines)
                object_paths.append(
                    f'    <path id="{path_id}" d="{path_d}" '
                    f'stroke="{obj_stroke_hex}" '
                    f'stroke-width="{stroke_width_val}" fill="{final_fill}" '
                    'fill-rule="nonzero" />'
                )

            if empty_splines:
                indices = ', '.join(map(str, empty_splines[:5]))
                if len(empty_splines) > 5:
                    indices += ', ...'
                warnings.append(
                    f"'{obj.name}': skipped {len(empty_splines)} empty "
                    f"spline(s), fewer than two points (splines {indices})"
                )
            if unsupported_types:
                warnings.append(
                    f"'{obj.name}': skipped {', '.join(sorted(unsupported_types))} "
                    "splines (convert to Bezier or Poly first)"
                )
            svg_objects.append((obj, safe_obj_name, object_paths))

    if not path_count:
        return {'NO_GEOMETRY'}

    grouping = getattr(props, 'grouping', 'OBJECT')
    collection_paths = {}
    if grouping in {'COLLECTION', 'COLLECTION_HIERARCHY'}:
        scene_collection = (
            props._export_scene_collection
            if hasattr(props, '_export_scene_collection')
            else context.scene.collection
        )
        collection_root = scene_collection
        if props.export_scope == 'COLLECTION':
            collection_root = (
                props._export_collection
                if hasattr(props, '_export_collection')
                else context.view_layer.active_layer_collection.collection
            )
        collection_paths = get_object_collection_paths(
            collection_root, scene_collection
        )
    svg_content = group_svg_objects(
        svg_objects, grouping, used_ids, collection_paths, center_data
    )
    if center_data is not None:
        bounds = center_data['bounds']

    svg_tag = '<svg xmlns="http://www.w3.org/2000/svg" version="1.1">'
    transform = f'scale({scale_factor}, {-scale_factor})'
    if center_svg:
        if bounds is None:
            width = height = 1.0
            translate_x = translate_y = 0.0
        else:
            min_x, min_y, max_x, max_y = bounds
            # Allow stroke joins with SVG's default miter limit of 4.
            padding = 2.0 * stroke_width_val
            width = max(
                (max_x - min_x + 2.0 * padding) * scale_factor, 0.000001
            )
            height = max(
                (max_y - min_y + 2.0 * padding) * scale_factor, 0.000001
            )
            translate_x = width / 2.0 - (min_x + max_x) / 2.0 * scale_factor
            translate_y = height / 2.0 + (min_y + max_y) / 2.0 * scale_factor

        svg_tag = (
            '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
            f'width="{width:.6f}" height="{height:.6f}" '
            f'viewBox="0 0 {width:.6f} {height:.6f}">'
        )
        transform = (
            f'translate({translate_x:.6f}, {translate_y:.6f}) {transform}'
        )

    if fixed_page:
        page_size = props.page_size
        if page_size == 'CUSTOM':
            unit = getattr(props, 'page_unit', 'mm')
            factor = {'mm': 96 / 25.4, 'cm': 96 / 2.54, 'in': 96, 'px': 1}[unit]
            page_width = getattr(props, 'page_width', 210.0)
            page_height = getattr(props, 'page_height', 297.0)
        else:
            unit, factor = 'mm', 96 / 25.4
            page_width, page_height = (210.0, 297.0) if page_size == 'A4' else (215.9, 279.4)
            if getattr(props, 'page_orientation', 'PORTRAIT') == 'LANDSCAPE':
                page_width, page_height = page_height, page_width
        width, height = page_width * factor, page_height * factor
        min_x, min_y, max_x, max_y = bounds
        cx = (min_x + max_x) / 2 if center_svg else 0.0
        cy = (min_y + max_y) / 2 if center_svg else 0.0
        if getattr(props, 'fit_page', False):
            margin = getattr(props, 'page_margin', 10.0) * 96 / 25.4
            available_width, available_height = width - 2 * margin, height - 2 * margin
            if available_width <= 0 or available_height <= 0:
                warnings.append("Page margins must leave a positive drawing area")
                return {'INVALID_PAGE'}
            padding = 2.0 * stroke_width_val
            extent_x = max(abs(min_x - cx), abs(max_x - cx)) + padding
            extent_y = max(abs(min_y - cy), abs(max_y - cy)) + padding
            scale_factor = min(available_width / (2 * extent_x),
                               available_height / (2 * extent_y))
        svg_tag = (
            '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
            f'width="{page_width:.6f}{unit}" height="{page_height:.6f}{unit}" '
            f'viewBox="{-width / 2:.6f} {-height / 2:.6f} {width:.6f} {height:.6f}" '
            'overflow="visible">'
        )
        transform = (f'scale({scale_factor}, {-scale_factor}) '
                     f'translate({-cx:.6f}, {-cy:.6f})')

    camera_defs = []
    clip_attribute = ''
    if camera is not None:
        width = camera.width * scale_factor
        height = camera.height * scale_factor
        svg_tag = (
            '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" '
            f'width="{width:.6f}" height="{height:.6f}" '
            f'viewBox="0 0 {width:.6f} {height:.6f}" overflow="visible">'
        )
        transform = f'translate(0, {height:.6f}) {transform}'
        if getattr(props, 'clip_camera', True):
            clip_id = unique_svg_id('camera_frame', used_ids)
            camera_defs = [
                f'<defs><clipPath id="{clip_id}" clipPathUnits="userSpaceOnUse">'
                f'<rect x="0" y="0" width="{camera.width:.6f}" '
                f'height="{camera.height:.6f}" /></clipPath></defs>'
            ]
            clip_attribute = f' clip-path="url(#{clip_id})"'

    svg_header = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        svg_tag,
        *camera_defs,
        f'<g transform="{transform}"{clip_attribute}{stroke_style_attributes}>',
    ]
    svg_content = svg_header + svg_content
    svg_content.append('</g>')
    svg_content.append('</svg>')

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(svg_content))
            
        if warnings:
            return {'FINISHED_WITH_WARNING'}
        return {'FINISHED'}
        
    except Exception as e:
        print(f"Export Error: {e}")
        return {'FILE_ERROR'}

# --- 4. Operator Class ---

STROKE_STYLES = {
    'cap': (
        "Caps",
        "Choose the stroke end shape on open paths. Caps affect only the SVG "
        "stroke and extend according to Stroke Width; closed paths have no "
        "caps. The source curve geometry is unchanged.",
        (
            (
                'BUTT', "Butt",
                "End the stroke flat at each open endpoint, without extending "
                "past it. Use when the visible line should end exactly at the "
                "source endpoints. Closed paths are unaffected.",
            ),
            (
                'ROUND', "Round",
                "Add a semicircular end extending half the stroke width beyond "
                "each open endpoint. Use for soft ends on line drawings and "
                "strokes. Closed paths are unaffected.",
            ),
            (
                'SQUARE', "Square",
                "Add a flat square end extending half the stroke width beyond "
                "each open endpoint. Use for extended flat ends. Closed paths "
                "are unaffected.",
            ),
        ),
    ),
    'join': (
        "Joins",
        "Choose the stroke corner shape where SVG path segments meet, "
        "including the closing corner of closed paths. Affects the stroke "
        "outline; the path centerline and fill shape stay unchanged.",
        (
            (
                'MITER', "Miter",
                "Extend adjoining stroke edges into a sharp corner. Very acute "
                "corners fall back to a bevel when they exceed the SVG default "
                "miter limit of 4. Use for pointed geometric corners.",
            ),
            (
                'ROUND', "Round",
                "Join adjoining stroke edges with a rounded outer corner. Use "
                "to soften angles on polygonal paths; the source path and its "
                "fill shape keep their original corners.",
            ),
            (
                'BEVEL', "Bevel",
                "Trim the outer stroke corner with a straight edge between "
                "adjoining segments. Use for flat-cut corners without long "
                "spikes; the source path and fill shape are unchanged.",
            ),
        ),
    ),
}
_stroke_icons = {}


def stroke_style_property(style):
    name, description, items = STROKE_STYLES[style]
    return EnumProperty(
        name=name, description=description,
        items=[
            (*item, _stroke_icons.get((style, item[0]), 0), index)
            for index, item in enumerate(items)
        ],
        default=items[0][0],
    )


def register_stroke_icons():
    """Load bundled vector icons without writing to the extension directory."""
    directory = os.path.join(os.path.dirname(__file__), "icons")
    try:
        for style, (_, _, items) in STROKE_STYLES.items():
            for value, _, _ in items:
                path = os.path.join(directory, f'{style}_{value.lower()}.dat')
                icon_id = bpy.app.icons.new_triangles_from_file(path)
                _stroke_icons[style, value] = icon_id
    except Exception:
        unregister_stroke_icons()
        raise


def unregister_stroke_icons():
    for icon_id in _stroke_icons.values():
        bpy.app.icons.release(icon_id)
    _stroke_icons.clear()


class CURVE_OT_export_svg(bpy.types.Operator, ExportHelper):
    """Export Curve, Curves, Text, Grease Pencil, Surface, and Mesh objects as SVG"""
    bl_idname = "export_curve.svg"
    bl_label = "Export SVG"
    bl_description = (
        "Export enabled object types as SVG paths. Use Include to choose "
        "objects, Layout to project and position them, and Style to set fills "
        "and strokes. Conversions and modifier evaluation are temporary; "
        "warnings identify skipped or unsupported geometry."
    )
    filename_ext = ".svg"
    
    filter_glob: StringProperty(
        default="*.svg",
        options={'HIDDEN'},
        maxlen=255,
        description=(
            "Show SVG files in the file browser when choosing an export "
            "destination."
        ),
    )

    include_curve: BoolProperty(
        name="Curve",
        default=True,
        description=(
            "Include legacy Curve objects from the chosen Limit to scope. "
            "Bezier and Poly splines export directly; NURBS is sampled at "
            "Blender curve resolution. Increase that resolution for smoother "
            "NURBS output. Bevel and curve thickness are omitted; Style "
            "controls the SVG stroke width. Empty or single-point splines are "
            "skipped with a warning."
        ),
    )

    include_curves: BoolProperty(
        name="Curves",
        default=True,
        description=(
            "Include Curves objects, including hair curves, from the chosen "
            "Limit to scope. Bezier and Poly export directly, Catmull-Rom "
            "becomes cubic Bezier, and NURBS is sampled at Blender curve "
            "resolution. Curve radius and hair shading are omitted; Style "
            "supplies uniform width and colors. Geometry Nodes instances must "
            "be realized before export."
        ),
    )

    include_text: BoolProperty(
        name="Text",
        default=True,
        description=(
            "Export Text objects as temporary glyph outlines, including holes "
            "in letters such as O and B. SVG letters are paths; the original "
            "Blender text and fonts stay editable. Extrusion and bevel are "
            "omitted. With Apply Modifiers enabled, text that evaluates only "
            "to a mesh is skipped; disable it to export the original outlines."
        ),
    )

    include_grease_pencil: BoolProperty(
        name="Grease Pencil",
        default=True,
        description=(
            "Export visible layers at the current frame as stroke centerlines "
            "and fill contours. Enable Apply Modifiers to include generated "
            "Line Art. Fill contours follow the SVG Fill mode. Single-point "
            "strokes are skipped. SVG uses the chosen colors and uniform "
            "width; pressure, textures, masks, holdouts, effects and layer "
            "blending are omitted."
        ),
    )

    include_surface: BoolProperty(
        name="Surface",
        default=True,
        description=(
            "Export NURBS Surface objects as tessellated wireframes. Increase "
            "Blender U/V resolution for finer sampling and larger SVG files; "
            "Apply Modifiers includes supported evaluated geometry. All "
            "tessellation edges are included, even internal and hidden edges. "
            "Filled surface patches, silhouettes and hidden-line removal are "
            "not generated. Original surfaces stay editable."
        ),
    )

    include_mesh: BoolProperty(
        name="Mesh",
        default=False,
        description=(
            "Export all mesh edges, including loose and hidden edges, as "
            "straight SVG paths. Continuous chains join through vertices with "
            "two edges; paths split at branches and closed chains form loops. "
            "Enable Apply Modifiers for the evaluated wireframe. Isolated "
            "vertices and filled faces are omitted; use Line Art for visible "
            "outlines."
        ),
    )

    export_scope: EnumProperty(
        name="Limit to",
        items=[
            (
                'VISIBLE', "Visible",
                "Export enabled object types visible in the source 3D "
                "viewport, respecting viewport visibility and Local View. If "
                "no viewport is available, use current view-layer visibility. "
                "Render visibility is not used."
            ),
            (
                'ACTIVE', "Active Object",
                "Export only the active object if its type is enabled. "
                "Selecting several objects still exports only the active one. "
                "Set the active object before opening this dialog."
            ),
            (
                'SELECTED', "Selected Objects",
                "Export selected objects whose types are enabled in Include. "
                "Select them before opening this dialog; changing the scene "
                "selection while the dialog is open does not update the "
                "captured list."
            ),
            (
                'COLLECTION', "Active Collection",
                "Export enabled object types from the active collection and "
                "all its child collections, including hidden or excluded "
                "objects. Set the active collection before opening this "
                "dialog."
            ),
            (
                'ALL', "All",
                "Export every object of the enabled types in the current "
                "scene, including hidden and excluded objects. Objects "
                "belonging only to other scenes are not included."
            ),
        ],
        default='SELECTED',
        description=(
            "Choose which objects are considered, then filter them with the "
            "enabled Object Types. Object lists are captured when opening the "
            "export dialog. Reopen the dialog after changing selection, "
            "visibility or the active collection."
        ),
    )

    grouping: EnumProperty(
        name="Grouping",
        items=[
            (
                'OBJECT', "Objects",
                "Create one named SVG group per exported object and keep the "
                "existing object order. With Re-center Groups, each object "
                "group is centered independently."
            ),
            (
                'COLLECTION', "Collections",
                "Reproduce nested collection groups containing the exported "
                "object groups. Objects linked to multiple collections use the "
                "first matching collection; empty groups are omitted."
            ),
            (
                'HIERARCHY', "Object Hierarchy",
                "Nest SVG groups along Blender object-parent chains. Parents "
                "outside the export scope appear as structure-only groups; "
                "their geometry is included only if allowed by Include."
            ),
            (
                'COLLECTION_HIERARCHY', "Collections + Hierarchy",
                "Follow the collection tree, then object parenting within each "
                "collection. Cross-collection parents may appear as "
                "structure-only groups. Multi-collection objects use the first "
                "matching collection, and each object geometry is exported "
                "once."
            ),
        ],
        default='OBJECT',
        description=(
            "Organize exported paths into named SVG groups for easier "
            "selection in a vector editor. Object and collection names are "
            "retained. Nesting can change the stacking order of overlapping "
            "objects. Parents used for structure do not add geometry outside "
            "the Include settings."
        ),
    )

    apply_modifiers: BoolProperty(
        name="Apply Modifiers",
        default=False,
        description=(
            "Export the result of viewport-enabled modifiers without applying "
            "or removing them in Blender. Enable this for generated Line Art. "
            "Curve, Curves and Text sources export curve components; mesh-only "
            "results from these sources are skipped with a warning. Mesh and "
            "Surface sources support evaluated wireframes. Unsupported "
            "components are skipped; Geometry Nodes instances require Realize "
            "Instances."
        ),
    )

    # --- Properties: Projection ---
    global_scale: FloatProperty(
        name="Scale",
        default=100.0,
        min=0.01,
        description=(
            "Multiply artwork coordinates and stroke width by this value. "
            "For Top, Front, Side and Current View, Scale 100 gives 100 SVG "
            "pixels per Blender coordinate unit; scene unit scale is not "
            "used. In Camera mode, Scale 100 makes page width match the "
            "full render resolution. Fit to Page overrides this manual "
            "scale."
        ),
    )
    
    projection_axis: EnumProperty(
        name="Projection View",
        items=[
            (
                'TOP', "Top (XY)",
                "Use world X horizontally and Y vertically, discarding Z "
                "depth. Produces an orthographic plan view and keeps Bezier "
                "segments as cubic curves."
            ),
            (
                'FRONT', "Front (XZ)",
                "Use world X horizontally and Z vertically, discarding Y "
                "depth. Produces an orthographic elevation and keeps Bezier "
                "segments as cubic curves."
            ),
            (
                'SIDE', "Side (YZ)",
                "Use world Y horizontally and Z vertically, discarding X "
                "depth. Produces an orthographic side view and keeps Bezier "
                "segments as cubic curves."
            ),
            (
                'CAMERA', "Camera",
                "Use the scene's active perspective or orthographic camera, "
                "including its evaluated transform, lens, shift, sensor fit "
                "and render pixel aspect. Panoramic cameras are unsupported. "
                "Perspective splines crossing or behind the camera plane are "
                "skipped with a warning."
            ),
            (
                'VIEW', "Current View",
                "Use the source 3D viewport rotation captured when opening the "
                "export dialog. Projection is orthographic; viewport "
                "perspective, zoom and pan are ignored. When launched outside "
                "a viewport, use the largest 3D viewport in the current "
                "window."
            ),
        ],
        default='TOP',
        description=(
            "Choose how 3D geometry is projected onto the SVG page. Top, "
            "Front, Side and Current View use orthographic projection; Camera "
            "follows the scene camera. Object transforms are respected. "
            "Projection alone does not remove hidden edges or occluded paths."
        ),
    )

    clip_camera: BoolProperty(
        name="Clip to Camera Frame",
        default=True,
        description=(
            "Hide geometry outside the camera rectangle using an SVG clipping "
            "mask. Disable this to keep outside-frame paths visible while "
            "retaining the camera page size. This does not remove occluded "
            "geometry or enforce camera near/far clipping distances."
        ),
    )

    camera_tolerance: FloatProperty(
        name="Curve Accuracy",
        default=0.25,
        min=0.01,
        max=10.0,
        description=(
            "Tolerance in final SVG pixels when approximating perspective "
            "Bezier curves with straight segments. Lower values make curves "
            "smoother, increase SVG size and take longer to export; 0.25 "
            "targets a quarter-pixel deviation. This does not increase NURBS "
            "or surface sampling resolution; adjust their Blender resolution "
            "instead."
        ),
    )

    page_size: EnumProperty(
        name="Size",
        default='CONTENT',
        items=[
            (
                'CONTENT', "Fit to Content",
                "With Re-center On or Groups, size the page to the artwork "
                "bounds plus stroke padding. With Re-center Off, preserve "
                "unframed coordinates without calculating page dimensions. "
                "Fixed page fitting and margins are not used."
            ),
            (
                'A4', "A4",
                "Set A4 paper dimensions: 210 by 297 mm in portrait. "
                "Orientation can swap width and height. Enable Fit to Page to "
                "resize the artwork inside margins; choosing this preset alone "
                "retains manual artwork scale."
            ),
            (
                'LETTER', "Letter",
                "Set US Letter dimensions: 8.5 by 11 inches in portrait. "
                "Orientation can swap width and height. Enable Fit to Page to "
                "resize the artwork inside margins; choosing this preset alone "
                "retains manual artwork scale."
            ),
            (
                'CUSTOM', "Custom",
                "Set the page Width and Height in the selected Units. Enable "
                "Fit to Page to fit the artwork inside margins. Changing "
                "dimensions alone retains manual artwork scale, and geometry "
                "outside the page is retained."
            ),
        ],
        description=(
            "Choose the SVG page dimensions. Fixed paper sizes retain manual "
            "artwork scale unless Fit to Page is enabled. Geometry outside a "
            "fixed page is retained. Camera projection uses its camera frame "
            "instead and shows From Camera."
        ),
    )
    camera_page_size: EnumProperty(
        name="Page Size",
        items=[
            (
                'CAMERA', "From Camera",
                "Page size is controlled by the active camera, so this "
                "dropdown is unavailable. At Scale 100, width matches the full "
                "render resolution and height accounts for render pixel "
                "aspect. Render resolution percentage and render border "
                "cropping are ignored. Change the camera/render settings or "
                "Scale to change the output frame."
            ),
        ],
        get=lambda self: 0,
        options={'HIDDEN', 'SKIP_SAVE'},
        description=(
            "Page size is controlled by the active camera, so this dropdown is "
            "unavailable. At Scale 100, width matches the full render "
            "resolution and height accounts for render pixel aspect. Render "
            "resolution percentage and render border cropping are ignored. "
            "Change the camera/render settings or Scale to change the output "
            "frame."
        ),
    )
    page_orientation: EnumProperty(
        name="Orientation",
        default='PORTRAIT',
        items=[
            (
                'PORTRAIT', "Portrait",
                "Use the preset portrait dimensions, with the shorter page "
                "edge as Width. Changes page dimensions; the artwork keeps its "
                "orientation."
            ),
            (
                'LANDSCAPE', "Landscape",
                "Swap the preset width and height so the longer page edge is "
                "horizontal. Changes page dimensions; the artwork keeps its "
                "orientation."
            ),
        ],
        description=(
            "Choose portrait or landscape dimensions for A4 and Letter. "
            "Landscape swaps the preset width and height; it does not rotate "
            "the artwork. For a Custom page, set Width and Height directly."
        ),
    )
    page_unit: EnumProperty(
        name="Units",
        default='mm',
        items=[
            (
                'mm', "Millimeters",
                "Enter Custom page Width and Height in millimeters. 25.4 mm "
                "equals one inch; the SVG stores physical page dimensions."
            ),
            (
                'cm', "Centimeters",
                "Enter Custom page Width and Height in centimeters. One "
                "centimeter equals 10 mm; the SVG stores physical page "
                "dimensions."
            ),
            (
                'in', "Inches",
                "Enter Custom page Width and Height in inches. One inch equals "
                "25.4 mm and 96 SVG pixels."
            ),
            (
                'px', "Pixels",
                "Enter Custom page Width and Height in SVG pixels at 96 pixels "
                "per inch. The SVG remains vector artwork and can be scaled "
                "when viewed or printed."
            ),
        ],
        description=(
            "Units for the Custom page Width and Height. Physical sizes use 96 "
            "SVG pixels per inch. Changing this setting reinterprets the "
            "existing numbers in the new unit; it does not convert the numbers "
            "automatically. Artwork scale changes only when you adjust Scale "
            "or enable Fit to Page."
        ),
    )
    page_width: FloatProperty(
        name="Width",
        default=210.0,
        min=0.001,
        description=(
            "Custom page width in the selected Units. Sets the SVG canvas "
            "width. Manual artwork scale is retained unless Fit to Page is "
            "enabled. For landscape, enter a width larger than Height."
        ),
    )
    page_height: FloatProperty(
        name="Height",
        default=297.0,
        min=0.001,
        description=(
            "Custom page height in the selected Units. Sets the SVG canvas "
            "height. Manual artwork scale is retained unless Fit to Page is "
            "enabled. For portrait, enter a height larger than Width."
        ),
    )
    fit_page: BoolProperty(
        name="Fit to Page",
        default=False,
        description=(
            "Scale the artwork proportionally to fit inside the page Margin, "
            "including stroke padding. Overrides manual Scale; stroke width "
            "scales with the artwork. With Re-center Off, fitting scales about "
            "the original origin and accounts for offsets. Available for A4, "
            "Letter and Custom pages."
        ),
    )
    page_margin: FloatProperty(
        name="Margin (mm)",
        default=10.0,
        min=0.0,
        description=(
            "Minimum space between the artwork, including stroke padding, and "
            "each page edge when Fit to Page is enabled. Always measured in "
            "millimeters, regardless of Custom page Units. Larger margins make "
            "the fitted artwork smaller. Export is cancelled if the margins "
            "leave no drawing area."
        ),
    )

    center_mode: EnumProperty(
        name="Re-center",
        items=[
            (
                'NONE', "Off",
                "Keep the original projected placement. Fixed pages put the "
                "original origin at the page center; Fit to Content leaves the "
                "SVG unframed. Fit to Page scales about that origin and "
                "preserves the offset proportionally."
            ),
            (
                'ALL', "On",
                "Center the complete artwork using its projected bounds while "
                "preserving relative positions between objects and groups. Fit "
                "to Content includes stroke padding around the combined "
                "result."
            ),
            (
                'GROUPS', "Groups",
                "Move each top-level SVG group to the same page center while "
                "preserving the arrangement of its children. Group by "
                "determines which groups are centered. Independent groups can "
                "overlap."
            ),
        ],
        default='ALL',
        description=(
            "Choose how projected artwork is positioned on the page. On "
            "centers the complete export; Groups centers each top-level SVG "
            "group independently. Group by determines which groups are moved. "
            "Camera projection keeps its camera framing and ignores this "
            "setting."
        ),
    )

    # --- Properties: Stroke ---
    stroke_width: FloatProperty(
        name="Width",
        default=0.05,
        min=0.001,
        description=(
            "Uniform width for every exported stroke, measured before Scale is "
            "applied. For example, Width 0.05 at Scale 100 produces a 5-pixel "
            "SVG stroke. Fit to Page changes the effective scale. Blender "
            "bevel, curve radius and Grease Pencil pressure do not determine "
            "this width."
        ),
    )

    stroke_cap: stroke_style_property('cap')
    stroke_join: stroke_style_property('join')
    
    stroke_color_source: EnumProperty(
        name="Color",
        items=[
            (
                'CUSTOM', "Custom",
                "Use the color picker below for every exported object. Object "
                "and material colors do not affect this choice."
            ),
            (
                'OBJECT', "Object",
                "Use each object's Viewport Display Color for all its paths. Set "
                "it in Object Properties before export. Uses RGB only; object "
                "alpha is ignored."
            ),
            (
                'MATERIAL', "Material",
                "Use the active material, falling back to the first material, "
                "for the whole object. Reads the first Principled/Diffuse base "
                "color, or the material viewport color when unavailable; "
                "objects without a material use black. Textures, shader "
                "effects, transparency and per-face or per-stroke colors are "
                "not reproduced."
            ),
            (
                'RANDOM', "Random",
                "Generate one fresh color per object on every export. Stroke "
                "and Fill generate their colors independently, so they can "
                "differ and repeated exports can produce different results."
            ),
        ],
        default='CUSTOM',
        description=(
            "Choose one stroke color per exported object, applied to all its "
            "paths. Custom uses the picker below; Object and Material read "
            "Blender properties; Random generates new colors on each export. "
            "Colors are opaque RGB; object and material alpha are not "
            "exported."
        ),
    )
    user_stroke_color: FloatVectorProperty(
        name="Color",
        subtype='COLOR',
        default=(0.0, 0.0, 0.0),
        min=0.0,
        max=1.0,
        description=(
            "Choose the stroke color used for all exported objects when Color "
            "Source is Custom. Applies to outlines and open paths, including "
            "text and wireframes. This picker is ignored when another color "
            "source is selected."
        ),
    )

    # --- Properties: Fill ---
    fill_settings: EnumProperty(
        name="Mode",
        items=[
            (
                'NONE', "None",
                "Leave all path interiors unfilled and draw only their "
                "strokes. Use for outlines, line drawings and wireframes. Fill "
                "color settings have no effect."
            ),
            (
                'CLOSED', "Closed Curves",
                "Fill only paths marked closed or cyclic, including closed "
                "text outlines and Grease Pencil fill contours. Touching "
                "endpoints on an open spline do not count as closed. Closed "
                "wireframe loops are filled as path shapes, not mesh faces."
            ),
            (
                'OBJECT', "Filled 2D Curves",
                "Fill closed paths when the source Curve or Text data is set "
                "to 2D with a Blender Fill Mode enabled. Objects without these "
                "2D fill settings need Closed Curves or All to receive SVG "
                "fills."
            ),
            (
                'ALL', "All",
                "Fill every path, including open paths. SVG implicitly joins "
                "the endpoints with a straight edge for the fill, which can "
                "create unexpected polygons. The stroke still follows the "
                "original open path."
            ),
        ],
        default='NONE',
        description=(
            "Choose which projected path interiors receive a fill. This fills "
            "path shapes, not 3D surface or mesh faces. Text outlines preserve "
            "holes between contours; other separate contours are filled "
            "independently. Choose None for line drawings, or Closed Curves to "
            "fill explicitly closed paths."
        ),
    )
    fill_color_source: EnumProperty(
        name="Color",
        items=[
            (
                'CUSTOM', "Custom",
                "Use the color picker below for every exported object. Object "
                "and material colors do not affect this choice."
            ),
            (
                'OBJECT', "Object",
                "Use each object's Viewport Display Color for all its paths. Set "
                "it in Object Properties before export. Uses RGB only; object "
                "alpha is ignored."
            ),
            (
                'MATERIAL', "Material",
                "Use the active material, falling back to the first material, "
                "for the whole object. Reads the first Principled/Diffuse base "
                "color, or the material viewport color when unavailable; "
                "objects without a material use black. Textures, shader "
                "effects, transparency and per-face or per-stroke colors are "
                "not reproduced."
            ),
            (
                'RANDOM', "Random",
                "Generate one fresh color per object on every export. Stroke "
                "and Fill generate their colors independently, so they can "
                "differ and repeated exports can produce different results."
            ),
        ],
        default='CUSTOM',
        description=(
            "Choose one fill color per exported object, applied to its filled "
            "paths. Custom uses the picker below; Object and Material read "
            "Blender properties; Random generates new colors on each export. "
            "Only used when Fill Mode allows a fill. Object and material alpha "
            "are not exported."
        ),
    )
    user_fill_color: FloatVectorProperty(
        name="Color",
        subtype='COLOR',
        default=(0.0, 0.0, 0.0),
        min=0.0,
        max=1.0,
        description=(
            "Choose the interior color for all filled paths when Color Source "
            "is Custom. Fill Mode determines which paths receive it. This "
            "color has no effect with Fill Mode None or when another fill "
            "color source is selected."
        ),
    )

    def invoke(self, context, event):
        self._view_matrix = get_view_orientation_matrix(context)
        self._export_scene_collection = context.scene.collection
        self._export_collection = (
            context.view_layer.active_layer_collection.collection
        )
        # Capture every scope so it can be changed inside the file browser.
        self._export_curves = {
            scope: get_export_curves(context, scope)
            for scope in ('VISIBLE', 'ACTIVE', 'SELECTED', 'COLLECTION', 'ALL')
        }
        return ExportHelper.invoke(self, context, event)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        camera_view = self.projection_axis == 'CAMERA'

        options = layout.column()
        options.use_property_split = False
        options.prop(self, "apply_modifiers")
        layout.separator(factor=0.5)

        header, body = layout.panel("svg_include", default_closed=False)
        header.label(text="Include")
        if body is not None:
            body.label(text="Object Types")
            grid = body.grid_flow(columns=2, even_columns=True, align=True)
            grid.use_property_split = False
            for name in ("include_curve", "include_curves", "include_text",
                         "include_grease_pencil", "include_surface", "include_mesh"):
                grid.prop(self, name)
            body.separator(factor=0.5)
            body.prop(self, "export_scope")

        header, body = layout.panel("svg_layout", default_closed=False)
        header.label(text="Layout")
        if body is not None:
            body.prop(self, "projection_axis", text="Projection")
            if camera_view:
                body.prop(self, "clip_camera", text="Clip to Frame")
                camera = context.scene.camera
                if camera is not None and camera.data.type == 'PERSP':
                    body.prop(self, "camera_tolerance", text="Accuracy")

            body.separator(factor=0.5)
            body.prop(self, "grouping", text="Group by")
            if not camera_view:
                body.prop(self, "center_mode")

            body.separator(factor=0.75)
            if camera_view:
                page = body.row()
                page.enabled = False
                page.prop(self, "camera_page_size")
                body.prop(self, "global_scale")
            else:
                body.prop(self, "page_size", text="Page Size")
                fixed_page = self.page_size != 'CONTENT'
                if self.page_size in {'A4', 'LETTER'}:
                    body.prop(self, "page_orientation")
                elif self.page_size == 'CUSTOM':
                    body.prop(self, "page_unit")
                    column = body.column(align=True)
                    column.prop(self, "page_width")
                    column.prop(self, "page_height")
                if fixed_page:
                    body.prop(self, "fit_page")
                if fixed_page and self.fit_page:
                    body.prop(self, "page_margin")
                else:
                    body.prop(self, "global_scale")

        header, body = layout.panel("svg_appearance", default_closed=True)
        header.label(text="Style")
        if body is not None:
            header, fill = body.panel("svg_fill", default_closed=False)
            header.label(text="Fill")
            if fill is not None:
                fill.prop(self, "fill_settings")
                if self.fill_settings != 'NONE':
                    fill.prop(self, "fill_color_source", text="Color Source")
                    if self.fill_color_source == 'CUSTOM':
                        fill.prop(self, "user_fill_color", text="Color")

            header, stroke = body.panel("svg_stroke", default_closed=False)
            header.label(text="Stroke")
            if stroke is not None:
                stroke.prop(self, "stroke_color_source", text="Color Source")
                if self.stroke_color_source == 'CUSTOM':
                    stroke.prop(self, "user_stroke_color", text="Color")
                stroke.prop(self, "stroke_width")
                for prop, label in (("stroke_cap", "Caps"), ("stroke_join", "Joins")):
                    split = stroke.split(factor=0.4)
                    split.use_property_split = False
                    heading = split.row()
                    heading.alignment = 'RIGHT'
                    heading.label(text=label)
                    choice = split.row(align=True)
                    choice.template_icon_view(
                        self, prop, show_labels=True, scale=2.0, scale_popup=3.0,
                    )
                    items = self.properties.bl_rna.properties[prop].enum_items
                    selected = items[getattr(self, prop)]
                    name = choice.row()
                    name.scale_y = 2.0
                    name.label(text=selected.name)

    def execute(self, context):
        warnings = []
        result = write_svg(context, self.filepath, self, warnings)
        
        if result == {'FINISHED'}:
            self.report({'INFO'}, f"SVG Export Successful: {self.filepath}")
            return {'FINISHED'}
        elif result == {'FINISHED_WITH_WARNING'}:
            self.report(
                {'WARNING'}, "SVG exported with warnings: " + "; ".join(warnings)
            )
            return {'FINISHED'}
        elif result == {'NO_GEOMETRY'}:
            self.report(
                {'WARNING'}, "Nothing exported: " + "; ".join(warnings)
            )
            return {'CANCELLED'}
        elif result == {'NO_INCLUDED_OBJECTS'}:
            self.report({'ERROR'}, "No objects match the enabled types in Include.")
            return {'CANCELLED'}
        elif result == {'NO_OBJECTS'}:
            messages = {
                'VISIBLE': "No visible Curve, Curves, Text, Grease Pencil, Surface, or Mesh objects to export.",
                'ACTIVE': "The active object must be a Curve, Curves, Text, Grease Pencil, Surface, or Mesh object.",
                'SELECTED': "No Curve, Curves, Text, Grease Pencil, Surface, or Mesh objects selected.",
                'COLLECTION': "No Curve, Curves, Text, Grease Pencil, Surface, or Mesh objects in the active collection.",
                'ALL': "No Curve, Curves, Text, Grease Pencil, Surface, or Mesh objects in the current scene.",
            }
            self.report({'ERROR'}, messages[self.export_scope])
            return {'CANCELLED'}
        elif result == {'INVALID_PAGE'}:
            self.report({'ERROR'}, '; '.join(warnings))
            return {'CANCELLED'}
        elif result == {'NO_CAMERA'}:
            self.report({'ERROR'}, '; '.join(warnings))
            return {'CANCELLED'}
        elif result == {'NO_VIEW'}:
            self.report(
                {'ERROR'},
                "No 3D viewport available. Open export from a 3D Viewport.",
            )
            return {'CANCELLED'}
        elif result == {'FILE_ERROR'}:
            self.report({'ERROR'}, "Could not write to file. Check permissions.")
            return {'CANCELLED'}
            
        return {'CANCELLED'}

# --- 5. Registration ---

def menu_func_export(self, context):
    self.layout.operator(CURVE_OT_export_svg.bl_idname, text="Curve as SVG (.svg)")

classes = (CURVE_OT_export_svg,)

def register():
    register_stroke_icons()
    # Bind the fixed enum items after their runtime icon IDs are available.
    for style in STROKE_STYLES:
        prop = stroke_style_property(style)
        CURVE_OT_export_svg.__annotations__['stroke_' + style] = prop
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    unregister_stroke_icons()
