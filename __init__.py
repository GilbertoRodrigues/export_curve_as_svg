import bpy
import os
import random
from bpy_extras.io_utils import ExportHelper
from bpy.props import StringProperty, FloatProperty, EnumProperty, FloatVectorProperty

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

def get_smart_material_color(obj):
    """Attempts to retrieve the visual color of an object."""
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
    """Determines the final Hex color string based on the user's selection."""
    if source_type == 'CUSTOM':
        return rgb_to_hex(user_color)
    elif source_type == 'OBJECT':
        return rgb_to_hex(obj.color[:3])
    elif source_type == 'MATERIAL':
        raw_color = get_smart_material_color(obj)
        if raw_color:
            return rgb_to_hex(raw_color)
        else:
            return "#000000"
    elif source_type == 'RANDOM':
        rand_rgb = (random.random(), random.random(), random.random())
        return rgb_to_hex(rand_rgb)
    return "#000000"

# --- 2. Geometry Helpers ---

def get_2d_coords(vector, axis_mode):
    """Projects a 3D world vector onto the selected 2D plane."""
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

# --- 3. Main Export Logic ---

def write_svg(context, filepath, props):
    """Main function to iterate selected objects and write the SVG file structure."""
    scale_factor = props.global_scale
    stroke_width_val = props.stroke_width
    axis_mode = props.projection_axis
    
    fill_mode = props.fill_settings
    fill_source = props.fill_color_source
    fill_user_col = props.user_fill_color
    
    stroke_source = props.stroke_color_source
    stroke_user_col = props.user_stroke_color

    selected_curves = [obj for obj in context.selected_objects if obj.type == 'CURVE']
    
    if not selected_curves:
        return {'NO_SELECTION'}

    svg_content = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" version="1.1">'
    ]
    
    svg_content.append(f'<g transform="scale({scale_factor}, {-scale_factor})">') 
    
    nurbs_encountered = False 

    for obj in selected_curves:
        matrix = obj.matrix_world
        safe_obj_name = obj.name.replace(" ", "_")
        
        svg_content.append(f'  <g id="{safe_obj_name}">')

        obj_fill_hex = resolve_color(obj, fill_source, fill_user_col)
        obj_stroke_hex = resolve_color(obj, stroke_source, stroke_user_col)

        for i, spline in enumerate(obj.data.splines):
            path_d = ""
            
            if spline.type == 'BEZIER':
                path_d = get_bezier_path_d(spline, matrix, axis_mode)
            elif spline.type == 'POLY':
                path_d = get_poly_path_d(spline, matrix, axis_mode)
            elif spline.type == 'NURBS':
                nurbs_encountered = True
                continue 
            
            if path_d:
                final_fill = "none"
                should_fill = False
                is_cyclic = spline.use_cyclic_u

                if fill_mode == 'ALL':
                    should_fill = True
                elif fill_mode == 'CLOSED' and is_cyclic:
                    should_fill = True
                elif fill_mode == 'OBJECT':
                    is_2d = (obj.data.dimensions == '2D')
                    has_fill = (obj.data.fill_mode != 'NONE')
                    if is_2d and has_fill and is_cyclic:
                        should_fill = True
                
                if should_fill:
                    final_fill = obj_fill_hex
                
                final_stroke = obj_stroke_hex
                spline_id = f"{safe_obj_name}_{i}"
                
                svg_content.append(
                    f'    <path id="{spline_id}" d="{path_d}" stroke="{final_stroke}" stroke-width="{stroke_width_val}" fill="{final_fill}" />'
                )

        svg_content.append('  </g>')

    svg_content.append('</g>')
    svg_content.append('</svg>')

    try:
        with open(filepath, 'w') as f:
            f.write("\n".join(svg_content))
            
        if nurbs_encountered:
            return {'FINISHED_WITH_WARNING'}
        return {'FINISHED'}
        
    except Exception as e:
        print(f"Export Error: {e}")
        return {'FILE_ERROR'}

# --- 4. Operator Class ---

class CURVE_OT_export_svg(bpy.types.Operator, ExportHelper):
    """Export selected Curve Objects (Bezier/Poly) as SVG"""
    bl_idname = "export_curve.svg"
    bl_label = "Export Curve as SVG"
    bl_description = "Export selected curve objects as Scalable Vector Graphics"
    filename_ext = ".svg"
    
    filter_glob: StringProperty(default="*.svg", options={'HIDDEN'}, maxlen=255)

    # --- Properties: Projection ---
    global_scale: FloatProperty(
        name="Scale", 
        default=100.0, 
        min=0.01,
        description="Scale multiplier for the exported SVG"
    )
    
    projection_axis: EnumProperty(
        name="Projection View",
        items=[
            ('TOP', "Top View (XY Plane)", "Project geometry onto the XY plane"),
            ('FRONT', "Front View (XZ Plane)", "Project geometry onto the XZ plane"),
            ('SIDE', "Side View (YZ Plane)", "Project geometry onto the YZ plane")
        ],
        default='TOP'
    )

    # --- Properties: Stroke ---
    stroke_width: FloatProperty(
        name="Width", 
        default=0.05, 
        min=0.001,
        description="Stroke width of the SVG paths"
    )
    
    stroke_color_source: EnumProperty(
        name="Color",
        items=[
            ('CUSTOM', "Custom", "Use the color picker below"),
            ('OBJECT', "Object", "Use the Viewport Display Color object property"),
            ('MATERIAL', "Material", "Use the Material's base color"),
            ('RANDOM', "Random", "Generate a random color"),
        ],
        default='CUSTOM'
    )
    user_stroke_color: FloatVectorProperty(
        name="Color", 
        subtype='COLOR', 
        default=(0.0, 0.0, 0.0), 
        min=0.0, max=1.0
    )

    # --- Properties: Fill ---
    fill_settings: EnumProperty(
        name="Mode",
        items=[
            ('NONE', "None", "Do not fill any curves"),
            ('CLOSED', "Closed Curves", "Fill any curve that forms a closed loop"),
            ('OBJECT', "Filled 2D Curves", "Fill only 2D curves with Fill Mode enabled"),
            ('ALL', "All", "Fill every exported curve"),
        ],
        default='NONE'
    )
    fill_color_source: EnumProperty(
        name="Color",
        items=[
            ('CUSTOM', "Custom", "Use the color picker below"),
            ('OBJECT', "Object", "Use the object's Viewport Display Color"),
            ('MATERIAL', "Material", "Use the Material's base color"),
            ('RANDOM', "Random", "Generate a random color"),
        ],
        default='CUSTOM'
    )
    user_fill_color: FloatVectorProperty(
        name="Color", 
        subtype='COLOR', 
        default=(0.0, 0.0, 0.0), 
        min=0.0, max=1.0
    )

    def draw(self, context):
        layout = self.layout
        
        box = layout.box()
        box.label(text="Projection")
        box.prop(self, "projection_axis", text="") 
        box.prop(self, "global_scale")

        box = layout.box()
        box.label(text="Stroke")
        box.prop(self, "stroke_width")
        box.prop(self, "stroke_color_source")
        if self.stroke_color_source == 'CUSTOM':
            box.prop(self, "user_stroke_color", text="") 

        box = layout.box()
        box.label(text="Fill")
        box.prop(self, "fill_settings")
        
        if self.fill_settings != 'NONE':
            box.prop(self, "fill_color_source")
            if self.fill_color_source == 'CUSTOM':
                box.prop(self, "user_fill_color", text="")

    def execute(self, context):
        result = write_svg(context, self.filepath, self)
        
        if result == {'FINISHED'}:
            self.report({'INFO'}, f"SVG Export Successful: {self.filepath}")
            return {'FINISHED'}
        elif result == {'FINISHED_WITH_WARNING'}:
            self.report({'WARNING'}, "SVG Exported, NURBS curves are skipped (Convert to Bezier first)")
            return {'FINISHED'}
        elif result == {'NO_SELECTION'}:
            self.report({'ERROR'}, "No Curve objects selected!")
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
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)