# export_curve_as_svg
An extension for Blender.

This adds an exporter to Blender, for saving the selected Bezier and Poly curve objects as SVG (.svg). It supports exporting both 2D and 3D Bezier and Poly curve objects, with option to pick from multiple projection views (Top, Front, Side),
    a scale slider export property, configurable vector fill (none, all, closed curves, filled 2D curves), fill colors (Custom, Object, Material, Random), Stroke width and stroke color (Custom, Object, Material, Random).
    
Limitations: The extension does not yet support NURBs, "Curves" Object, and Text Object, So convert them to Beziér curve before exporting. Other limitation is that this exporter ignores the Modifiers.

Usage after install:
- Select curve objects
- File > Export > Curve as SVG (.svg)
- Configure export settings
- Click Export SVG

Install intructions: https://docs.blender.org/manual/en/latest/editors/preferences/extensions.html
