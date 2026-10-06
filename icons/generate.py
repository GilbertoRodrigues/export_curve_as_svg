# SPDX-License-Identifier: CC0-1.0

"""Original stroke icons. Run manually with Blender to regenerate the .dat files."""

import math
from pathlib import Path
from mathutils import Vector

ICONS = Path(__file__).resolve().parent

# Blender doubles geometry icons to toolbar size around their left edge and
# vertical center. Inset the artwork to keep it inside a regular 16px button.

def generate_stroke_icons():
    """Write the bundled icon geometry in Blender's VCO format."""
    from mathutils.geometry import tessellate_polygon

    def add_icon(style, value, layers):
        coordinates = bytearray()
        colors = bytearray()
        for polygon, shade in layers:
            vertices = [Vector((x, y, 0)) for x, y in polygon]
            for triangle in tessellate_polygon([vertices]):
                for index in triangle:
                    vertex = vertices[index]
                    coordinates.extend((round(vertex.x / 2), round(64 + vertex.y / 2)))
                    colors.extend((shade, shade, shade, 255))
        target = ICONS / f'{style}_{value.lower()}.dat'
        target.write_bytes(b'VCO\x00' + bytes((255, 255, 0, 0)) + coordinates + colors)

    # Endpoint guides distinguish a butt cap from an extending square cap.
    guides = [
        ([(154, 40), (166, 40), (166, 76), (154, 76)], 140),
        ([(154, 180), (166, 180), (166, 216), (154, 216)], 140),
        ([(32, 122), (160, 122), (160, 134), (32, 134)], 80),
    ]
    for value in ('BUTT', 'ROUND', 'SQUARE'):
        if value == 'ROUND':
            end = [
                (160 + 40 * math.cos(angle), 128 + 40 * math.sin(angle))
                for angle in (-math.pi / 2 + math.pi * i / 16 for i in range(17))
            ]
        else:
            x = 160 if value == 'BUTT' else 200
            end = [(x, 88), (x, 168)]
        polygon = [(32, 88), *end, (32, 168)]
        add_icon('cap', value, [(polygon, 210), *guides])

    # A thick chevron makes the three different outer corners easy to compare.
    for value in ('MITER', 'ROUND', 'BEVEL'):
        if value == 'MITER':
            corner = [(128, 229.333)]
        elif value == 'ROUND':
            start = math.atan2(19.2, -25.6)
            end = math.atan2(19.2, 25.6)
            corner = [
                (128 + 32 * math.cos(angle), 176 + 32 * math.sin(angle))
                for angle in (start + (end - start) * i / 16 for i in range(1, 16))
            ]
        else:
            corner = []
        polygon = [
            (6.4, 67.2), (102.4, 195.2), *corner, (153.6, 195.2),
            (249.6, 67.2), (198.4, 28.8), (128, 122.667), (57.6, 28.8),
        ]
        add_icon('join', value, [(polygon, 210)])


if __name__ == "__main__":
    generate_stroke_icons()
