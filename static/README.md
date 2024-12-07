we use python to control `blender` to render our desired image. This is the template of python code (`bpy`) that can conduct `blender` to render image.

 - `bg.ply` is a big white background mesh for render.
 - `blender_render_script_figure.template.py` is used to render the articulated object with new color for each sub-parts. The image are provided to GPT-4o to generate text condition.
 - `blender_render_script.template.py` is used to render the articulated object with its original color/material in dataset. The image serves as image condition.