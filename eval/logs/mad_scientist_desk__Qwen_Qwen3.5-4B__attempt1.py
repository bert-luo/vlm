# Model: Qwen/Qwen3.5-4B
# Prompt: mad scientist desk
# Attempt: 1
# Status: SYNTAX OK
# Timestamp: 2026-03-31T01:14:45.432328
# Lines: 91 | Chars: 3499
# ============================================================
python
import bpy
import math
import os
import sys

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

def create_material(name, color=(1, 1, 1), roughness=0.5, metallic=0.2):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    tree = mat.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    emb = tree.nodes.new("ShaderNodeEmission")
    tree.links.new(emb.outputs["Color"], bsdf.inputs["Emission Color"])
    emb.inputs["Color"].default_value = color
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat

def create_cube(size=1, location=(0, 0, 0), material=None, scale=(1, 1, 1)):
    bpy.ops.mesh.primitive_cube_add(size=size, location=location, scale=scale)
    return bpy.data.objects["Cube"]

def main():
    bpy.ops.object.select_all(action="DESELECT")
    
    # Parse output path
    if len(sys.argv) > 2 and sys.argv[2].startswith("--"):
        for arg in sys.argv[2:]:
            if arg.startswith("--output"):
                output_path = os.path.sep.join(arg.split("--output")[-1].split())
    else:
        output_path = "output.png"
    
    output_path = os.path.abspath(output_path)
    
    # Clear scene
    clear_scene()
    
    # Create Desk - wooden surface on floor
    desk_surface = create_cube(size=2.0, location=(-0.5, -0.5, 0.0),
                              material=create_material("wood_desk", (0.7, 0.5, 0.3), 0.8, 0.1))
    desk_legs = create_cube(size=0.1, location=(-1.0, 0.2, 0.5), material=None)
    desk_legs.data.materials[:] = [create_material("metal_leg", (0.8, 0.8, 0.9), 0.95, 0.8)]
    desk_legs1 = create_cube(size=0.1, location=(0.0, 0.2, 0.5), material=None)
    desk_legs1.data.materials[:] = [create_material("metal_leg", (0.8, 0.8, 0.9), 0.95, 0.8)]
    desk_legs2 = create_cube(size=0.1, location=(0.5, 0.2, 0.5), material=None)
    desk_legs2.data.materials[:] = [create_material("metal_leg", (0.8, 0.8, 0.9), 0.95, 0.8)]
    
    # Calculator
    calc_box = create_cube(size=0.5, location=(2.0, -0.2, 0.3), material=create_material("gray_calc", (0.3, 0.3, 0.4), 0.3, 0.2))
    
    # Bottle
    bottle = create_cube(size=0.2, location=(-2.5, -0.2, 0.5), material=create_material("glass_bottle", (1.0, 0.8, 0.8), 0.95, 0.1))
    
    # Add light (AREA)
    bpy.ops.object.light_add(type="AREA", size=2, location=(5, 5, 5))
    light_data = bpy.data.objects["Area Light"]
    bpy.context.scene.cycles.workspace_grid = "UNIT_RACES"
    
    # Add Camera
    bpy.ops.object.camera_add(location=(0, 0, 5))
    bpy.context.scene.camera = bpy.context.active_object
    
    # Configure render settings
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    bpy.context.scene.render.resolution_x = 800
    bpy.context.scene.render.resolution_y = 600
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.film_transparency = 1.0
    
    # World background for EEVEE
    world = bpy.data.worlds["World"]
    if not world:
        world = bpy.data.worlds.new("World")
    world.use_nodes = False
    tree = world.node_tree
    bg = tree.nodes.new("ShaderNodeBackground")
    tree.nodes[bg.node_tree_index]
    tree.inputs["World Layer Color"].default_value = (0.2, 0.15, 0.1, 1.0)
    
    bpy.context.scene.cycles.render_samples = 128

    bpy.ops.render.render(write_still=True)

if __name__ == "__main__":
    main()