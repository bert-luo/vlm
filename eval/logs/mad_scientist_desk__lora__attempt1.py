# Model: lora
# Prompt: mad scientist desk
# Attempt: 1
# Status: SYNTAX ERROR: invalid syntax (<unknown>, line 132)
# Timestamp: 2026-03-31T01:15:53.113894
# Lines: 195 | Chars: 8436
# ============================================================
import bpy, math, os, sys

def parse_output():
    argv = sys.argv
    if "--" in argv:
        args = argv[argv.index("--") + 1:]
        if "--output" in args:
            return args[args.index("--output") + 1]
    return "/tmp/output.png"

def make_mat(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.5
    bsdf.inputs["Metallic"].default_value = 0.0
    return mat

def add_cube(loc, name, scale):
    bpy.ops.mesh.primitive_cube_add(location=loc)
    o = bpy.context.active_object
    o.name = name
    o.scale = scale
    o.material.append(make_mat(name, (0.12, 0.05, 0.02)))

def add_cyl(loc, name, scale, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    o.scale = scale

def main():
    output_path = parse_output()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.object.camera_add(location=(2.5, -1.5, 1.85))

    # ── Materials ───────────────────────────────────────────────
    M_ = {
        "desk":      (0.35, 0.14, 0.03),  # dark wood
        "metal":     (0.52, 0.52, 0.54),  # glossy steel
        "green_m":   (0.05, 0.30, 0.20),  # green plastic
        "beaker_g":  (0.06, 0.40, 0.18),  # green fluid
        "frosted":   (0.75, 0.75, 0.75),  # white frosted plastic
        "chair":     (0.20, 0.20, 0.20),  # grey upholstered
        "chair_leg": (0.5, 0.45, 0.30),   # wood chair legs
    }
    m = {**M_, "drum": M_["metal"], "gear": M_["metal"],
         "pipe": M_["metal"], "bottle": M_["green_m"], "dial": M_["dial": 0.05, 0.04, 0.05]}
    m["chair_widget"] = make_mat("chair_widget", (0.08, 0.20, 0.06))
    m["leg_flat"] = make_mat("leg_flat", (0.6, 0.58, 0.55))
    m["chair_seat"] = make_mat("chair_seat", (0.50, 0.50, 0.48))
    for mn, mv in [["Chair_back", M_["chair"]],
                    ["Chair_arm_r", M_["chair"]],
                    ["Chair_arm_l", M_["chair"]]]:
        pass

    # ── S bg ─────────────────────────────────────────────────────
    bpy.world.node_tree.nodes["Background"].inputs["Color"].default_value = (*M_["desk"], 1)
    bpy.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.1

    # ── Floor ─────────────────────────────────────────────────────
    add_cyl((0, 0, -0.01), "Floor", (3.5, 3.5, 0.01))

    # ── Desk units ────────────────────────────────────────────────
    defs = [
        ("desk_xspan", (-1.5, 1.5),  (3.0, 0.6, 0.06)),
        ("desk_left",   (-1.8,-0.6), (4.22, 0.52, 0.06)),
        ("desk_right",  ( 1.8,-0.6), (4.22, 0.52, 0.06)),
    ]
    for two, ln, sc in defs:
        bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
    for loc, nm, sc in defs:
        add_cube(loc, nm, sc)

    # ── Four metal drums ─────────────────────────────────────────
    for i, dx in enumerate([-2.0, 2.0, -2.0, 2.0]):
        add_cyl((dx, -3.5, 6.7), f"drum_{i}", (0.85, 0.85, 1.25))

    # Seven drums in a ring
    for i, dx in enumerate([-5.0, 5.0, -5.0, 5.0, -5.0, 5.0, -5.0]):
        add_cyl((dx * (1 - i*0.7), -3.5, 6.3), f"small_drum_{i}", (0.55, 0.55, 0.92))

    # ── Organic pipe clusters ─────────────────────────────────────
    for i, xx in enumerate([-4, 0, 4]):
        for j, (y, sy) in enumerate([(-3.5, 0.7), (-9.0, 0.9), (9.0, 0.9)]):
            style = 2 if j==2 else 1
            add_cyl((xx+0.5, y+sy, 6.0 + i*1.5), f"pipe_{i}_{j}",
                    (0.18 if style else 0.26, 0.18, 1.4))

    golden_dome = add_cyl((0.0, -6.8, 4.55), "dome_body", (1.32, 1.32, 2.08))
    golden_disk = add_cyl(( 1.52,-6.8, 1.25), "dome_rim", (1.68, 1.68, 0.05))

    # ── Chemical green veiligheidskegels ─────────────────────────
    for i, xx in enumerate([-6.0, 6.0, -6.0, 6.0]):
        add_cyl((xx, 2.5, 5.5), f"kegel_{i}", (0.22, 0.22, 0.38))
        add_cyl((xx, 2.3, 5.82), f"top_{i}", (0.22, 0.22, 0.10))
        glow = ("green_glow", 0.03, 0.8, 1.0)
        bpy.ops.object.light_add(type="AREA", location=(xx-0.35, 2.3, 5.78))
        p = bpy.context.active_object
        p.data.energy = 2.0
        p.data.size = 0.55
        p.data.shadow_soft_size = 0.2

    # Chemiebench grotsen
    for i in range(5):
        add_cyl((i*1.8-4.5, 2.0, 6.0), f"lab_bench_{i}", (1.6, 1.2, 0.06))

    # ── Gautes ───────────────────────────────────────────────────
    for slot, (gx, gy) in enumerate([(-5.0,3.0),(5.0,3.0),(-5.0,-3.0),(5.0,-3.0)]):
        for j, (py, pz) in enumerate([(2.0, 7.2), (3.0, 7.6), (4.0, 7.0), (2.8,7.8)]):
            add_cyl((gx+0.3, gy+1.5, pz), f"lab_gut_{slot}_{j}", (0.28, 0.28, 1.5))
            for (tx, ty) in [(-1.5,0),(1.5,0)]:
                add_cyl((gx+tx, gy+1.5, pz+2.5), f"lab_gut_{slot}_bulb_{j}", (0.12, 0.12, 0.85))

    # ── Walkway treddes ───────────────────────────────────────────
    for board in [(-3.0, 0), (3.0, 0), (-3.0, -6.0), (3.0, -6.0)]:
        add_cyl(board, f"spdler_{board}", (0.9, 0.9, 0.12))

    # ── Backmade L－ alegene ──────────────────────────────────────
    add_cube((0, -6.25, 1.05), "ch_b_as", (0.26, 0.26, 0.26))
    add_cube((0, -6.25, 1.28), "ch_b_ac", (0.26, 0.26, 0.08))
    for i in range(2):
        add_cube((0.3*i, -7.5, 0.28), f"ch_as_{i}", (0.12, 0.42, 0.06))
    for lx in [-0.30, 0.35]:
        add_cube((lx, -5.8, 0.48), f"ch_spout_{lx}", (0.05, 0.30, 0.06))

    # ── Esperas ───────────────────────────────────────────────────
    bx = [-1.5,3.5, -2.0, 4.0, -3.5, 3.0, -4.5, 1.5]
    By = [-7.8, -7.8, -7.8, -7.8, -5.8, -5.8, -4.8, -3.8]
    For i, ((x,y)) in enumerate(zip(bx, By)):
        if y > -6:
            add_cube((x, y, 2.0), f"veranda_stem_{i}", (0.6, 0.6, 1.0))
            add_cyl((x, y-0.12, 3.1), f"veranda_gut_{i}", (0.14, 0.14, 1.1))
        else:
            add_cube((x, y, 2.0), f"metal_column_{i}", (0.6, 0.6, 1.0))
            for lx, lz in [(-0.5, 1.8),(0.5,1.8),(0.5,3.8),(-0.5,3.8)]:
                add_cyl((lx, y+0.25, lz), f"met法律依据_{i}", (0.20, 0.20, 0.75))
            add_cube((y, 2.22, 1.05), f"top_plate_{i}", (1.1, 1.1, 0.5))

    # ── Sticker knoppen ──────────────────────────────────────────
    for dim, ((ax, ay), sx) in zip([(-5.0,3.0),(5.0,3.0),(-5.0,-3.0),(5.0,-3.0)],
                                    [(-2.5,0),(0,0),(2.5,0),(-1.5,-3)]):
        add_cyl((ax, ay, 1.8), "knop", (0.09,0.09,0.09))

    # ── Cabine ───────────────────────────────────────────────────
    add_cube((-1.8,-7.65,1.2), "ch_seat", (0.72,0.72,0.72), (0.45,0.45,0.45))
    add_cube((-1.8,-7.65,1.9), "ch_back", (0.48,0.48,0.65), (0.52,0.52,0.62))
    add_cube((-0.65,-7.65,1.6), "ch_arm_r", (0.48,0.12,0.65), (0.52,0.06,0.62))
    add_cube(( 0.65,-7.65,1.6), "ch_arm_l", (0.48,0.12,0.65), (0.52,0.06,0.62))

    # ── Retro-watervlamp ─────────────────────────────────────────
    for i in range(6):
        add_cube((i*2.2, -3.4, 0.08), f"retro_", i)
        bp = bpy.ops.object.light_add(type="AREA")
        loc = (i*2.2+2.0, -3.4, 1.2)
        bpy.ops.object.light_add(type="AREA", location=loc)
        l = bpy.context.active_object
        l.data.energy = 5.0
        l.data.size = 1.2
        l.rotation_euler = (0, 0, 0)

    # ── Lighting ─────────────────────────────────────────────────
    bpy.ops.object.light_add(type="AREA", location=(6, 0, 6.5))
    l = bpy.context.active_object
    l.data.energy = 800
    l.data.color = (1.0, 0.92, 0.65)
    l.data.size = 10
    l.rotation_euler = (math.radians(30), 0, math.radians(45))

    bpy.ops.object.light_add(type="AREA", location=(-5, -9, 5))
    l2 = bpy.context.active_object
    l2.data.energy = 400
    l2.data.color = (0.85, 0.82, 0.65)
    l2.data.size = 8
    l2.rotation_euler = (math.radians(50), 0, math.radians(-20))

    # ── Camera ───────────────────────────────────────────────────
    cam = bpy.context.scene.camera
    cam.data.lens = 50
    cam.rotation_euler = (math.radians(56), 0, math.radians(49))
    cam.location = (4.2, -1.3, 1.88)

    # ── Render ───────────────────────────────────────────────────
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 800
    scene.render.resolution_y = 600
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = os.path.abspath(output_path)

    bpy.ops.render.render(write_still=True)

main()