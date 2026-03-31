# Generation Comparison: "mad scientist desk"

Comparing two generation attempts:

- **LoRA** — `logs/mad_scientist_desk__lora__attempt1.py`
- **Base Qwen3.5-4B** — `logs/mad_scientist_desk__Qwen_Qwen3.5-4B__attempt1.py`

---

## LoRA (`lora__attempt1.py`) — More Ambitious, More Broken

**Scene complexity:** Very ambitious. It attempts to build a full mad-scientist environment with a desk, metal drums, pipe clusters, a golden dome, chemical "beakers," a chair, lab benches, columns, walkways, and atmospheric multi-light setups.

### Critical Bugs

- **Line 57** — `M_["dial": 0.05, 0.04, 0.05]` is invalid Python (dict slice syntax), which is the reported syntax error.
- **Line 32** — `o.material.append(...)` should be `o.data.materials.append(...)`.
- **Line 139** — `For` is capitalized (`For i, ...`), another syntax error.
- **Lines 155–158** — `add_cube` called with extra positional args the function doesn't accept.
- **Line 162** — `f"retro_", i)` passes `i` as a separate argument, not inside the f-string.
- **Line 146** — contains Chinese characters in a name (`f"met法律依据_{i}"`), suggesting multilingual confusion.
- **Line 67** — uses `bpy.world` (doesn't exist; should be `bpy.context.scene.world`).
- Camera is added but never assigned to `bpy.context.scene.camera`.
- Comments are a mix of English and Dutch ("veiligheidskegels", "knoppen", "Retro-watervlamp"), indicating the model is leaking multilingual training data.

### Strengths

- Two area lights with warm tones and intentional angles — good for atmosphere.
- Proper camera lens (50mm) and angled positioning.
- Thoughtful material palette (dark wood, glossy steel, green fluid, frosted plastic).
- Scene *conceptually* captures "mad scientist" very well.

---

## Base Qwen3.5-4B (`Qwen_Qwen3.5-4B__attempt1.py`) — Simpler, Still Buggy

**Scene complexity:** Very sparse. A desk surface, 3 legs (not even 4), a "calculator" cube, and a "bottle" cube. Barely qualifies as a "mad scientist desk."

### Critical Bugs

- **Line 8** — stray `python` token (markdown artifact) would cause a `NameError`.
- **Line 33** — `create_cube` always returns `bpy.data.objects["Cube"]`, which breaks after the first call since subsequent cubes get auto-names like `Cube.001`.
- **Lines 53–54** — `material` parameter is accepted but never applied to the object.
- **Line 55** — `desk_legs.data.materials[:] = [...]` doesn't work like that in the Blender API.
- **Line 70** — `bpy.context.scene.cycles.workspace_grid = "UNIT_RACES"` is a completely made-up property.
- **Lines 87–91** — disables nodes (`use_nodes = False`) then immediately tries to manipulate `world.node_tree`, and references non-existent attributes (`bg.node_tree_index`, `tree.inputs["World Layer Color"]`).
- **Line 93** — sets Cycles render samples while using the EEVEE engine.
- Never sets `render.filepath`, so the output would go nowhere.
- Camera at `(0, 0, 5)` with no rotation — straight-down bird's-eye view.

### Strengths

- Passes syntax checking (no parse-level errors).
- Cleaner code structure and proper `if __name__ == "__main__"` guard.
- Readable helper functions.

---

## Verdict

**The LoRA generation is higher quality overall**, despite having more syntax errors.

| Dimension | LoRA | Base Qwen3.5-4B |
|---|---|---|
| Prompt fidelity ("mad scientist desk") | Strong — pipes, drums, chemicals, lab benches, atmospheric lighting | Weak — a desk, calculator, bottle |
| Scene design / creativity | Rich, multi-element composition | Minimal, 5-6 simple cubes |
| Material quality | Curated palette with named semantic colors | Basic colors, some hallucinated properties |
| Lighting | Two warm area lights at different angles | One default light, no tuning |
| Camera work | 50mm lens, angled shot | Bird's-eye, no lens or rotation set |
| Syntactic correctness | Fails (multiple errors) | Passes parse, but many runtime errors |
| Blender API accuracy | Several mistakes, but mostly correct patterns | More hallucinated/non-existent API calls |

The LoRA model clearly learned more about **what makes a good Blender scene** (composition, lighting, materials, camera angles) even though it makes more syntax mistakes. The base model produces structurally cleaner but creatively barren output with its own set of runtime-fatal API hallucinations. If you fixed the bugs in both, the LoRA output would render a far more interesting and faithful "mad scientist desk" scene.
