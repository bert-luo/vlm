# Data Generation Plan — Blender Scene Fine-Tuning

## Goal

Fine-tune a VLM to generate better Blender Python scripts (`bpy`) for interactive-world scenes
targeting **gaming** (AAA, indie, stylized) and **robotics** (manipulation, navigation, inspection)
use cases.

Each training example: `prompt → bpy script → rendered PNG`

---

## Dimensions

### 1. Domain
| Bucket | Sub-types |
|---|---|
| **Gaming** | fantasy, sci-fi, post-apocalyptic, modern/action, horror, stylized/indie |
| **Robotics** | manipulation, navigation, inspection, collaborative, outdoor autonomy |
| **Mixed** | cross-domain test labs, sim environments, smart factories |

### 2. Scene Scale
| Level | Examples |
|---|---|
| Micro | tabletop, workbench, sample tray |
| Room | bedroom, dungeon cell, server room |
| Multi-room | hospital corridor, open-plan office, warehouse aisle |
| Outdoor area | courtyard, loading dock, agricultural row |
| Large / open | arena, cargo hold, test track |

### 3. Functional Role (what an agent does here)
- **Navigate** — clear paths, obstacles, doorways, ramps, sightlines
- **Manipulate** — reachable surfaces, graspable objects, placement targets
- **Inspect** — pipelines, panels, structural elements close-up
- **Complete a task** — cooking, assembly, sorting, dispensing
- **Combat / stealth** — cover geometry, sightlines, chokepoints (gaming)

### 4. Object Density
| Level | Characteristic |
|---|---|
| Sparse | 1–5 objects, open floor, baseline test |
| Medium | typical furnished scene, navigable |
| Dense | cluttered, narrow paths, manipulation challenge |

### 5. Lighting Conditions
- Bright natural (daylight, windows)
- Warm artificial (lanterns, desk lamps, fire)
- Dim / moody (single bulb, candles)
- Emergency (red strip lighting, bunkers)
- Night + neon (cyberpunk, urban)
- Dramatic beam (projector, sunrise through blinds, spot surgical)
- Bioluminescent / colored (underwater, fantasy cave)

### 6. Structural Complexity
- Flat single-level
- Multi-level (stairs, catwalks, platforms, mezzanines)
- Confined (crawlspace, submarine, shipping container)
- Vertical (mine shaft, silo, tower)
- Circular / radial (arena, tank, dome)
- Irregular / ruined (collapsed sections, debris blocking paths)

### 7. Artistic Style
- Realistic / PBR
- Low-poly / flat-shaded game art
- Sci-fi futuristic
- Fantasy medieval
- Post-apocalyptic / gritty
- Stylized / exaggerated proportions
- Period (art deco, brutalist, retro 80s, roman)

---

## Prompt Taxonomy

200 prompts in `prompts.jsonl`. Each line:
```json
{
  "prompt": "scene description text",
  "category": "gaming_fantasy",
  "domain": "gaming",
  "tags": ["dungeon", "fantasy", "small", "moody"]
}
```

### Category breakdown
| Category | Count | Coverage focus |
|---|---|---|
| `gaming_fantasy` | 17 | dungeons, castles, taverns, temples |
| `gaming_scifi` | 11 | spaceships, stations, cyberpunk |
| `gaming_postapoc` | 9 | ruins, bunkers, survivor camps |
| `gaming_modern` | 8 | police, military, casino, museum |
| `gaming_horror` | 5 | asylum, mansion, church |
| `gaming_stylized` | 9 | hobbit-hole, clockwork, retro arcade |
| `gaming_genre` | 9 | stealth, puzzle, platformer, racing |
| `robotics_manipulation` | 15 | kitchen, lab, assembly, sorting |
| `robotics_navigation` | 16 | hospital, warehouse, mall, park |
| `robotics_inspection` | 10 | pipelines, solar, aircraft, tunnels |
| `robotics_collaborative` | 9 | OR, rehab gym, pharmacy, classroom |
| `robotics_outdoor` | 11 | sidewalk, farm, loading dock, rooftop |
| `structural` | 18 | empty rooms, multi-level, confined |
| `lighting` | 10 | one scene per lighting archetype |
| `style_variant` | 10 | same scene type, different style |
| `density` | 5 | sparse → cluttered continuum |
| `technical` | 10 | cleanroom, substation, broadcast |
| `educational` | 7 | lab, studio, workshop |
| `mixed` | 7 | cross-domain, sim environments |
| `reference` | 5 | canonical baseline scenes |

---

## Generation Pipeline

```
prompts.jsonl
    │
    ▼
generate.py  ──►  Claude API (claude-sonnet-4-6)
    │              system prompt: bpy script rules
    │              user: "Scene description: <prompt>"
    │
    ▼
data/scripts/<slug>.py      ← generated bpy script (syntax-validated)
    │
    ▼
blender --background --python <script> -- --output <path>
    │
    ▼
data/output/<slug>.png      ← rendered image
```

### Quality gates in `generate.py`
1. `ast.parse()` on generated code — syntax check before saving
2. Retry with error message if syntax fails (up to 2 attempts)
3. Verify output PNG exists after Blender exits (catches silent runtime errors)

---

## Fine-Tuning Data Format

Each example should be a `(prompt, script, image)` triple:

```jsonl
{
  "prompt": "fantasy tavern common room with wooden tables ...",
  "script_path": "data/scripts/fantasy_tavern_common_room_with_wooden.py",
  "image_path": "data/output/fantasy_tavern_common_room_with_wooden.png"
}
```

For VLM fine-tuning the primary signal is `prompt → script` (text-to-code).
The rendered image enables:
- Visual quality filtering (reject blank/broken renders)
- Image-conditioned modes: `image → script` (scene recreation) or `(image, edit_request) → script`

---

## Batch Generation

Run all 200 prompts:
```bash
python3 -c "
import json, subprocess, sys
for line in open('data/prompts.jsonl'):
    p = json.loads(line)['prompt']
    subprocess.run([sys.executable, 'generate.py', p], check=False)
"
```

Or in parallel with a pool:
```bash
cat data/prompts.jsonl \
  | python3 -c "import sys,json; [print(json.loads(l)['prompt']) for l in sys.stdin]" \
  | xargs -P 4 -I{} python3 generate.py "{}"
```

---

## Coverage Gaps to Fill Later

- **Dynamic / animated scenes** (moving objects, rigged characters) — needs animation timeline prompts
- **Exterior terrain** (heightmaps, large outdoor biomes) — currently only small outdoor areas
- **Procedural variation** — same prompt with randomized object counts / positions
- **Failure modes** (intentionally broken scripts) — for error-recovery fine-tuning
- **Camera perspective variants** — first-person, top-down orthographic, close-up detail shots
