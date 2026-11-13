# STORM DROP characters

Four original survivor / soldier characters, generated entirely in code by
`tools/make_characters.py` (Blender 4.2 `bpy` from PyPI + numpy + scikit-image).
None of them is based on any existing franchise, celebrity or real person.

| File | Name | Look |
|---|---|---|
| `char_a.glb` | Warden | broad male soldier: high-cut helmet with team band, plate carrier, mottled field camo, assault pack, short beard, team armbands |
| `char_b.glb` | Kestrel | lean female scout: team cap + ponytail, charcoal softshell with team shoulder yoke, light chest rig, sling bag, low boots |
| `char_c.glb` | Bulwark | heavy-set male survivalist: team knit beanie, full beard, canvas work jacket with hood, big trekking pack with team bedroll |
| `char_d.glb` | Vesper | female operator: curly crop, headset, team neck gaiter, quilted bomber jacket, slick plate carrier, grey camo pants, knee pads |

Renders are in `renders/`; triangle counts, sizes, heights and clip lengths are
in `stats.json`.

## Contract for `index.html`

* Units are metres, glTF +Y is up, the character faces **+Z**, and the feet stand on y = 0. Each character is 1.83–1.86 m tall, including headgear.
* **One shared skeleton.** Every file has the same 19 joints with identical rest poses, so a clip from one file can drive any other character:
  `Hips Spine Chest Neck Head Shoulder.L/R UpperArm.L/R LowerArm.L/R Hand.L/R UpperLeg.L/R LowerLeg.L/R Foot.L/R`.
  The rest pose is an A-pose.
* **Animations** (30 fps, all in place, so the game moves the root):
  * Loops: `Idle Walk Run Sprint StrafeLeft StrafeRight WalkBack CrouchIdle CrouchWalk Fall Slide Skydive Glide AimIdle`.
  * One-shots: `Jump` (takeoff, then holds the air tuck), `Land`, `Vault`, `Death` (ends lying on the back), `Swing` (right-handed pickaxe swing).
  * All ground clips (Idle, locomotion, crouch, Jump/Fall/Land, Slide) carry a rifle: the right hand stays on the pistol grip and the left hand on the handguard. `AimIdle` holds the rifle at shoulder height, aimed along +Z.
  * `Glide` holds a glider bar about 2.0 m above the feet, with the hands 0.46 m apart.
* **`GunMount`** is an empty parented to `Hand.R`. To attach a weapon from `models/guns/` (grip at its origin, barrel along +Z), make the gun a child of `GunMount` with an identity transform.
* **`Team`** is a near-white fabric material on armbands, the cap, the helmet band, the beanie and bedroll, the shoulder yoke and the gaiter. Set `material.color` to tint it per player.
* Other materials: `Skin`, `Hair`, `Fabric`, `Leather`, `Gear`. Each has an embedded JPEG base colour, roughness/metal and normal texture of at most 1024 px. No Draco, no tangents (three.js derives them).

## Rebuilding

```
pip install bpy==4.2.0 scikit-image
python tools/make_characters.py              # build, verify, render all
python tools/make_characters.py b --no-render
python tools/make_characters.py --render-only
```

Every export is re-imported and checked: the animation list, the bone names, `GunMount` sitting under `Hand.R` and lining up with forward in `AimIdle`, the `Team` material, the height, feet on y = 0, and the triangle budget.
