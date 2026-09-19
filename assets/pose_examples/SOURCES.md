# Pose-guide asset sources

The repository contains only the eight pose images used by the application. The Index guides are AI-generated. The Middle, Ring, and Little guides were restyled on 2026-09-19 from local renders of the project owner's saved [Light Dropper hand setups](https://www.lightdropper.com/references/posable-hands-3d-reference.html), using the approved Index guides as the visual reference.

| File | Purpose | Source status |
| --- | --- | --- |
| `index_extension_avatar_hand.png` | Active Index-finger maximum-extension avatar guide, displayed as its own image | AI-generated project asset created on 2026-08-29 from the project-owner-approved Index pose; transparent background; no patient photograph is shown |
| `index_flexion_avatar_hand.png` | Active Index-finger maximum-flexion avatar guide, displayed as its own image | AI-generated project asset created on 2026-08-29 from the project-owner-approved Index pose; transparent background; no patient photograph is shown |
| `middle_extension_avatar_hand.png` | Active Middle-finger maximum-extension avatar guide | AI-restyled from the project owner's saved pose render, with the Index guide as the visual reference; transparent background; no patient photograph is shown |
| `middle_flexion_avatar_hand.png` | Active Middle-finger maximum-flexion avatar guide | AI-restyled from the project owner's saved pose render, with the Index guide as the visual reference; transparent background; no patient photograph is shown |
| `ring_extension_avatar_hand.png` | Active Ring-finger maximum-extension avatar guide | AI-restyled from the project owner's saved pose render, with the Index guide as the visual reference; transparent background; no patient photograph is shown |
| `ring_flexion_avatar_hand.png` | Active Ring-finger maximum-flexion avatar guide | AI-restyled from the project owner's saved pose render, with the Index guide as the visual reference; transparent background; no patient photograph is shown |
| `little_extension_avatar_hand.png` | Active Little-finger maximum-extension avatar guide | AI-restyled from the project owner's saved pose render, with the Index guide as the visual reference; transparent background; no patient photograph is shown |
| `little_flexion_avatar_hand.png` | Active Little-finger maximum-flexion avatar guide | AI-restyled from the project owner's saved pose render, with the Index guide as the visual reference; transparent background; no patient photograph is shown |

The Middle, Ring, and Little source renders used saved bone positions and rotations, the viewer's blue model color, and a local copy of the viewer's publicly served hand model. The active PNGs use those source renders for pose guidance and the approved Index PNGs for their smooth anatomical appearance, blue material, lighting, and landscape framing. The Middle guides use the upper, back-of-hand view; the Ring and Little guides use the under, palm-facing view. AI restyling can introduce small anatomical differences from the original 3D bone configuration. The source JSON files and hand model are not bundled in this deployment repository.

The application shows a short Arabic positioning instruction below the selected finger-specific avatar guide. These images are not clinician-approved photographs, medically precise anatomy, diagnostic evidence, or proof that a patient's pose is correct. They illustrate finger positions; users should follow the separate side-view photo checklist for camera placement.

All four active finger guides use separate transparent extension and flexion images. These assets are displayed without mirroring, so users reverse the physical hand orientation for the opposite hand.

If these guides are replaced with photographs, use only project-owner-supplied, clinician-approved, public-domain, or appropriately licensed images. Record source, license, consent/release status, acquisition date, and attribution requirements here.
