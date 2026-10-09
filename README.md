# Tire Sidewall UV

**Blender addon** that projects a clean planar disc UV onto tire sidewalls — sized to the tire diameter, aligned to the wheel axis — with a live texture preview, ready for export to **Unreal Engine 5**.

by **Alfredo Mendez**

---

## Download

👉 **[Download the latest version](../../releases/latest)** — grab `tire_sidewall_uv_v1.1.zip` from the Assets list.

## Install

1. In Blender go to **Edit → Preferences → Add-ons**.
2. Click the **▾** menu (top right) → **Install from Disk…** and pick the downloaded `.zip`.
3. Enable **Tire Sidewall UV**.
4. Open the **N-panel** in the 3D Viewport → **Tire UV** tab.

Compatible with **Blender 4.3 and above** (tested up to 5.1).

## How to use

1. **Select the tire mesh** (Object Mode).
2. **Texture** — pick an image already in the file or load one from disk (sidewall lettering / logo disc, ideally with alpha).
3. **Detect & Map** — finds the wheel axis and center automatically (PCA + bead-seat circle fit), detects the sidewall faces and projects the UV.
   - **Side:** Outer, Inner or Both.
   - **Normal Threshold / Min Radius:** fine-tune which faces count as sidewall.
   - **Flip Outer/Inner:** use it if the outer side was detected on the wrong face.
4. **Adjust** — Rotation, Scale and U/V Offset update live.
5. **Preview On** — shows the texture on the sidewall with a temporary material. **Turn Preview Off before exporting** (UVs are kept, original materials restored).

### UV Mode

| Mode | What it does |
|---|---|
| **Same UV** (default) | Writes the disc into the tire's main UV map (UE channel 0). *Park Rest of Tire* moves all non-sidewall UVs to a transparent corner of the texture so the lettering only shows on the sidewall. |
| **Separate UV** | Writes the disc into its own UV map (e.g. `UV_Sidewall`) → an extra UV channel in Unreal. |

The panel shows the tire diameter, texture density (px/cm) and the target UE UV channel.

## Changelog

- **v1.1** — UV Mode: Same UV (default) or Separate UV; *Park Rest of Tire* option for single-UV workflows.
- **v1.0** — Initial release.

## License

GPL-3.0-or-later — see [LICENSE](LICENSE).
