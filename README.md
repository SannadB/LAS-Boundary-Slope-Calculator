# LAS Boundary Slope Calculator

Calculates terrain slope within a user-defined boundary on a point cloud, while filtering out non-terrain objects (trees, cranes, vehicles, temporary structures) so they don't distort the result.

## How it works

```
User selects boundary
        ↓
Extract points within boundary
        ↓
Generate local ground/surface representation  (grid min-Z + morphological smoothing)
        ↓
Identify and remove external objects           (height-above-surface threshold)
        ↓
Generate clean surface
        ↓
Calculate local slope                          (elevation gradient over the smoothed grid)
        ↓
Mark/display slope results
```

Unlike a single global best-fit plane, the ground surface is reconstructed **locally** on an XY grid, so it correctly handles sites with real slope, hills, or multiple elevation levels — it won't mistake a sloped site for "all outliers" the way a single-plane fit would.

### Object removal

1. The site is divided into an XY grid (default 1m × 1m). The minimum Z in each cell is recorded as a first approximation of ground height.
2. That min-Z grid is smoothed with a **morphological opening** (erosion then dilation), sized to remove the influence of isolated objects (crane booms, tree canopies) while preserving real terrain features.
3. Every point is compared against the smoothed surface at its XY location. Points within `--height-threshold` of the surface are classified as **ground**; points significantly above or below are classified as **objects/outliers**.

### Slope calculation

Slope is computed as the elevation gradient over the smoothed ground grid:

```
∂Z/∂x, ∂Z/∂y            (gradient components)
S = sqrt((∂Z/∂x)² + (∂Z/∂y)²)
Slope(deg) = atan(S)
```

This gives one slope value per grid cell (Option A: fixed-size slope grid).

## Installation

```bash
pip install laspy numpy scipy
```

## Usage

```bash
python main.py input.las output_prefix \
    [--polygon-file polygon.json] \
    [--grid-size 1.0] \
    [--height-threshold 0.3] \
    [--opening-size 7] \
    [--max-grid-cells-in-json 200000] \
    [--slope-threshold 45] \
    [--chunk-size 2000000]
```

If `input.las` is already the boundary-selected region (e.g. exported from an earlier boundary pick in your viewer/tool), just pass it directly — no `--polygon-file` needed. If you need to filter a larger file down to a polygon first, pass `--polygon-file` pointing at a JSON file shaped like:

```json
{
  "polygon": [
    {"x": 100.0, "y": 200.0},
    {"x": 150.0, "y": 200.0},
    {"x": 150.0, "y": 250.0},
    {"x": 100.0, "y": 250.0}
  ]
}
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `input_las` | — | Path to the input `.las` file |
| `output_prefix` | — | Prefix used for all output files |
| `--polygon-file` | none | Optional JSON file defining the boundary polygon; omit if the input file is already clipped to the region of interest |
| `--grid-size` | `1.0` | Grid cell size (file units, typically meters) used for both ground reconstruction and slope calculation |
| `--height-threshold` | `0.3` | Max height above/below the local ground surface a point can be and still count as ground |
| `--opening-size` | `7` | Morphological opening window, in grid cells. Should be bigger than the widest expected object footprint (crane base, tree canopy), smaller than real terrain features |
| `--max-grid-cells-in-json` | `200000` | Cap on how many cells get written into the JSON's per-cell grid / threshold list before it falls back to summary-only |
| `--slope-threshold` | `45` | Degrees. Every grid cell with slope strictly greater than this is listed in the JSON, sorted steepest-first |
| `--chunk-size` | `2000000` | Points read per streaming chunk. Lower this if you're memory-constrained |

## Outputs

| File | Contents |
|---|---|
| `<prefix>_analyzed.las` | All points from the boundary region. `classification` = `2` (ground) or `7` (object/outlier). `user_data` = local slope in degrees at that point's grid cell, rounded and clipped to 0–255. CloudCompare imports `User Data` as a per-point scalar field automatically, so this can be colorized as a slope heatmap directly |
| `<prefix>_clean.las` | Ground/surface points only (outliers removed) |
| `<prefix>_outliers.las` | Only the points classified as objects/outliers, for visual QC of what got removed |
| `<prefix>_slope_summary.json` | `max_slope`, `max_slope_location`, `average_slope`, the full per-cell grid (unless it exceeds `--max-grid-cells-in-json`), and every cell above `--slope-threshold` |

### Example `_slope_summary.json`

```json
{
  "max_slope": 23.4,
  "max_slope_location": [103.0, 212.0],
  "average_slope": 6.1,
  "grid_size": 1.0,
  "cells": [ { "x": 100.5, "y": 200.5, "grid_size": 1.0, "slope": 4.2 }, "..." ],
  "slope_threshold": 45,
  "count_above_threshold": 3,
  "cells_above_threshold": [
    { "x": 103.0, "y": 212.0, "grid_size": 1.0, "slope": 23.4 },
    "..."
  ]
}
```

## Streaming design

Both the ground-surface pass and the classify/write pass stream the LAS file in chunks rather than loading it into memory — memory use is bounded by `--chunk-size` and the grid dimensions, not by total point count. This is intended to work on very large site-wide point clouds (hundreds of millions of points) without running out of memory.

## Tuning notes / known limitations

- **`--height-threshold`**: the main knob for ground/object separation. Too tight and legitimately rough terrain (gravel, rubble) gets misclassified as outlier; too loose and low obstructions slip through as ground. Tune against known-good and known-bad regions in your own data before trusting it blindly.
- **`--opening-size`**: must be bigger than the object footprints you want removed but smaller than real terrain features, or it will either miss objects or flatten real terrain changes.
- Cells with zero points at all are filled in via nearest-neighbor before smoothing (so the morphological operation doesn't break on holes), but those filled cells are **excluded** from the slope statistics and JSON output — only cells that actually contained points count.
- `max_slope_location` is the grid cell **center**, not an actual measured point coordinate.
- Slope is a property of the reconstructed *grid*, not of individual points — finer `--grid-size` gives more local detail but is noisier; coarser smooths real local steepness away.
