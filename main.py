#!/usr/bin/env python3
"""
Boundary-based slope calculation, following the feasibility workflow:

  User selects boundary
        -> Extract points within boundary
        -> Generate local ground/surface representation (grid min-Z + smoothing)
        -> Identify and remove external objects (height-above-surface threshold)
        -> Generate clean surface
        -> Calculate local slope (elevation gradient over the smoothed grid)
        -> Mark/display slope results

Marking strategy:
  - LAS `classification` field: 2 = ground/surface, 7 = object/outlier
    (same convention as the earlier ground-filter script)
  - LAS `user_data` field: local slope in degrees at that point's grid cell,
    rounded and clipped to 0-255 (slope is 0-90 deg so this fits easily).
    CloudCompare imports `User Data` as a per-point scalar field
    automatically, so you can colorize it as a slope heatmap with no
    extra setup.
  - A JSON summary with max_slope, max_slope_location, average_slope, and
    (for reasonably sized selections) the full per-cell slope grid.

If the input file is already the boundary-selected region (e.g. exported
from an earlier boundary pick), just pass it directly with no --polygon-file.
If you need to filter a larger file down to a polygon first, pass
--polygon-file pointing at a JSON file: {"polygon": [{"x":..,"y":..}, ...]}

Usage:
  python las_boundary_slope.py input.las output_prefix \
      [--polygon-file polygon.json] \
      [--grid-size 1.0] [--height-threshold 0.3] [--opening-size 7] \
      [--max-grid-cells-in-json 200000] [--chunk-size 2000000]

Required: pip install laspy numpy scipy
"""

import sys
import json
import argparse
import numpy as np
import laspy

try:
    from scipy.ndimage import grey_opening, distance_transform_edt
except ImportError:
    print("[ERROR] This script requires scipy. Install with: pip install scipy", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Optional polygon (boundary) filtering
# ---------------------------------------------------------------------------

def load_polygon(polygon_file):
    with open(polygon_file, "r") as f:
        data = json.load(f)
    polygon = data["polygon"]
    poly_x = np.array([p["x"] for p in polygon], dtype=np.float64)
    poly_y = np.array([p["y"] for p in polygon], dtype=np.float64)
    return poly_x, poly_y


def points_in_polygon(x, y, poly_x, poly_y):
    """Vectorized point-in-polygon test (ray casting / PNPOLY)."""
    n = len(poly_x)
    inside = np.zeros(len(x), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        denom = (yj - yi)
        denom = denom if abs(denom) > 1e-12 else 1e-12
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / denom + xi)
        inside ^= cond
        j = i
    return inside


# ---------------------------------------------------------------------------
# Pass 1: per-cell minimum Z (streaming, memory = O(grid cells))
# ---------------------------------------------------------------------------

def build_min_height_grid(input_path, grid_size, chunk_size, poly_x, poly_y):
    with laspy.open(input_path) as reader:
        header = reader.header
        total_points = header.point_count

        if poly_x is not None:
            x_min, x_max = poly_x.min(), poly_x.max()
            y_min, y_max = poly_y.min(), poly_y.max()
        else:
            x_min, x_max = header.x_min, header.x_max
            y_min, y_max = header.y_min, header.y_max

        nx = int(np.ceil((x_max - x_min) / grid_size)) + 1
        ny = int(np.ceil((y_max - y_min) / grid_size)) + 1
        print(f"[INFO] File contains {total_points:,} points", file=sys.stderr)
        print(f"[INFO] Grid: {nx} x {ny} cells at {grid_size}m resolution "
              f"({nx * ny:,} cells)", file=sys.stderr)

        min_grid = np.full((nx, ny), np.inf, dtype=np.float32)
        has_data = np.zeros((nx, ny), dtype=bool)

        n_seen = 0
        n_in_boundary = 0
        for chunk in reader.chunk_iterator(chunk_size):
            x = np.asarray(chunk.x, dtype=np.float64)
            y = np.asarray(chunk.y, dtype=np.float64)
            z = np.asarray(chunk.z, dtype=np.float64)

            if poly_x is not None:
                box_mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
                keep = np.zeros(len(x), dtype=bool)
                if np.any(box_mask):
                    keep[box_mask] = points_in_polygon(x[box_mask], y[box_mask], poly_x, poly_y)
                x, y, z = x[keep], y[keep], z[keep]

            if len(x) > 0:
                ix = np.clip(((x - x_min) / grid_size).astype(np.int64), 0, nx - 1)
                iy = np.clip(((y - y_min) / grid_size).astype(np.int64), 0, ny - 1)
                np.minimum.at(min_grid, (ix, iy), z.astype(np.float32))
                has_data[ix, iy] = True
                n_in_boundary += len(x)

            n_seen += len(chunk)
            if n_seen % 20_000_000 < chunk_size:
                print(f"[INFO] Grid pass: {n_seen:,}/{total_points:,} points scanned "
                      f"({n_in_boundary:,} inside boundary so far)", file=sys.stderr)

    if n_in_boundary == 0:
        raise ValueError("No points found inside the given boundary/file extent")

    print(f"[INFO] {n_in_boundary:,} points fall inside the selected boundary", file=sys.stderr)
    return min_grid, has_data, header, x_min, y_min


def smooth_ground_grid(min_grid, has_data, opening_size):
    if np.any(~has_data):
        idx = distance_transform_edt(~has_data, return_distances=False, return_indices=True)
        filled = min_grid[tuple(idx)]
        print(f"[INFO] Filled {np.sum(~has_data):,} empty grid cells via nearest-neighbor "
              f"(will be excluded from slope stats)", file=sys.stderr)
    else:
        filled = min_grid

    smoothed = grey_opening(filled, size=(opening_size, opening_size))
    return smoothed.astype(np.float32)


# ---------------------------------------------------------------------------
# Slope calculation over the smoothed ground grid (Section 7 of the doc)
# ---------------------------------------------------------------------------

def compute_slope_grid(ground_grid, grid_size):
    dzdy, dzdx = np.gradient(ground_grid, grid_size)  # np.gradient axis order matches array axes
    slope_magnitude = np.sqrt(dzdx ** 2 + dzdy ** 2)
    slope_deg = np.degrees(np.arctan(slope_magnitude))
    return slope_deg.astype(np.float32)


def summarize_slope(slope_grid, has_data, grid_size, x_min, y_min,
                     max_grid_cells_in_json, slope_threshold=None):
    valid_slopes = slope_grid[has_data]
    if valid_slopes.size == 0:
        raise ValueError("No valid grid cells to compute slope statistics from")

    max_slope = float(np.max(valid_slopes))
    average_slope = float(np.mean(valid_slopes))

    max_idx_flat = np.argmax(np.where(has_data, slope_grid, -np.inf))
    max_ix, max_iy = np.unravel_index(max_idx_flat, slope_grid.shape)
    max_location = [
        float(x_min + (max_ix + 0.5) * grid_size),
        float(y_min + (max_iy + 0.5) * grid_size),
    ]

    summary = {
        "max_slope": round(max_slope, 2),
        "max_slope_location": [round(max_location[0], 3), round(max_location[1], 3)],
        "average_slope": round(average_slope, 2),
        "grid_size": grid_size,
    }

    def cell_list(mask):
        cells = []
        ix_list, iy_list = np.nonzero(mask)
        # Sort steepest-first so the most important cells are visible even
        # if the list later gets truncated/skimmed by the consumer.
        order = np.argsort(-slope_grid[ix_list, iy_list])
        for k in order:
            ix, iy = ix_list[k], iy_list[k]
            cells.append({
                "x": round(float(x_min + (ix + 0.5) * grid_size), 3),
                "y": round(float(y_min + (iy + 0.5) * grid_size), 3),
                "grid_size": grid_size,
                "slope": round(float(slope_grid[ix, iy]), 2),
            })
        return cells

    n_valid_cells = int(np.sum(has_data))
    if n_valid_cells <= max_grid_cells_in_json:
        summary["cells"] = cell_list(has_data)
    else:
        summary["cells"] = None
        summary["cells_omitted_reason"] = (
            f"{n_valid_cells:,} valid cells exceeds --max-grid-cells-in-json "
            f"({max_grid_cells_in_json:,}); only summary stats included. "
            f"Increase the limit or use a coarser --grid-size if you need the full grid."
        )

    if slope_threshold is not None:
        above_mask = has_data & (slope_grid > slope_threshold)
        n_above = int(np.sum(above_mask))
        summary["slope_threshold"] = slope_threshold
        summary["count_above_threshold"] = n_above

        if n_above <= max_grid_cells_in_json:
            summary["cells_above_threshold"] = cell_list(above_mask)
        else:
            summary["cells_above_threshold"] = None
            summary["cells_above_threshold_omitted_reason"] = (
                f"{n_above:,} cells exceed the threshold, which is more than "
                f"--max-grid-cells-in-json ({max_grid_cells_in_json:,}); only the "
                f"count is included. Raise the limit, raise the threshold, or use "
                f"a coarser --grid-size to shrink the list."
            )

    return summary


# ---------------------------------------------------------------------------
# Pass 2: stream through the file again, classify + mark + write
# ---------------------------------------------------------------------------

def stream_mark_and_write(input_path, output_prefix, ground_grid, slope_grid,
                           grid_size, x_min, y_min, height_threshold, chunk_size,
                           poly_x, poly_y):
    nx, ny = ground_grid.shape

    analyzed_path = f"{output_prefix}_analyzed.las"
    clean_path = f"{output_prefix}_clean.las"
    outliers_path = f"{output_prefix}_outliers.las"

    n_total = 0
    n_ground = 0

    with laspy.open(input_path) as reader:
        header = reader.header
        with laspy.open(analyzed_path, mode="w", header=header) as w_analyzed, \
             laspy.open(clean_path, mode="w", header=header) as w_clean, \
             laspy.open(outliers_path, mode="w", header=header) as w_outliers:

            for chunk in reader.chunk_iterator(chunk_size):
                x = np.asarray(chunk.x, dtype=np.float64)
                y = np.asarray(chunk.y, dtype=np.float64)
                z = np.asarray(chunk.z, dtype=np.float64)

                if poly_x is not None:
                    box_mask = (x >= x_min) & (y >= y_min)
                    in_boundary = np.zeros(len(x), dtype=bool)
                    if np.any(box_mask):
                        in_boundary[box_mask] = points_in_polygon(
                            x[box_mask], y[box_mask], poly_x, poly_y
                        )
                    chunk = chunk[in_boundary]
                    x, y, z = x[in_boundary], y[in_boundary], z[in_boundary]

                if len(chunk) == 0:
                    continue

                ix = np.clip(((x - x_min) / grid_size).astype(np.int64), 0, nx - 1)
                iy = np.clip(((y - y_min) / grid_size).astype(np.int64), 0, ny - 1)

                local_ground_z = ground_grid[ix, iy]
                height_above_ground = z - local_ground_z
                inlier_mask = np.abs(height_above_ground) <= height_threshold

                local_slope = slope_grid[ix, iy]
                user_data_values = np.clip(np.round(local_slope), 0, 255).astype(np.uint8)

                chunk.classification[inlier_mask] = 2   # ASPRS: Ground
                chunk.classification[~inlier_mask] = 7  # ASPRS: Low Point (noise) / object
                chunk.user_data[:] = user_data_values

                w_analyzed.write_points(chunk)
                if np.any(inlier_mask):
                    w_clean.write_points(chunk[inlier_mask])
                if np.any(~inlier_mask):
                    w_outliers.write_points(chunk[~inlier_mask])

                n_total += len(chunk)
                n_ground += int(np.sum(inlier_mask))

    print(f"[SUCCESS] Wrote {analyzed_path} ({n_total:,} points; "
          f"classification=ground/object, user_data=slope degrees)", file=sys.stderr)
    print(f"[SUCCESS] Wrote {clean_path} ({n_ground:,} points)", file=sys.stderr)
    print(f"[SUCCESS] Wrote {outliers_path} ({n_total - n_ground:,} points)", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Boundary-based ground filtering + slope calculation for LAS files")
    parser.add_argument("input_las", help="Path to input .las/.laz file (already boundary-clipped, or use --polygon-file)")
    parser.add_argument("output_prefix", help="Prefix for output files")
    parser.add_argument("--polygon-file", default=None,
                         help="Optional JSON file: {\"polygon\": [{\"x\":..,\"y\":..}, ...]} to filter to a boundary")
    parser.add_argument("--grid-size", type=float, default=1.0,
                         help="Grid cell size in file units, typically meters (default: 1.0)")
    parser.add_argument("--height-threshold", type=float, default=0.3,
                         help="Max height above/below local ground surface to still count as ground (default: 0.3)")
    parser.add_argument("--opening-size", type=int, default=7,
                         help="Morphological opening window, in grid cells (default: 7)")
    parser.add_argument("--max-grid-cells-in-json", type=int, default=200_000,
                         help="Max number of cells to include in the JSON per-cell grid before falling back to summary-only (default: 200,000)")
    parser.add_argument("--slope-threshold", type=float, default=45,
                         help="If set, list ALL grid cells with slope strictly greater than this value "
                              "(degrees), sorted steepest-first, in the JSON under 'cells_above_threshold'")
    parser.add_argument("--chunk-size", type=int, default=2_000_000,
                         help="Points per streaming chunk (default: 2,000,000)")
    args = parser.parse_args()

    try:
        poly_x, poly_y = (None, None)
        if args.polygon_file:
            poly_x, poly_y = load_polygon(args.polygon_file)

        min_grid, has_data, header, x_min, y_min = build_min_height_grid(
            args.input_las, args.grid_size, args.chunk_size, poly_x, poly_y
        )
        ground_grid = smooth_ground_grid(min_grid, has_data, args.opening_size)
        slope_grid = compute_slope_grid(ground_grid, args.grid_size)

        summary = summarize_slope(
            slope_grid, has_data, args.grid_size, x_min, y_min,
            args.max_grid_cells_in_json, slope_threshold=args.slope_threshold
        )
        summary_path = f"{args.output_prefix}_slope_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[SUCCESS] Wrote {summary_path} "
              f"(max_slope={summary['max_slope']}, average_slope={summary['average_slope']})",
              file=sys.stderr)

        stream_mark_and_write(
            args.input_las, args.output_prefix, ground_grid, slope_grid,
            args.grid_size, x_min, y_min, args.height_threshold, args.chunk_size,
            poly_x, poly_y
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()