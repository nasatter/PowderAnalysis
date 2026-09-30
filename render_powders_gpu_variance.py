#!/usr/bin/env python3
"""Fast GPU rendering and batch analysis of COMSOL powder-grain exports.

Each grain is drawn as a sphere impostor: a single instanced quad whose
fragments compute a sphere surface and correct depth on the GPU.  This is much
faster and more memory-efficient than building a triangle mesh per sphere.

Install:
    python -m pip install numpy moderngl glfw Pillow matplotlib

Examples:
    python render_powders_gpu.py "results 1_2_3.txt"
    python render_powders_gpu.py particles.txt --color diameter --msaa 4
    python render_powders_gpu.py particles.txt --screenshot powder.png
    python render_powders_gpu.py ./txt_files --output-dir ./powder_results

Directory mode processes every .txt file, writes exact .npy heightmaps and
.png previews named after the source files, and creates powder_features.csv
with transverse roughness, convex-hull relative-density, and coordination features. It also
writes coordination-number statistics and one representative filled-particle/
hull density-slice image per file. All files share one outlier-filtered
transverse domain and one mm/pixel value; spreading-axis length is detected
separately for each file.

Controls:
    Left drag   orbit
    Right drag  pan
    Wheel       zoom
    R           reset camera
    S           save screenshot
    Esc         quit
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import multiprocessing as mp
import os
import re
from pathlib import Path
import sys

import numpy as np


VERTEX_SHADER = r"""
#version 330

uniform mat4 u_view;
uniform mat4 u_proj;

in vec4 in_sphere;  // xyz center and radius, in world units
in vec3 in_color;

out vec2 v_disk;
out vec3 v_center_view;
out float v_radius;
out vec3 v_color;

const vec2 CORNERS[6] = vec2[6](
    vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
    vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

void main() {
    vec2 corner = CORNERS[gl_VertexID];
    vec3 center_view = (u_view * vec4(in_sphere.xyz, 1.0)).xyz;
    vec3 surface_view = center_view + vec3(corner * in_sphere.w, 0.0);
    gl_Position = u_proj * vec4(surface_view, 1.0);
    v_disk = corner;
    v_center_view = center_view;
    v_radius = in_sphere.w;
    v_color = in_color;
}
"""


FRAGMENT_SHADER = r"""
#version 330

uniform mat4 u_proj;
uniform vec3 u_light_view;
in vec2 v_disk;
in vec3 v_center_view;
in float v_radius;
in vec3 v_color;

out vec4 frag_color;

void main() {
    float rr = dot(v_disk, v_disk);
    if (rr > 1.0) discard;

    // Front hemisphere in eye space; camera looks toward negative z.
    vec3 normal = vec3(v_disk, sqrt(max(0.0, 1.0 - rr)));
    vec3 surface_view = v_center_view + normal * v_radius;

    // Write the real curved-sphere depth instead of the billboard depth.
    vec4 clip = u_proj * vec4(surface_view, 1.0);
    float ndc_depth = clip.z / clip.w;
    gl_FragDepth = 0.5 * ndc_depth + 0.5;

    vec3 light_dir = normalize(u_light_view - surface_view);
    vec3 view_dir = normalize(-surface_view);
    vec3 half_dir = normalize(light_dir + view_dir);
    float diffuse = max(dot(normal, light_dir), 0.0);
    float specular = pow(max(dot(normal, half_dir), 0.0), 42.0);
    float rim = pow(1.0 - max(dot(normal, view_dir), 0.0), 3.0);
    vec3 color = v_color * (0.24 + 0.72 * diffuse) + 0.24 * specular;
    color += 0.06 * rim * mix(v_color, vec3(1.0), 0.4);
    frag_color = vec4(color, 1.0);
}
"""


HEIGHTMAP_VERTEX_SHADER = r"""
#version 330

uniform mat4 u_view;
uniform mat4 u_proj;

in vec4 in_sphere;
out vec2 v_disk;
out vec3 v_center_view;
out float v_radius;

const vec2 CORNERS[6] = vec2[6](
    vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
    vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

void main() {
    vec2 corner = CORNERS[gl_VertexID];
    vec3 center_view = (u_view * vec4(in_sphere.xyz, 1.0)).xyz;
    vec3 surface_view = center_view + vec3(corner * in_sphere.w, 0.0);
    gl_Position = u_proj * vec4(surface_view, 1.0);
    v_disk = corner;
    v_center_view = center_view;
    v_radius = in_sphere.w;
}
"""


HEIGHTMAP_FRAGMENT_SHADER = r"""
#version 330

uniform mat4 u_proj;

in vec2 v_disk;
in vec3 v_center_view;
in float v_radius;
out vec4 frag_color;

void main() {
    float rr = dot(v_disk, v_disk);
    if (rr > 1.0) discard;

    // With the camera directly above the bed, this is the sphere's top surface.
    float dz = sqrt(max(0.0, 1.0 - rr));
    vec3 surface_view = v_center_view + vec3(v_disk, dz) * v_radius;
    vec4 clip = u_proj * vec4(surface_view, 1.0);
    gl_FragDepth = 0.5 * (clip.z / clip.w) + 0.5;
    frag_color = vec4(1.0);
}
"""


def load_comsol_grains(
    path: Path,
    filter_outliers: bool = True,
    outlier_mad_threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Return valid grains and diagnostic counts for one COMSOL export."""
    try:
        data = np.loadtxt(path, comments="%", dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Could not parse {path}: {exc}") from exc
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 6:
        raise RuntimeError(
            f"Expected at least 6 columns (index, contacts, diameter, x, y, z); "
            f"found {data.shape[1]}."
        )
    centers = np.ascontiguousarray(data[:, 3:6], dtype=np.float32)
    diameters = np.ascontiguousarray(data[:, 2], dtype=np.float32)
    contacts = np.ascontiguousarray(data[:, 1], dtype=np.float32)
    basic_valid = (
        np.isfinite(centers).all(axis=1)
        & np.isfinite(diameters)
        & (diameters > 0)
    )
    if not np.any(basic_valid):
        raise RuntimeError("No grains have finite positions and positive diameters.")

    valid_indices = np.flatnonzero(basic_valid)
    coordinate_outliers = np.zeros(len(data), dtype=bool)
    if filter_outliers and len(valid_indices) >= 4:
        candidate_centers = centers[valid_indices].astype(np.float64)
        median = np.median(candidate_centers, axis=0)
        mad = np.median(np.abs(candidate_centers - median), axis=0)
        robust_scale = 1.4826 * mad

        # Axes with no measurable spread cannot provide a meaningful score.
        usable_axis = robust_scale > np.finfo(np.float64).eps * np.maximum(np.abs(median), 1.0)
        candidate_outliers = np.zeros(len(candidate_centers), dtype=bool)
        if np.any(usable_axis):
            robust_distance = np.abs(
                (candidate_centers[:, usable_axis] - median[usable_axis])
                / robust_scale[usable_axis]
            )
            candidate_outliers = np.any(
                robust_distance > outlier_mad_threshold, axis=1
            )
        coordinate_outliers[valid_indices[candidate_outliers]] = True

    retained = basic_valid & ~coordinate_outliers
    if not np.any(retained):
        raise RuntimeError("Coordinate outlier rejection removed every valid grain.")

    diagnostics: dict[str, object] = {
        "input_particle_rows": int(len(data)),
        "invalid_particle_rows_removed": int(np.count_nonzero(~basic_valid)),
        "coordinate_outlier_rows_removed": int(np.count_nonzero(coordinate_outliers)),
        "rendered_particle_rows": int(np.count_nonzero(retained)),
        "coordinate_outlier_filter": "median_absolute_deviation" if filter_outliers else "disabled",
        "coordinate_outlier_mad_threshold": (
            float(outlier_mad_threshold) if filter_outliers else None
        ),
    }
    return centers[retained], diameters[retained], contacts[retained], diagnostics


def compute_coordination_numbers(
    centers_mm: np.ndarray,
    radii_mm: np.ndarray,
    eps_mm: float = 1e-3,
) -> np.ndarray:
    """Count touching neighbours using the supplied surface-gap definition.

    A pair is counted when ``center_distance - radius_i - radius_j`` is no
    greater than ``eps_mm``. This includes tangent particles and small DEM
    overlaps. A spatial grid avoids constructing the full
    N-by-N distance matrix used by the original function while preserving the
    same contact test.
    """
    centers_mm = np.asarray(centers_mm, dtype=np.float64)
    radii_mm = np.asarray(radii_mm, dtype=np.float64)
    if centers_mm.ndim != 2 or centers_mm.shape[1] != 3:
        raise ValueError("centers_mm must have shape (N, 3)")
    if radii_mm.shape != (len(centers_mm),):
        raise ValueError("radii_mm must have shape (N,)")
    if eps_mm < 0.0:
        raise ValueError("eps_mm must be nonnegative")

    count = len(centers_mm)
    coordination = np.zeros(count, dtype=np.int32)
    if count < 2:
        return coordination

    # Any touching pair must lie in the same or an adjacent grid cell because
    # its center distance cannot exceed 2 * max(radius) + eps.
    cell_size = max(2.0 * float(np.max(radii_mm)) + eps_mm, np.finfo(float).eps)
    cell_coordinates = np.floor(centers_mm / cell_size).astype(np.int64)
    cell_members: dict[tuple[int, int, int], list[int]] = {}
    for particle_index, cell in enumerate(cell_coordinates):
        cell_members.setdefault(tuple(int(value) for value in cell), []).append(particle_index)

    neighbour_offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]
    for particle_index, cell in enumerate(cell_coordinates):
        candidates: list[int] = []
        cx, cy, cz = (int(value) for value in cell)
        for dx, dy, dz in neighbour_offsets:
            candidates.extend(cell_members.get((cx + dx, cy + dy, cz + dz), ()))
        candidate_indices = np.asarray(candidates, dtype=np.int64)
        candidate_indices = candidate_indices[candidate_indices > particle_index]
        if candidate_indices.size == 0:
            continue

        center_distances = np.linalg.norm(
            centers_mm[candidate_indices] - centers_mm[particle_index], axis=1
        )
        surface_gaps = (
            center_distances
            - radii_mm[particle_index]
            - radii_mm[candidate_indices]
        )
        touching_indices = candidate_indices[surface_gaps <= eps_mm]
        coordination[particle_index] += int(touching_indices.size)
        np.add.at(coordination, touching_indices, 1)
    return coordination


def _fill_short_false_runs(mask: np.ndarray, maximum_gap: int) -> np.ndarray:
    """Fill short False runs that are bounded by accepted bins on both sides."""
    out = np.asarray(mask, dtype=bool).copy()
    if maximum_gap <= 0 or len(out) < 3:
        return out
    index = 0
    while index < len(out):
        if out[index]:
            index += 1
            continue
        stop = index
        while stop < len(out) and not out[stop]:
            stop += 1
        if index > 0 and stop < len(out) and stop - index <= maximum_gap:
            out[index:stop] = True
        index = stop
    return out


def _gaussian_low_pass(values: np.ndarray, pixel_size_mm: float, width_mm: float) -> np.ndarray:
    """Low-pass a one-dimensional profile with a Gaussian of the requested FWHM."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3 or width_mm <= pixel_size_mm:
        return values.copy()
    sigma_pixels = max(width_mm / 2.354820045 / pixel_size_mm, 0.5)
    radius = max(2, int(math.ceil(3.0 * sigma_pixels)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_pixels) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def determine_spreading_roi_from_heightmap(
    heightmap_mm: np.ndarray,
    metadata: dict[str, object],
    spread_axis: str,
    substrate_height_mm: float,
    filter_width_mm: float,
    diagnostic_path: Path | None = None,
    source_label: str = "",
) -> dict[str, object]:
    """Find the stable powder-bed plateau from the height profile along spreading.

    The exact rendered heightmap is reduced to a transverse mean surface-height
    profile after replacing empty pixels with the substrate height. A Gaussian
    low-pass filter suppresses particle-scale fluctuations. The nominal bed
    height is estimated from the early/middle portion of the spread, before the
    terminal pile-up, and the ROI is the longest stable run through that anchor
    region whose filtered height and slope remain close to the nominal plateau.
    This explicitly trims the leading and trailing transition legs.
    """
    heights = heightmap_mm if spread_axis == "x" else heightmap_mm.T
    measurement = np.where(np.isfinite(heights), heights, substrate_height_mm)
    raw_height_mm = np.mean(measurement, axis=0, dtype=np.float64)

    bounds = metadata["raster_bounds_mm"]
    pixel_size_mm = float(metadata["pixel_size_mm"])
    if spread_axis == "x":
        positions_mm = (
            float(bounds["x"][0])
            + (np.arange(raw_height_mm.size, dtype=np.float64) + 0.5) * pixel_size_mm
        )
    else:
        positions_mm = (
            float(bounds["y"][1])
            - (np.arange(raw_height_mm.size, dtype=np.float64) + 0.5) * pixel_size_mm
        )
        # Work in increasing physical coordinate regardless of array orientation.
        positions_mm = positions_mm[::-1]
        raw_height_mm = raw_height_mm[::-1]

    if raw_height_mm.size < 5:
        start_mm = float(positions_mm[0] - 0.5 * pixel_size_mm)
        end_mm = float(positions_mm[-1] + 0.5 * pixel_size_mm)
        return {
            "spreading_roi_start_mm": start_mm,
            "spreading_roi_end_mm": end_mm,
            "spreading_roi_pixels_included": int(raw_height_mm.size),
            "spreading_roi_pixels_total": int(raw_height_mm.size),
            "spreading_roi_filter_width_mm": float(filter_width_mm),
            "spreading_roi_nominal_height_mm": float(np.median(raw_height_mm)),
            "spreading_roi_height_half_band_mm": 0.0,
            "spreading_roi_slope_limit_mm_per_mm": float("inf"),
            "spreading_roi_profile_definition": "transverse mean of absolute heightmap with empty pixels set to substrate",
        }

    filtered_height_mm = _gaussian_low_pass(
        raw_height_mm, pixel_size_mm=pixel_size_mm, width_mm=filter_width_mm
    )
    count = len(filtered_height_mm)

    # The stable powder bed occurs before the terminal pile-up in these spreads.
    # Use a broad early/middle anchor rather than the global height mode so a
    # wide pile-up cannot become the nominal level.
    anchor_start = int(round(0.15 * count))
    anchor_end = max(anchor_start + 5, int(round(0.55 * count)))
    anchor_end = min(anchor_end, count)
    anchor = filtered_height_mm[anchor_start:anchor_end]
    nominal_height_mm = float(np.median(anchor))
    anchor_mad_mm = float(
        1.4826 * np.median(np.abs(anchor - nominal_height_mm))
    )

    # A robust height band defines the plateau; the slope test removes both
    # transition legs even if their heights temporarily pass through the band.
    height_half_band_mm = max(
        3.0 * anchor_mad_mm,
        0.25 * max(abs(nominal_height_mm - substrate_height_mm), 0.1),
        0.15,
    )
    slope_mm_per_mm = np.gradient(filtered_height_mm, positions_mm)
    slope_limit = max(
        0.04,
        2.5 * anchor_mad_mm / max(filter_width_mm, pixel_size_mm),
        0.12 * max(abs(nominal_height_mm - substrate_height_mm), 0.1)
        / max(filter_width_mm, pixel_size_mm),
    )

    accepted = (
        (np.abs(filtered_height_mm - nominal_height_mm) <= height_half_band_mm)
        & (np.abs(slope_mm_per_mm) <= slope_limit)
    )
    maximum_gap_pixels = max(
        1, int(round(0.5 * filter_width_mm / max(pixel_size_mm, 1e-12)))
    )
    accepted = _fill_short_false_runs(accepted, maximum_gap_pixels)

    padded = np.r_[False, accepted, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    candidates = [
        (int(start), int(stop))
        for start, stop in zip(starts, stops)
        if stop > anchor_start and start < anchor_end
    ]
    if candidates:
        start_pixel, end_pixel = max(candidates, key=lambda pair: pair[1] - pair[0])
    elif starts.size:
        lengths = stops - starts
        winner = int(np.argmax(lengths))
        start_pixel, end_pixel = int(starts[winner]), int(stops[winner])
    else:
        start_pixel, end_pixel = anchor_start, anchor_end

    minimum_pixels = max(5, int(math.ceil(0.20 * count)))
    if end_pixel - start_pixel < minimum_pixels:
        # If the slope criterion is too restrictive, retain the height plateau
        # but still choose the run that crosses the early/middle anchor.
        height_only = np.abs(filtered_height_mm - nominal_height_mm) <= height_half_band_mm
        height_only = _fill_short_false_runs(height_only, maximum_gap_pixels)
        padded = np.r_[False, height_only, False].astype(np.int8)
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        stops = np.flatnonzero(changes == -1)
        candidates = [
            (int(start), int(stop))
            for start, stop in zip(starts, stops)
            if stop > anchor_start and start < anchor_end
        ]
        if candidates:
            start_pixel, end_pixel = max(candidates, key=lambda pair: pair[1] - pair[0])
        elif starts.size:
            winner = int(np.argmax(stops - starts))
            start_pixel, end_pixel = int(starts[winner]), int(stops[winner])

    start_mm = float(positions_mm[start_pixel] - 0.5 * pixel_size_mm)
    end_mm = float(positions_mm[end_pixel - 1] + 0.5 * pixel_size_mm)
    result: dict[str, object] = {
        "spreading_roi_start_mm": start_mm,
        "spreading_roi_end_mm": end_mm,
        "spreading_roi_pixels_included": int(end_pixel - start_pixel),
        "spreading_roi_pixels_total": int(count),
        "spreading_roi_filter_width_mm": float(filter_width_mm),
        "spreading_roi_nominal_height_mm": nominal_height_mm,
        "spreading_roi_height_half_band_mm": float(height_half_band_mm),
        "spreading_roi_slope_limit_mm_per_mm": float(slope_limit),
        "spreading_roi_anchor_start_fraction": 0.15,
        "spreading_roi_anchor_end_fraction": 0.55,
        "spreading_roi_profile_definition": "transverse mean of absolute heightmap with empty pixels set to substrate",
        "spreading_axis": spread_axis,
    }

    if diagnostic_path is not None:
        save_spreading_roi_height_plot(
            diagnostic_path,
            positions_mm=positions_mm,
            raw_height_mm=raw_height_mm,
            filtered_height_mm=filtered_height_mm,
            spreading_roi=result,
            substrate_height_mm=substrate_height_mm,
            source_label=source_label,
        )
        result["spreading_roi_height_plot"] = diagnostic_path.name
    return result


def save_spreading_roi_height_plot(
    path: Path,
    positions_mm: np.ndarray,
    raw_height_mm: np.ndarray,
    filtered_height_mm: np.ndarray,
    spreading_roi: dict[str, object],
    substrate_height_mm: float,
    source_label: str,
) -> None:
    """Save the raw/low-pass height profile and the accepted spreading ROI."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "ROI diagnostic plots require matplotlib: python -m pip install matplotlib"
        ) from exc

    start_mm = float(spreading_roi["spreading_roi_start_mm"])
    end_mm = float(spreading_roi["spreading_roi_end_mm"])
    nominal = float(spreading_roi["spreading_roi_nominal_height_mm"])
    half_band = float(spreading_roi["spreading_roi_height_half_band_mm"])

    figure, axis = plt.subplots(figsize=(11.0, 5.2))
    axis.plot(positions_mm, raw_height_mm, linewidth=1.0, alpha=0.55, label="Raw mean height")
    axis.plot(positions_mm, filtered_height_mm, linewidth=2.0, label="Low-pass height")
    axis.axvspan(start_mm, end_mm, alpha=0.18, label="Analysis ROI")
    axis.axhline(nominal, linestyle="--", linewidth=1.0, label="Nominal bed height")
    axis.axhline(nominal - half_band, linestyle=":", linewidth=0.9)
    axis.axhline(nominal + half_band, linestyle=":", linewidth=0.9)
    if substrate_height_mm != 0.0:
        axis.axhline(substrate_height_mm, linestyle="-.", linewidth=0.9, label="Substrate")
    axis.set_xlabel(f"{spreading_roi.get('spreading_axis', 'x')} position [mm]")
    axis.set_ylabel("Absolute powder height [mm]")
    axis.set_title(f"{source_label} - spreading ROI from powder-height profile")
    axis.legend(loc="best")
    axis.grid(alpha=0.15)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)
    print(f"Saved spreading ROI height plot {path}")

def extract_coordination_features(
    centers: np.ndarray,
    diameters: np.ndarray,
    tolerance_mm: float,
    spreading_roi: dict[str, object],
    spread_axis: str,
) -> dict[str, object]:
    """Summarize coordination only inside the accepted spreading ROI.

    Contact counts are still computed against all retained particles so a
    particle near the ROI boundary does not lose a real neighbour merely
    because that neighbour lies just outside the reporting region.
    """
    centers_mm = np.asarray(centers, dtype=np.float64) * 1000.0
    radii_mm = np.asarray(diameters, dtype=np.float64) * 500.0
    coordination = compute_coordination_numbers(
        centers_mm=centers_mm,
        radii_mm=radii_mm,
        eps_mm=tolerance_mm,
    )
    spread_index = 0 if spread_axis == "x" else 1
    roi_start = float(spreading_roi["spreading_roi_start_mm"])
    roi_end = float(spreading_roi["spreading_roi_end_mm"])
    in_roi = (centers_mm[:, spread_index] >= roi_start) & (centers_mm[:, spread_index] < roi_end)
    if not np.any(in_roi):
        raise RuntimeError("No particles lie inside the accepted spreading ROI.")
    selected = coordination[in_roi]
    return {
        "mean_coordination_number": float(np.mean(selected)),
        "coordination_number_variance": float(np.var(selected, ddof=0)),
        "coordination_number_std": float(np.std(selected, ddof=0)),
        "median_coordination_number": float(np.median(selected)),
        "minimum_coordination_number": int(np.min(selected)),
        "maximum_coordination_number": int(np.max(selected)),
        "particles_with_at_least_one_contact": int(np.count_nonzero(selected)),
        "coordination_particles_evaluated": int(len(selected)),
        "coordination_particles_total_retained": int(len(coordination)),
        "coordination_tolerance_mm": float(tolerance_mm),
    }


def colorize(values: np.ndarray, mode: str, centers: np.ndarray) -> np.ndarray:
    if mode == "uniform":
        return np.tile(np.array([[0.74, 0.53, 0.25]], np.float32), (len(values), 1))
    scalar = centers[:, 2] if mode == "z" else values
    lo, hi = np.percentile(scalar, [1.0, 99.0])
    t = np.clip((scalar - lo) / max(float(hi - lo), 1e-20), 0.0, 1.0)
    # Compact perceptually ordered blue -> cyan -> yellow -> red map.
    stops = np.array(
        [[0.10, 0.20, 0.58], [0.08, 0.72, 0.78], [0.98, 0.82, 0.22], [0.78, 0.12, 0.08]],
        dtype=np.float32,
    )
    x = t * (len(stops) - 1)
    i = np.minimum(x.astype(np.int32), len(stops) - 2)
    f = (x - i)[:, None]
    return np.ascontiguousarray(stops[i] * (1.0 - f) + stops[i + 1] * f, dtype=np.float32)


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / max(float(n), 1e-20)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)
    out = np.eye(4, dtype=np.float32)
    out[0, :3], out[1, :3], out[2, :3] = s, u, -f
    out[:3, 3] = -out[:3, :3] @ eye
    return out


def perspective(fovy: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fovy / 2.0)
    out = np.zeros((4, 4), dtype=np.float32)
    out[0, 0], out[1, 1] = f / aspect, f
    out[2, 2] = (far + near) / (near - far)
    out[2, 3] = 2.0 * far * near / (near - far)
    out[3, 2] = -1.0
    return out


def orthographic(left: float, right: float, bottom: float, top: float,
                 near: float, far: float) -> np.ndarray:
    out = np.eye(4, dtype=np.float32)
    out[0, 0] = 2.0 / (right - left)
    out[1, 1] = 2.0 / (top - bottom)
    out[2, 2] = -2.0 / (far - near)
    out[0, 3] = -(right + left) / (right - left)
    out[1, 3] = -(top + bottom) / (top - bottom)
    out[2, 3] = -(far + near) / (far - near)
    return out


class Camera:
    def __init__(self, centers: np.ndarray, radii: np.ndarray):
        lower = np.min(centers - radii[:, None], axis=0)
        upper = np.max(centers + radii[:, None], axis=0)
        self.initial_target = (lower + upper) * 0.5
        self.scene_radius = max(float(np.linalg.norm(upper - lower) * 0.5), 1e-6)
        self.reset()

    def reset(self) -> None:
        self.target = self.initial_target.copy()
        self.yaw = math.radians(-72.0)
        self.pitch = math.radians(24.0)
        self.distance = 2.15 * self.scene_radius

    def eye(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        direction = np.array(
            [cp * math.cos(self.yaw), cp * math.sin(self.yaw), math.sin(self.pitch)],
            dtype=np.float32,
        )
        return self.target + self.distance * direction

    def view(self) -> np.ndarray:
        return look_at(self.eye(), self.target, np.array([0.0, 0.0, 1.0], np.float32))

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw -= 0.007 * dx
        self.pitch = float(np.clip(self.pitch + 0.007 * dy, -1.48, 1.48))

    def pan(self, dx: float, dy: float, height: int) -> None:
        eye = self.eye()
        forward = normalize(self.target - eye)
        right = normalize(np.cross(forward, np.array([0.0, 0.0, 1.0], np.float32)))
        up = normalize(np.cross(right, forward))
        scale = 2.0 * self.distance * math.tan(math.radians(45.0) / 2.0) / max(height, 1)
        self.target += (-right * dx + up * dy) * scale

    def zoom(self, amount: float) -> None:
        self.distance *= math.exp(-0.11 * amount)
        self.distance = float(np.clip(self.distance, self.scene_radius * 0.03, self.scene_radius * 50.0))


def save_framebuffer(ctx, width: int, height: int, path: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Saving images requires Pillow: python -m pip install Pillow") from exc
    pixels = ctx.screen.read(components=3, alignment=1)
    image = Image.frombytes("RGB", (width, height), pixels).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    print(f"Saved {path}")


def _heightmap_paths(output: Path) -> tuple[Path, Path, Path]:
    """Return exact-data, preview, and metadata paths from one output prefix."""
    prefix = output.with_suffix("") if output.suffix.lower() in {".npy", ".png", ".json"} else output
    return Path(f"{prefix}.npy"), Path(f"{prefix}.png"), Path(f"{prefix}.json")


def _largest_true_run(mask: np.ndarray) -> tuple[int, int]:
    """Return [start, stop) for the longest contiguous True run."""
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    if starts.size == 0:
        raise RuntimeError("Could not detect a sufficiently occupied powder-bed region.")
    index = int(np.argmax(stops - starts))
    return int(starts[index]), int(stops[index])



def extract_heightmap_features(
    heightmap_mm: np.ndarray,
    spread_axis: str,
    transverse_occupancy: float,
    active_fraction: float,
    substrate_height_mm: float,
    metadata: dict[str, object],
    spreading_roi: dict[str, object],
) -> dict[str, object]:
    """Extract scalar features from one exact heightmap.

    Add future features to the returned dictionary. Batch CSV columns are
    generated from these dictionary keys, so no CSV restructuring is needed.
    """
    # Internal layout: rows are transverse; columns follow spreading.
    heights = heightmap_mm if spread_axis == "x" else heightmap_mm.T
    occupied = np.isfinite(heights)

    bounds = metadata["raster_bounds_mm"]
    pixel_size_mm = float(metadata["pixel_size_mm"])
    if spread_axis == "x":
        spread_positions_mm = float(bounds["x"][0]) + (np.arange(heights.shape[1]) + 0.5) * pixel_size_mm
    else:
        spread_positions_mm = float(bounds["y"][1]) - (np.arange(heights.shape[1]) + 0.5) * pixel_size_mm
    roi_start = float(spreading_roi["spreading_roi_start_mm"])
    roi_end = float(spreading_roi["spreading_roi_end_mm"])
    in_spreading_roi = (spread_positions_mm >= roi_start) & (spread_positions_mm < roi_end)

    preliminary_active = (occupied.mean(axis=0) >= 0.25) & in_spreading_roi
    if not np.any(preliminary_active):
        raise RuntimeError("No occupied powder region was found in the heightmap.")
    transverse_fraction = occupied[:, preliminary_active].mean(axis=1)
    transverse_start, transverse_end = _largest_true_run(
        transverse_fraction >= transverse_occupancy
    )

    roi = heights[transverse_start:transverse_end, :]
    finite = np.isfinite(roi)
    occupied_fraction = finite.mean(axis=0)
    included = (occupied_fraction >= active_fraction) & in_spreading_roi
    if not np.any(included):
        raise RuntimeError("No spreading positions satisfy the active-fraction threshold.")

    # Missing pixels inside the fixed transverse region are exposed substrate.
    measurement = np.where(finite, roi, substrate_height_mm)
    roughness_by_position_mm = np.std(measurement, axis=0, ddof=0)
    included_roughness_mm = roughness_by_position_mm[included]

    return {
        "mean_transverse_rms_roughness_mm": float(np.mean(included_roughness_mm)),
        "transverse_rms_roughness_variance_mm2": float(np.var(included_roughness_mm, ddof=0)),
        "roughness_transverse_start_pixel": transverse_start,
        "roughness_transverse_end_pixel_exclusive": transverse_end,
        "roughness_slices_included": int(np.count_nonzero(included)),
        "roughness_slices_total": int(included.size),
        "roughness_slices_in_spreading_roi": int(np.count_nonzero(in_spreading_roi)),
    }


def _polygon_area(points: np.ndarray) -> float:
    """Return the unsigned area of a two-dimensional polygon."""
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Return unique two-dimensional points on the convex hull in CCW order."""
    unique = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if len(unique) < 3:
        return np.empty((0, 2), dtype=np.float64)
    ordered = unique[np.lexsort((unique[:, 1], unique[:, 0]))]

    def cross(origin: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - origin[0]) * (b[1] - origin[1])
                     - (a[1] - origin[1]) * (b[0] - origin[0]))

    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in ordered[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    return hull if _polygon_area(hull) > 0.0 else np.empty((0, 2), dtype=np.float64)


def _circle_polygon_intersection_area(
    center: np.ndarray,
    radius: float,
    polygon: np.ndarray,
) -> float:
    """Return the exact area of a circle clipped to a polygon.

    Each polygon edge is split at its circle intersections. Subsegments inside
    the circle contribute triangular area; outside subsegments contribute the
    corresponding circular-sector area through Green's theorem.
    """
    if radius <= 0.0 or len(polygon) < 3:
        return 0.0
    shifted = np.asarray(polygon, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    radius_squared = radius * radius
    area = 0.0
    for edge_index, start in enumerate(shifted):
        end = shifted[(edge_index + 1) % len(shifted)]
        delta = end - start
        a = float(np.dot(delta, delta))
        breakpoints = [0.0, 1.0]
        if a > 0.0:
            b = 2.0 * float(np.dot(start, delta))
            c = float(np.dot(start, start)) - radius_squared
            discriminant = b * b - 4.0 * a * c
            if discriminant > 0.0:
                root = math.sqrt(discriminant)
                for value in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
                    if 0.0 < value < 1.0:
                        breakpoints.append(value)
        breakpoints.sort()
        for first, second in zip(breakpoints[:-1], breakpoints[1:]):
            p = start + first * delta
            q = start + second * delta
            midpoint = start + 0.5 * (first + second) * delta
            cross_value = float(p[0] * q[1] - p[1] * q[0])
            if float(np.dot(midpoint, midpoint)) <= radius_squared:
                area += 0.5 * cross_value
            else:
                area += 0.5 * radius_squared * math.atan2(cross_value, float(np.dot(p, q)))
    return abs(area)


def _clipped_hull_slice_worker(
    task: tuple[int, np.ndarray, np.ndarray, np.ndarray, float],
) -> tuple[int, float, float]:
    """Compute clipped particle area for one density slice in a worker process."""
    (
        slice_index,
        section_centers,
        cross_radii,
        center_hull_points,
        center_hull_area_mm2,
    ) = task
    if center_hull_area_mm2 <= 0.0 or len(center_hull_points) < 3:
        return slice_index, 0.0, math.nan
    clipped_powder_area_mm2 = float(sum(
        _circle_polygon_intersection_area(center, float(radius), center_hull_points)
        for center, radius in zip(section_centers, cross_radii)
    ))
    return (
        slice_index,
        clipped_powder_area_mm2,
        clipped_powder_area_mm2 / center_hull_area_mm2,
    )


def extract_convex_hull_density(
    centers: np.ndarray,
    diameters: np.ndarray,
    metadata: dict[str, object],
    spread_axis: str,
    stack_height_factor: float,
    hull_directions: int,
    density_workers: int = 0,
    spreading_roi: dict[str, object] | None = None,
    diagnostic_path: Path | None = None,
    source_label: str = "",
) -> dict[str, object]:
    """Calculate fixed-hull and clipped center-hull area fractions.

    Fixed-hull geometry is evaluated in the parent process. The expensive
    circle/polygon clipping for qualifying slices is distributed across
    multiple spawned worker processes so the OpenGL context is never inherited
    by a child process. ``density_workers=0`` selects an automatic worker count;
    ``density_workers=1`` forces the original serial clipping behavior.
    """
    centers_mm = centers.astype(np.float64) * 1000.0
    radii_mm = diameters.astype(np.float64) * 500.0
    maximum_diameter_mm = float(np.max(diameters) * 1000.0)
    required_height_mm = stack_height_factor * maximum_diameter_mm
    stack_height_tolerance_mm = max(1e-9, 1e-6 * required_height_mm)

    shape = metadata["shape"]
    bounds = metadata["raster_bounds_mm"]
    pixel_size_mm = float(metadata["pixel_size_mm"])
    if spread_axis == "x":
        number_of_slices = int(shape[1])
        start = float(bounds["x"][0])
        slice_positions = start + (np.arange(number_of_slices) + 0.5) * pixel_size_mm
        axial_centers = centers_mm[:, 0]
        transverse_centers = centers_mm[:, [1, 2]]
    else:
        number_of_slices = int(shape[0])
        start = float(bounds["y"][1])
        slice_positions = start - (np.arange(number_of_slices) + 0.5) * pixel_size_mm
        axial_centers = centers_mm[:, 1]
        transverse_centers = centers_mm[:, [0, 2]]

    if spreading_roi is None:
        roi_start_mm, roi_end_mm = -math.inf, math.inf
    else:
        roi_start_mm = float(spreading_roi["spreading_roi_start_mm"])
        roi_end_mm = float(spreading_roi["spreading_roi_end_mm"])
    roi_slice_mask = (slice_positions >= roi_start_mm) & (slice_positions < roi_end_mm)

    angles = np.linspace(0.0, 2.0 * math.pi, hull_directions, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    densities: list[float] = []
    clipped_hull_densities: list[float] = []
    density_records: list[tuple[int, float, float]] = []
    fixed_records: list[tuple[int, float]] = []
    clipping_tasks: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, float]] = []
    clipping_results: dict[int, tuple[float, float]] = {}
    intersected_slices = 0
    clipped_slices = 0
    clipped_hull_slices_capped = 0
    center_hull_degenerate_slices = 0
    maximum_hull_height_mm = -math.inf
    maximum_height_slice_index: int | None = None
    worker_count_used = 1

    def slice_geometry(slice_index: int) -> dict[str, object] | None:
        position = float(slice_positions[slice_index])
        axial_offset = position - axial_centers
        active = np.abs(axial_offset) < radii_mm
        if not np.any(active):
            return None
        cross_radii = np.sqrt(
            np.maximum(radii_mm[active] ** 2 - axial_offset[active] ** 2, 0.0)
        )
        section_centers = transverse_centers[active]
        hull_height_mm = float(np.max(section_centers[:, 1] + cross_radii)
                               - np.min(section_centers[:, 1] - cross_radii))
        support_values = section_centers @ directions.T + cross_radii[:, None]
        winners = np.argmax(support_values, axis=0)
        hull_points = section_centers[winners] + cross_radii[winners, None] * directions
        transverse_coordinate = hull_points[:, 0]
        height_coordinate = hull_points[:, 1]
        hull_area_mm2 = 0.5 * abs(float(
            np.dot(transverse_coordinate, np.roll(height_coordinate, -1))
            - np.dot(height_coordinate, np.roll(transverse_coordinate, -1))
        ))
        powder_area_mm2 = float(math.pi * np.sum(cross_radii ** 2))
        density = powder_area_mm2 / hull_area_mm2 if hull_area_mm2 > 0.0 else math.nan

        center_hull_points = _convex_hull(section_centers)
        center_hull_area_mm2 = _polygon_area(center_hull_points)
        return {
            "slice_index": slice_index,
            "position_mm": position,
            "section_centers": section_centers,
            "cross_radii": cross_radii,
            "hull_points": hull_points,
            "hull_height_mm": hull_height_mm,
            "hull_area_mm2": hull_area_mm2,
            "powder_area_mm2": powder_area_mm2,
            "density": min(density, 1.0) if math.isfinite(density) else density,
            "center_hull_points": center_hull_points,
            "center_hull_area_mm2": center_hull_area_mm2,
            "clipped_powder_area_mm2": math.nan,
            "clipped_hull_density": math.nan,
        }

    # First pass: fixed-hull calculations remain vectorized in the parent. Only
    # qualifying center-hull clipping jobs are prepared for multiprocessing.
    for slice_index in range(number_of_slices):
        if not roi_slice_mask[slice_index]:
            continue
        geometry = slice_geometry(slice_index)
        if geometry is None:
            continue
        intersected_slices += 1
        hull_height_mm = float(geometry["hull_height_mm"])
        if hull_height_mm > maximum_hull_height_mm:
            maximum_hull_height_mm = hull_height_mm
            maximum_height_slice_index = slice_index
        if hull_height_mm <= required_height_mm + stack_height_tolerance_mm:
            continue
        hull_area_mm2 = float(geometry["hull_area_mm2"])
        if hull_area_mm2 <= 0.0:
            continue

        density = float(geometry["density"])
        raw_fixed_density = float(geometry["powder_area_mm2"]) / hull_area_mm2
        if raw_fixed_density > 1.0:
            clipped_slices += 1
        densities.append(density)
        fixed_records.append((slice_index, density))

        center_hull_area_mm2 = float(geometry["center_hull_area_mm2"])
        if center_hull_area_mm2 <= 0.0:
            center_hull_degenerate_slices += 1
            continue
        clipping_tasks.append((
            slice_index,
            np.asarray(geometry["section_centers"], dtype=np.float64),
            np.asarray(geometry["cross_radii"], dtype=np.float64),
            np.asarray(geometry["center_hull_points"], dtype=np.float64),
            center_hull_area_mm2,
        ))

    if clipping_tasks:
        if density_workers < 0:
            raise ValueError("density_workers must be nonnegative")
        if density_workers == 0:
            available = max(1, (os.cpu_count() or 2) - 1)
            worker_count_used = min(16, available, len(clipping_tasks))
        else:
            worker_count_used = min(max(1, density_workers), len(clipping_tasks))

        if worker_count_used == 1:
            clip_iterator = map(_clipped_hull_slice_worker, clipping_tasks)
            for slice_index, clipped_area, raw_clipped_density in clip_iterator:
                clipping_results[slice_index] = (clipped_area, raw_clipped_density)
        else:
            chunksize = max(1, len(clipping_tasks) // (worker_count_used * 4))
            spawn_context = mp.get_context("spawn")
            print(
                f"Clipped-hull density: {len(clipping_tasks)} slice(s) using "
                f"{worker_count_used} worker processes"
            )
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count_used,
                mp_context=spawn_context,
            ) as executor:
                for slice_index, clipped_area, raw_clipped_density in executor.map(
                    _clipped_hull_slice_worker, clipping_tasks, chunksize=chunksize
                ):
                    clipping_results[slice_index] = (clipped_area, raw_clipped_density)

    for slice_index, density in fixed_records:
        clip_result = clipping_results.get(slice_index)
        if clip_result is None:
            density_records.append((slice_index, density, math.nan))
            continue
        _clipped_area, raw_clipped_density = clip_result
        if raw_clipped_density > 1.0:
            clipped_hull_slices_capped += 1
        clipped_hull_density = min(raw_clipped_density, 1.0)
        clipped_hull_densities.append(clipped_hull_density)
        density_records.append((slice_index, density, clipped_hull_density))

    mean_density = float(np.mean(densities)) if densities else None
    fixed_density_variance = float(np.var(densities, ddof=0)) if densities else None
    mean_clipped_hull_density = (
        float(np.mean(clipped_hull_densities)) if clipped_hull_densities else None
    )
    clipped_hull_density_variance = (
        float(np.var(clipped_hull_densities, ddof=0)) if clipped_hull_densities else None
    )

    def attach_clipping(geometry: dict[str, object]) -> None:
        slice_index = int(geometry["slice_index"])
        clip_result = clipping_results.get(slice_index)
        if clip_result is None and float(geometry["center_hull_area_mm2"]) > 0.0:
            _, clipped_area, raw_clipped_density = _clipped_hull_slice_worker((
                slice_index,
                np.asarray(geometry["section_centers"], dtype=np.float64),
                np.asarray(geometry["cross_radii"], dtype=np.float64),
                np.asarray(geometry["center_hull_points"], dtype=np.float64),
                float(geometry["center_hull_area_mm2"]),
            ))
            clip_result = (clipped_area, raw_clipped_density)
        if clip_result is not None:
            clipped_area, raw_clipped_density = clip_result
            geometry["clipped_powder_area_mm2"] = clipped_area
            geometry["clipped_hull_density"] = min(raw_clipped_density, 1.0)

    representative: dict[str, object] | None = None
    representative_qualified = False
    if density_records:
        paired_records = [record for record in density_records if math.isfinite(record[2])]
        if paired_records and mean_clipped_hull_density is not None:
            representative_index, _, _ = min(
                paired_records,
                key=lambda item: (
                    abs(item[1] - float(mean_density))
                    + abs(item[2] - mean_clipped_hull_density)
                ),
            )
        else:
            representative_index, _, _ = min(
                density_records, key=lambda item: abs(item[1] - float(mean_density))
            )
        representative = slice_geometry(representative_index)
        if representative is not None:
            attach_clipping(representative)
        representative_qualified = True
    elif maximum_height_slice_index is not None:
        representative = slice_geometry(maximum_height_slice_index)
        if representative is not None:
            attach_clipping(representative)

    if diagnostic_path is not None and representative is not None:
        save_density_slice_image(
            diagnostic_path,
            representative,
            spread_axis,
            source_label,
            maximum_diameter_mm,
            representative_qualified,
        )

    result: dict[str, object] = {
        "mean_fixed_hull_relative_density": mean_density,
        "variance_fixed_hull_relative_density": fixed_density_variance,
        "mean_clipped_hull_relative_density": mean_clipped_hull_density,
        "variance_clipped_hull_relative_density": clipped_hull_density_variance,
        "mean_convex_hull_relative_density": mean_density,
        "variance_convex_hull_relative_density": fixed_density_variance,
        "density_slices_included": len(densities),
        "fixed_hull_density_slices_included": len(densities),
        "clipped_hull_density_slices_included": len(clipped_hull_densities),
        "clipped_hull_degenerate_slices": center_hull_degenerate_slices,
        "density_slices_intersecting_powder": intersected_slices,
        "density_slices_total": number_of_slices,
        "density_slices_in_spreading_roi": int(np.count_nonzero(roi_slice_mask)),
        "density_slices_clipped_to_one": clipped_slices,
        "clipped_hull_density_slices_clipped_to_one": clipped_hull_slices_capped,
        "density_worker_processes": worker_count_used,
        "density_clipping_tasks": len(clipping_tasks),
        "maximum_particle_diameter_mm": maximum_diameter_mm,
        "minimum_density_hull_height_mm": required_height_mm,
    }
    if representative is not None:
        representative_clipped_density = float(representative["clipped_hull_density"])
        result.update({
            "representative_density_slice_pixel": int(representative["slice_index"]),
            "representative_density_slice_position_mm": float(representative["position_mm"]),
            "representative_density_slice_value": float(representative["density"]),
            "representative_fixed_hull_density": float(representative["density"]),
            "representative_clipped_hull_density": (
                representative_clipped_density
                if math.isfinite(representative_clipped_density) else None
            ),
            "representative_density_slice_particle_count": int(len(representative["cross_radii"])),
            "density_slice_image": diagnostic_path.name if diagnostic_path is not None else None,
        })
    return result

def save_density_slice_image(
    path: Path,
    geometry: dict[str, object],
    spread_axis: str,
    source_label: str,
    maximum_diameter_mm: float,
    qualified: bool,
) -> None:
    """Render fixed particle-hull and clipped center-hull methods side by side."""
    try:
        from PIL import Image, ImageChops, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Density-slice images require Pillow") from exc

    width, height = 1900, 980
    left, right, top, bottom, gap = 90, 40, 165, 125, 70
    panel_width = (width - left - right - gap) / 2.0
    plot_height = height - top - bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    fallback_font = False
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 30)
        text_font = ImageFont.truetype("DejaVuSans.ttf", 21)
    except OSError:
        fallback_font = True
        title_font = text_font = ImageFont.load_default()

    def display_text(value: object) -> str:
        """Make labels safe for Pillow's Latin-1 fallback bitmap font."""
        value_text = str(value)
        if fallback_font:
            return value_text.encode("latin-1", errors="replace").decode("latin-1")
        return value_text

    section_centers = np.asarray(geometry["section_centers"])
    cross_radii = np.asarray(geometry["cross_radii"])
    hull_points = np.asarray(geometry["hull_points"])
    center_hull_points = np.asarray(geometry["center_hull_points"])
    transverse_min = float(np.min(section_centers[:, 0] - cross_radii))
    transverse_max = float(np.max(section_centers[:, 0] + cross_radii))
    z_min = float(np.min(section_centers[:, 1] - cross_radii))
    z_max = float(np.max(section_centers[:, 1] + cross_radii))
    transverse_span = max(transverse_max - transverse_min, 1e-12)
    z_span = max(z_max - z_min, 1e-12)
    margin = 0.06 * max(transverse_span, z_span)
    transverse_min -= margin
    transverse_max += margin
    z_min -= margin
    z_max += margin

    scale = min(panel_width / (transverse_max - transverse_min),
                plot_height / (z_max - z_min))
    used_width = scale * (transverse_max - transverse_min)
    used_height = scale * (z_max - z_min)

    axis_name = "y" if spread_axis == "x" else "x"
    slice_name = "x" if spread_axis == "x" else "y"
    title = f"{source_label} - representative transverse density render check"
    draw.text((left, 22), display_text(title), fill=(20, 20, 20), font=title_font)
    qualification = "included" if qualified else "excluded: no slice passed stacking rule"
    details = (
        f"{slice_name} = {float(geometry['position_mm']):.5g} mm   |   "
        f"fixed = {float(geometry['density']):.5f}   |   "
        f"clipped = {float(geometry['clipped_hull_density']):.5f}   |   "
        f"sections = {len(cross_radii)}   |   {qualification}"
    )
    draw.text((left, 68), display_text(details), fill=(35, 35, 35), font=text_font)

    def panel_geometry(panel_left: float):
        x_offset = panel_left + 0.5 * (panel_width - used_width)
        y_offset = top + 0.5 * (plot_height - used_height)

        def transform(point: np.ndarray) -> tuple[float, float]:
            return (
                x_offset + (float(point[0]) - transverse_min) * scale,
                y_offset + used_height - (float(point[1]) - z_min) * scale,
            )

        return x_offset, y_offset, transform

    def draw_axes(panel_left: float, x_offset: float, y_offset: float, transform) -> None:
        plot_left = x_offset
        plot_right = x_offset + used_width
        plot_top = y_offset
        plot_bottom = y_offset + used_height
        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom),
                       outline=(55, 55, 55), width=2)
        if z_min <= 0.0 <= z_max:
            _, substrate_y = transform(np.array([transverse_min, 0.0]))
            draw.line((plot_left, substrate_y, plot_right, substrate_y),
                      fill=(70, 70, 70), width=2)
        draw.text((panel_left + panel_width / 2.0 - 45, height - 80),
                  display_text(f"{axis_name} [mm]"), fill=(20, 20, 20), font=text_font)
        draw.text((panel_left - 58, top + plot_height / 2.0),
                  display_text("z [mm]"), fill=(20, 20, 20), font=text_font)

    fixed_left = float(left)
    fixed_x, fixed_y, fixed_transform = panel_geometry(fixed_left)
    fixed_hull_screen = [fixed_transform(point) for point in hull_points]
    draw.polygon(fixed_hull_screen, fill=(255, 239, 235))
    for center, radius in zip(section_centers, cross_radii):
        cx, cy = fixed_transform(center)
        radius_pixels = float(radius) * scale
        draw.ellipse((cx - radius_pixels, cy - radius_pixels,
                      cx + radius_pixels, cy + radius_pixels),
                     fill=(74, 134, 184), outline=(28, 73, 110), width=2)
    draw.line(fixed_hull_screen + [fixed_hull_screen[0]],
              fill=(211, 45, 38), width=5, joint="curve")
    draw_axes(fixed_left, fixed_x, fixed_y, fixed_transform)
    draw.text((fixed_left, top - 40),
              display_text(f"Fixed particle hull: {float(geometry['density']):.5f}"),
              fill=(150, 25, 25), font=text_font)

    clipped_left = float(left + panel_width + gap)
    clipped_x, clipped_y, clipped_transform = panel_geometry(clipped_left)
    center_hull_screen = [clipped_transform(point) for point in center_hull_points]
    # Faint full circles show exactly which portions are excluded by clipping.
    for center, radius in zip(section_centers, cross_radii):
        cx, cy = clipped_transform(center)
        radius_pixels = float(radius) * scale
        draw.ellipse((cx - radius_pixels, cy - radius_pixels,
                      cx + radius_pixels, cy + radius_pixels),
                     outline=(185, 185, 185), width=2)
    if len(center_hull_screen) >= 3:
        draw.polygon(center_hull_screen, fill=(238, 249, 239))
        particle_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        particle_draw = ImageDraw.Draw(particle_layer)
        for center, radius in zip(section_centers, cross_radii):
            cx, cy = clipped_transform(center)
            radius_pixels = float(radius) * scale
            particle_draw.ellipse((cx - radius_pixels, cy - radius_pixels,
                                   cx + radius_pixels, cy + radius_pixels),
                                  fill=(74, 134, 184, 255),
                                  outline=(28, 73, 110, 255), width=2)
        hull_mask = Image.new("L", canvas.size, 0)
        ImageDraw.Draw(hull_mask).polygon(center_hull_screen, fill=255)
        particle_layer.putalpha(ImageChops.multiply(
            particle_layer.getchannel("A"), hull_mask
        ))
        canvas.paste(particle_layer, (0, 0), particle_layer)
        draw = ImageDraw.Draw(canvas)
        draw.line(center_hull_screen + [center_hull_screen[0]],
                  fill=(34, 139, 79), width=5, joint="curve")
        for center in section_centers:
            cx, cy = clipped_transform(center)
            draw.ellipse((cx - 2.5, cy - 2.5, cx + 2.5, cy + 2.5), fill=(15, 70, 35))
    draw_axes(clipped_left, clipped_x, clipped_y, clipped_transform)
    draw.text((clipped_left, top - 40),
              display_text(
                  f"Clipped center hull: {float(geometry['clipped_hull_density']):.5f}"
              ), fill=(25, 110, 60), font=text_font)

    legend = (
        f"Red = particle-outline hull; green = projected-center hull; gray = clipped-off area. "
        f"Hull height = {float(geometry['hull_height_mm']):.5g} mm; "
        f"max diameter = {maximum_diameter_mm:.5g} mm"
    )
    draw.text((left, height - 42), display_text(legend),
              fill=(70, 70, 70), font=text_font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    print(f"Saved representative density slice {path}")


def calculate_heightmap_layout(
    centers: np.ndarray,
    radii: np.ndarray,
    transverse_pixels: int,
    spread_axis: str,
    fixed_transverse_bounds_m: tuple[float, float] | None = None,
) -> tuple[int, int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a square-pixel raster from a fixed transverse resolution.

    The transverse physical interval is assigned ``transverse_pixels`` pixels.
    The spreading interval is detected independently for each case and its
    pixel count is calculated from that same pixel size. In batch mode,
    ``fixed_transverse_bounds_m`` is shared by every file.
    """
    if transverse_pixels <= 0:
        raise ValueError("transverse_pixels must be positive")
    if spread_axis not in ("x", "y"):
        raise ValueError("spread_axis must be 'x' or 'y'")

    particle_lower = np.min(centers - radii[:, None], axis=0).astype(np.float64)
    particle_upper = np.max(centers + radii[:, None], axis=0).astype(np.float64)
    spread_index = 0 if spread_axis == "x" else 1
    transverse_index = 1 - spread_index

    if fixed_transverse_bounds_m is None:
        transverse_lower = float(particle_lower[transverse_index])
        transverse_upper = float(particle_upper[transverse_index])
    else:
        transverse_lower, transverse_upper = map(float, fixed_transverse_bounds_m)
        if (
            transverse_lower > float(particle_lower[transverse_index]) + 1e-12
            or transverse_upper < float(particle_upper[transverse_index]) - 1e-12
        ):
            raise ValueError("fixed transverse bounds do not contain all retained particles")

    transverse_span = transverse_upper - transverse_lower
    if not math.isfinite(transverse_span) or transverse_span <= 0.0:
        raise ValueError("transverse bounds must have a positive finite span")
    pixel_size_m = transverse_span / transverse_pixels

    spread_span = float(particle_upper[spread_index] - particle_lower[spread_index])
    spread_pixels = max(1, int(math.ceil(spread_span / pixel_size_m - 1e-12)))
    spread_center = 0.5 * (
        float(particle_lower[spread_index]) + float(particle_upper[spread_index])
    )
    spread_half_span = 0.5 * pixel_size_m * spread_pixels

    raster_lower = particle_lower.copy()
    raster_upper = particle_upper.copy()
    raster_lower[transverse_index] = transverse_lower
    raster_upper[transverse_index] = transverse_upper
    raster_lower[spread_index] = spread_center - spread_half_span
    raster_upper[spread_index] = spread_center + spread_half_span

    if spread_axis == "x":
        width, height = spread_pixels, transverse_pixels
    else:
        width, height = transverse_pixels, spread_pixels
    return width, height, pixel_size_m, particle_lower, particle_upper, raster_lower, raster_upper


def generate_heightmap(ctx, centers: np.ndarray, diameters: np.ndarray,
                       output: Path, transverse_pixels: int, spread_axis: str,
                       fixed_transverse_bounds_m: tuple[float, float] | None = None,
                       ) -> tuple[np.ndarray, dict[str, object], Path]:
    """Rasterize the uppermost sphere surface into a float32 heightmap on the GPU."""
    radii = 0.5 * diameters
    spheres = np.ascontiguousarray(np.column_stack((centers, radii)), dtype=np.float32)
    (width, height, pixel_size_m, particle_lower, particle_upper,
     raster_lower, raster_upper) = calculate_heightmap_layout(
        centers, radii, transverse_pixels, spread_axis, fixed_transverse_bounds_m
    )
    print(
        f"Detected heightmap bounds; output {width} x {height} pixels at "
        f"{pixel_size_m * 1000.0:.9g} mm/pixel"
    )
    center = 0.5 * (raster_lower + raster_upper)
    z_span = max(
        float(particle_upper[2] - particle_lower[2]), float(np.max(radii)), 1e-9
    )
    margin = max(0.01 * z_span, 1e-9)
    eye_z = float(particle_upper[2] + margin)
    near = margin
    far = float(eye_z - particle_lower[2] + margin)

    eye = np.array([center[0], center[1], eye_z], dtype=np.float32)
    target = np.array([center[0], center[1], center[2]], dtype=np.float32)
    view = look_at(eye, target, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    proj = orthographic(
        float(raster_lower[0] - center[0]), float(raster_upper[0] - center[0]),
        float(raster_lower[1] - center[1]), float(raster_upper[1] - center[1]), near, far,
    )

    program = ctx.program(
        vertex_shader=HEIGHTMAP_VERTEX_SHADER,
        fragment_shader=HEIGHTMAP_FRAGMENT_SHADER,
    )
    sphere_buffer = ctx.buffer(spheres.tobytes())
    vao = ctx.vertex_array(program, [(sphere_buffer, "4f /i", "in_sphere")])
    maximum_texture_size = int(ctx.info.get("GL_MAX_TEXTURE_SIZE", 0))
    if maximum_texture_size and max(width, height) > maximum_texture_size:
        vao.release()
        sphere_buffer.release()
        program.release()
        raise RuntimeError(
            f"Calculated heightmap size {width} x {height} exceeds the GPU texture "
            f"limit of {maximum_texture_size}. Reduce --heightmap-transverse-pixels."
        )
    color_texture = ctx.texture((width, height), 1, dtype="f1")
    depth_texture = ctx.depth_texture((width, height))
    framebuffer = ctx.framebuffer(
        color_attachments=[color_texture], depth_attachment=depth_texture
    )

    old_viewport = ctx.viewport
    framebuffer.use()
    ctx.viewport = (0, 0, width, height)
    framebuffer.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
    program["u_view"].write(view.T.astype("f4").tobytes())
    program["u_proj"].write(proj.T.astype("f4").tobytes())
    vao.render(vertices=6, instances=len(centers))

    depth = np.frombuffer(depth_texture.read(alignment=1), dtype=np.float32).reshape(height, width)
    # OpenGL returns bottom row first; image/array convention here is row 0 = y_max.
    depth = np.flipud(depth).copy()
    occupied = depth < (1.0 - 1e-7)
    heightmap_m = np.full((height, width), np.nan, dtype=np.float32)
    heightmap_m[occupied] = eye_z - near - depth[occupied] * (far - near)
    heightmap_mm = heightmap_m * np.float32(1000.0)

    npy_path, png_path, json_path = _heightmap_paths(output)
    for path in (npy_path, png_path, json_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, heightmap_mm)

    finite = np.isfinite(heightmap_mm)
    if np.any(finite):
        lo = float(np.nanmin(heightmap_mm))
        hi = float(np.nanmax(heightmap_mm))
        preview = np.zeros_like(heightmap_mm, dtype=np.uint16)
        preview[finite] = np.round(
            np.clip((heightmap_mm[finite] - lo) / max(hi - lo, 1e-20), 0.0, 1.0) * 65535.0
        ).astype(np.uint16)
    else:
        lo = hi = float("nan")
        preview = np.zeros_like(heightmap_mm, dtype=np.uint16)

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Heightmap PNG output requires Pillow: python -m pip install Pillow") from exc
    Image.fromarray(preview, mode="I;16").save(png_path)
    metadata = {
        "data_file": npy_path.name,
        "preview_file": png_path.name,
        "shape": [height, width],
        "source_coordinate_units": "m",
        "height_units": "mm",
        "value": "absolute top-surface z coordinate",
        "empty_pixels": "NaN in NPY; 0 in PNG preview",
        "array_orientation": "row 0 is y_max; column 0 is x_min",
        "pixel_size_mm": pixel_size_m * 1000.0,
        "pixel_area_mm2": (pixel_size_m * 1000.0) ** 2,
        "spreading_axis": spread_axis,
        "transverse_axis": "y" if spread_axis == "x" else "x",
        "transverse_pixels": transverse_pixels,
        "pixel_center_coordinates": {
            "x_mm": "raster_bounds_mm.x[0] + (column + 0.5) * pixel_size_mm",
            "y_mm": "raster_bounds_mm.y[1] - (row + 0.5) * pixel_size_mm",
        },
        "particle_bounds_mm": {
            "x": [float(particle_lower[0] * 1000.0), float(particle_upper[0] * 1000.0)],
            "y": [float(particle_lower[1] * 1000.0), float(particle_upper[1] * 1000.0)],
            "z": [float(particle_lower[2] * 1000.0), float(particle_upper[2] * 1000.0)],
        },
        "raster_bounds_mm": {
            "x": [float(raster_lower[0] * 1000.0), float(raster_upper[0] * 1000.0)],
            "y": [float(raster_lower[1] * 1000.0), float(raster_upper[1] * 1000.0)],
        },
        "finite_height_range_mm": [lo, hi],
    }
    print(f"Saved exact heightmap {npy_path}")
    print(f"Saved heightmap preview {png_path}")

    if ctx.screen is not None:
        ctx.screen.use()
    ctx.viewport = old_viewport
    framebuffer.release()
    depth_texture.release()
    color_texture.release()
    vao.release()
    sphere_buffer.release()
    program.release()
    return heightmap_mm, metadata, json_path


def finalize_heightmap_analysis(
    heightmap_mm: np.ndarray,
    centers: np.ndarray,
    diameters: np.ndarray,
    spreading_roi: dict[str, object],
    metadata: dict[str, object],
    metadata_path: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    """Extract features, add them to metadata, and write the JSON sidecar."""
    # Remove a legacy histogram diagnostic when rerunning into an existing output folder.
    legacy_histogram_path = metadata_path.with_name(
        f"{metadata_path.stem}_spreading_histogram.png"
    )
    if legacy_histogram_path.exists():
        legacy_histogram_path.unlink()

    features = extract_heightmap_features(
        heightmap_mm=heightmap_mm,
        spread_axis=args.spread_axis,
        transverse_occupancy=args.transverse_occupancy,
        active_fraction=args.active_fraction,
        substrate_height_mm=args.substrate_height_mm,
        metadata=metadata,
        spreading_roi=spreading_roi,
    )
    features.update(extract_convex_hull_density(
        centers=centers,
        diameters=diameters,
        metadata=metadata,
        spread_axis=args.spread_axis,
        stack_height_factor=args.density_stack_height_factor,
        hull_directions=args.density_hull_directions,
        density_workers=args.density_workers,
        spreading_roi=spreading_roi,
        diagnostic_path=metadata_path.with_name(
            f"{metadata_path.stem}_density_slice.png"
        ),
        source_label=metadata_path.stem,
    ))
    features.update(extract_coordination_features(
        centers=centers,
        diameters=diameters,
        tolerance_mm=args.coordination_tolerance_mm,
        spreading_roi=spreading_roi,
        spread_axis=args.spread_axis,
    ))
    features.update(spreading_roi)
    heightmap_shape = metadata["shape"]
    spreading_pixels = (
        int(heightmap_shape[1]) if args.spread_axis == "x" else int(heightmap_shape[0])
    )
    raster_bounds = metadata["raster_bounds_mm"]
    spreading_bounds = raster_bounds[args.spread_axis]
    transverse_axis = "y" if args.spread_axis == "x" else "x"
    transverse_bounds = raster_bounds[transverse_axis]
    features.update({
        "heightmap_pixel_size_mm": float(metadata["pixel_size_mm"]),
        "heightmap_transverse_pixels": int(metadata["transverse_pixels"]),
        "heightmap_spreading_pixels": spreading_pixels,
        "heightmap_transverse_span_mm": float(transverse_bounds[1] - transverse_bounds[0]),
        "heightmap_spreading_span_mm": float(spreading_bounds[1] - spreading_bounds[0]),
    })
    metadata["feature_extraction"] = {
        "spread_axis": args.spread_axis,
        "transverse_occupancy_threshold": args.transverse_occupancy,
        "active_fraction_threshold": args.active_fraction,
        "substrate_height_mm": args.substrate_height_mm,
        "fixed_hull_density_definition": "complete transverse circle areas divided by the convex hull area of the circle outlines",
        "clipped_hull_density_definition": "transverse circle areas clipped to the projected-center convex hull, divided by that center-hull area",
        "density_inclusion_rule": "convex hull height greater than maximum particle diameter times density_stack_height_factor",
        "density_stack_height_factor": args.density_stack_height_factor,
        "density_hull_support_directions": args.density_hull_directions,
        "coordination_definition": "count of neighbours whose sphere surface gap is no greater than coordination_tolerance_mm, including DEM overlap; statistics include only particles inside the accepted central spreading ROI",
        "spreading_roi_definition": "longest contiguous nominal particle-count region after coordinate-outlier removal; beginning and overfilled end regions are excluded from roughness, density, and coordination statistics",
        **features,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved heightmap metadata {metadata_path}")
    print(
        "Mean transverse RMS roughness: "
        f"{features['mean_transverse_rms_roughness_mm']:.9g} mm"
    )
    fixed_density = features["mean_fixed_hull_relative_density"]
    if fixed_density is None:
        print("Mean fixed-hull relative density: unavailable (no stacked slices)")
    else:
        print(f"Mean fixed-hull relative density: {fixed_density:.9g}")
    clipped_density = features["mean_clipped_hull_relative_density"]
    if clipped_density is None:
        print("Mean clipped-hull relative density: unavailable (no valid center hulls)")
    else:
        print(f"Mean clipped-hull relative density: {clipped_density:.9g}")
    print(f"Mean coordination number: {features['mean_coordination_number']:.9g}")
    print(
        "Roughness variance across included spreading slices: "
        f"{features['transverse_rms_roughness_variance_mm2']:.9g} mm^2"
    )
    fixed_var = features["variance_fixed_hull_relative_density"]
    if fixed_var is None:
        print("Fixed-hull density variance: unavailable")
    else:
        print(f"Fixed-hull density variance: {fixed_var:.9g}")
    clipped_var = features["variance_clipped_hull_relative_density"]
    if clipped_var is None:
        print("Clipped-hull density variance: unavailable")
    else:
        print(f"Clipped-hull density variance: {clipped_var:.9g}")
    print(
        "Coordination-number variance: "
        f"{features['coordination_number_variance']:.9g}"
    )
    return features


def run(args: argparse.Namespace) -> None:
    try:
        import glfw
        import moderngl
    except ImportError as exc:
        raise RuntimeError(
            "Missing GPU-rendering packages. Install them with:\n"
            "  python -m pip install numpy moderngl glfw Pillow matplotlib"
        ) from exc

    centers, diameters, contacts, particle_filter = load_comsol_grains(
        args.input, args.filter_outliers, args.outlier_mad_threshold
    )
    print(
        f"Particle filter: {particle_filter['rendered_particle_rows']:,} retained; "
        f"{particle_filter['invalid_particle_rows_removed']:,} invalid and "
        f"{particle_filter['coordinate_outlier_rows_removed']:,} coordinate outlier(s) removed"
    )
    radii = 0.5 * diameters * args.diameter_scale
    colors = colorize(diameters if args.color == "diameter" else contacts, args.color, centers)
    spheres = np.ascontiguousarray(np.column_stack((centers, radii)), dtype=np.float32)

    if not glfw.init():
        raise RuntimeError("GLFW could not initialize. A working OpenGL display is required.")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    if args.heightmap_only:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    if args.msaa:
        glfw.window_hint(glfw.SAMPLES, args.msaa)
    window = glfw.create_window(args.width, args.height, args.title, None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create an OpenGL 3.3 window.")

    glfw.make_context_current(window)
    glfw.swap_interval(1 if args.vsync else 0)
    ctx = moderngl.create_context(require=330)
    ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
    program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
    sphere_buffer = ctx.buffer(spheres.tobytes())
    color_buffer = ctx.buffer(colors.tobytes())
    vao = ctx.vertex_array(
        program,
        [
            (sphere_buffer, "4f /i", "in_sphere"),
            (color_buffer, "3f /i", "in_color"),
        ],
    )
    if args.heightmap is not None:
        heightmap_mm, metadata, metadata_path = generate_heightmap(
            ctx, centers, diameters, args.heightmap,
            args.heightmap_transverse_pixels, args.spread_axis,
        )
        metadata["particle_filter"] = particle_filter
        spreading_roi = determine_spreading_roi_from_heightmap(
            heightmap_mm=heightmap_mm,
            metadata=metadata,
            spread_axis=args.spread_axis,
            substrate_height_mm=args.substrate_height_mm,
            filter_width_mm=args.spreading_roi_filter_width_mm,
            diagnostic_path=metadata_path.with_name(
                f"{metadata_path.stem}_spreading_roi_height.png"
            ),
            source_label=metadata_path.stem,
        )
        print(
            f"Spreading ROI: {spreading_roi['spreading_roi_start_mm']:.6g} to "
            f"{spreading_roi['spreading_roi_end_mm']:.6g} mm "
            f"({spreading_roi['spreading_roi_pixels_included']}/"
            f"{spreading_roi['spreading_roi_pixels_total']} heightmap pixels)"
        )
        finalize_heightmap_analysis(
            heightmap_mm, centers, diameters, spreading_roi,
            metadata, metadata_path, args
        )
    if args.heightmap_only:
        vao.release()
        sphere_buffer.release()
        color_buffer.release()
        program.release()
        glfw.destroy_window(window)
        glfw.terminate()
        return
    camera = Camera(centers, radii)
    state = {"button": None, "last": (0.0, 0.0), "save": args.screenshot, "frames": 0}

    def mouse_button(_window, button, action, _mods):
        if action == glfw.PRESS and button in (glfw.MOUSE_BUTTON_LEFT, glfw.MOUSE_BUTTON_RIGHT):
            state["button"] = button
            state["last"] = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            state["button"] = None

    def cursor(_window, x, y):
        lx, ly = state["last"]
        if state["button"] == glfw.MOUSE_BUTTON_LEFT:
            camera.orbit(x - lx, y - ly)
        elif state["button"] == glfw.MOUSE_BUTTON_RIGHT:
            _, h = glfw.get_framebuffer_size(window)
            camera.pan(x - lx, y - ly, h)
        state["last"] = (x, y)

    def scroll(_window, _xoffset, yoffset):
        camera.zoom(yoffset)

    def key(_window, key_code, _scancode, action, _mods):
        if action != glfw.PRESS:
            return
        if key_code == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key_code == glfw.KEY_R:
            camera.reset()
        elif key_code == glfw.KEY_S:
            state["save"] = args.screenshot or Path("powder_render.png")

    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_cursor_pos_callback(window, cursor)
    glfw.set_scroll_callback(window, scroll)
    glfw.set_key_callback(window, key)

    print(f"Loaded {len(centers):,} grains; GPU: {ctx.info.get('GL_RENDERER', 'unknown')}")
    try:
        while not glfw.window_should_close(window):
            glfw.poll_events()
            width, height = glfw.get_framebuffer_size(window)
            if width == 0 or height == 0:
                continue
            ctx.viewport = (0, 0, width, height)
            bg = np.array(args.background, dtype=np.float32)
            ctx.clear(float(bg[0]), float(bg[1]), float(bg[2]), 1.0, depth=1.0)

            near = max(camera.distance - 2.5 * camera.scene_radius, camera.scene_radius * 1e-4)
            far = camera.distance + 2.5 * camera.scene_radius
            view = camera.view()
            proj = perspective(math.radians(45.0), width / height, near, far)
            program["u_view"].write(view.T.astype("f4").tobytes())
            program["u_proj"].write(proj.T.astype("f4").tobytes())
            program["u_light_view"].value = (-camera.scene_radius, camera.scene_radius, camera.scene_radius)
            vao.render(mode=moderngl.TRIANGLES, vertices=6, instances=len(centers))

            if state["save"] is not None and state["frames"] >= 1:
                save_framebuffer(ctx, width, height, Path(state["save"]))
                state["save"] = None
                if args.screenshot and args.exit_after_screenshot:
                    glfw.set_window_should_close(window, True)
            glfw.swap_buffers(window)
            state["frames"] += 1
    finally:
        vao.release()
        sphere_buffer.release()
        color_buffer.release()
        program.release()
        glfw.destroy_window(window)
        glfw.terminate()



EXPERIMENT_FACTOR_LEVELS = {
    1: {
        "PS": "50% R0.25mm & 50% R0.5mm",
        "PS_R0.25_percent": 50,
        "PS_R0.5_percent": 50,
        "RS_mm_s": 100,
        "RD_mm": 50,
    },
    2: {
        "PS": "25% R0.25mm & 75% R0.5mm",
        "PS_R0.25_percent": 25,
        "PS_R0.5_percent": 75,
        "RS_mm_s": 300,
        "RD_mm": 100,
    },
    3: {
        "PS": "75% R0.25mm & 25% R0.5mm",
        "PS_R0.25_percent": 75,
        "PS_R0.5_percent": 25,
        "RS_mm_s": 500,
        "RD_mm": 150,
    },
}


def extract_experiment_factors(path: Path) -> dict[str, object]:
    """Decode trailing PS_RS_RD level indices from a result filename.

    Examples
    --------
    ``solution 2_3_1.txt`` -> PS level 2, RS level 3, RD level 1.
    ``results 1-2-3.txt``   -> PS level 1, RS level 2, RD level 3.

    The three factors are decoded independently; the level-to-value mapping is
    the DOE table supplied for this study.
    """
    match = re.search(r"([123])[\s_-]+([123])[\s_-]+([123])$", path.stem)
    if match is None:
        return {
            "PS_level": None,
            "PS": None,
            "PS_R0.25_percent": None,
            "PS_R0.5_percent": None,
            "RS_level": None,
            "RS_mm_s": None,
            "RD_level": None,
            "RD_mm": None,
        }

    ps_level, rs_level, rd_level = (int(value) for value in match.groups())
    ps_values = EXPERIMENT_FACTOR_LEVELS[ps_level]
    rs_values = EXPERIMENT_FACTOR_LEVELS[rs_level]
    rd_values = EXPERIMENT_FACTOR_LEVELS[rd_level]
    return {
        "PS_level": ps_level,
        "PS": ps_values["PS"],
        "PS_R0.25_percent": ps_values["PS_R0.25_percent"],
        "PS_R0.5_percent": ps_values["PS_R0.5_percent"],
        "RS_level": rs_level,
        "RS_mm_s": rs_values["RS_mm_s"],
        "RD_level": rd_level,
        "RD_mm": rd_values["RD_mm"],
    }

def _write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write accumulated batch results; new feature keys become new columns."""
    result_fields = [
        "file",
        "PS_level",
        "PS",
        "PS_R0.25_percent",
        "PS_R0.5_percent",
        "RS_level",
        "RS_mm_s",
        "RD_level",
        "RD_mm",
        "mean_transverse_rms_roughness_mm",
        "transverse_rms_roughness_variance_mm2",
        "mean_fixed_hull_relative_density",
        "variance_fixed_hull_relative_density",
        "mean_clipped_hull_relative_density",
        "variance_clipped_hull_relative_density",
        "mean_convex_hull_relative_density",
        "mean_coordination_number",
        "coordination_number_variance",
        "status",
        "error",
    ]
    diagnostic_fields = sorted(
        {key for row in rows for key in row} - set(result_fields)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_fields + diagnostic_fields)
        writer.writeheader()
        writer.writerows(rows)


def run_batch(args: argparse.Namespace) -> Path:
    """Render and analyze every matching text file using one GPU context."""
    try:
        import glfw
        import moderngl
    except ImportError as exc:
        raise RuntimeError(
            "Missing GPU-rendering packages. Install them with:\n"
            "  python -m pip install numpy moderngl glfw Pillow matplotlib"
        ) from exc

    iterator = args.input.rglob(args.pattern) if args.recursive else args.input.glob(args.pattern)
    inputs = sorted((path for path in iterator if path.is_file()), key=lambda p: str(p).lower())
    if not inputs:
        raise RuntimeError(f"No files matching {args.pattern!r} were found in {args.input}.")

    output_dir = args.output_dir or args.input / "powder_analysis_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "powder_features.csv"

    # Establish one shared transverse physical interval after outlier removal.
    # Its fixed pixel count defines one exact mm/pixel value for the full batch.
    transverse_index = 1 if args.spread_axis == "x" else 0
    transverse_lower = math.inf
    transverse_upper = -math.inf
    transverse_bounds_files = 0
    for input_path in inputs:
        try:
            scan_centers, scan_diameters, _scan_contacts, _scan_filter = load_comsol_grains(
                input_path, args.filter_outliers, args.outlier_mad_threshold
            )
        except Exception:
            continue
        scan_radii = 0.5 * scan_diameters
        transverse_lower = min(
            transverse_lower,
            float(np.min(scan_centers[:, transverse_index] - scan_radii)),
        )
        transverse_upper = max(
            transverse_upper,
            float(np.max(scan_centers[:, transverse_index] + scan_radii)),
        )
        transverse_bounds_files += 1
    if transverse_bounds_files == 0:
        raise RuntimeError("No valid files were available to determine transverse bounds.")
    fixed_transverse_bounds_m = (transverse_lower, transverse_upper)
    shared_pixel_size_mm = (
        (transverse_upper - transverse_lower) * 1000.0
        / args.heightmap_transverse_pixels
    )
    print(
        f"Shared transverse bounds: {transverse_lower * 1000.0:.9g} to "
        f"{transverse_upper * 1000.0:.9g} mm; "
        f"{args.heightmap_transverse_pixels} pixels at "
        f"{shared_pixel_size_mm:.9g} mm/pixel"
    )

    if not glfw.init():
        raise RuntimeError("GLFW could not initialize. A working OpenGL display is required.")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(64, 64, args.title, None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Could not create a hidden OpenGL 3.3 window.")

    glfw.make_context_current(window)
    ctx = moderngl.create_context(require=330)
    ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
    rows: list[dict[str, object]] = []
    print(f"Found {len(inputs)} input file(s); results directory: {output_dir}")

    try:
        for index, input_path in enumerate(inputs, start=1):
            relative = input_path.relative_to(args.input)
            output_prefix = output_dir / relative.parent / input_path.stem
            print(f"[{index}/{len(inputs)}] Processing {relative}")
            factors = extract_experiment_factors(input_path)
            row: dict[str, object] = {
                "file": str(relative),
                **factors,
                "status": "ok",
                "error": "",
            }
            try:
                centers, diameters, _contacts, particle_filter = load_comsol_grains(
                    input_path, args.filter_outliers, args.outlier_mad_threshold
                )
                row.update(particle_filter)
                heightmap_mm, metadata, metadata_path = generate_heightmap(
                    ctx, centers, diameters, output_prefix,
                    args.heightmap_transverse_pixels, args.spread_axis,
                    fixed_transverse_bounds_m,
                )
                metadata["particle_filter"] = particle_filter
                metadata["experiment_factors"] = factors
                spreading_roi = determine_spreading_roi_from_heightmap(
                    heightmap_mm=heightmap_mm,
                    metadata=metadata,
                    spread_axis=args.spread_axis,
                    substrate_height_mm=args.substrate_height_mm,
                    filter_width_mm=args.spreading_roi_filter_width_mm,
                    diagnostic_path=metadata_path.with_name(
                        f"{metadata_path.stem}_spreading_roi_height.png"
                    ),
                    source_label=metadata_path.stem,
                )
                row.update(spreading_roi)
                print(
                    f"Spreading ROI: {spreading_roi['spreading_roi_start_mm']:.6g} to "
                    f"{spreading_roi['spreading_roi_end_mm']:.6g} mm "
                    f"({spreading_roi['spreading_roi_pixels_included']}/"
                    f"{spreading_roi['spreading_roi_pixels_total']} heightmap pixels)"
                )
                row.update(finalize_heightmap_analysis(
                    heightmap_mm, centers, diameters, spreading_roi,
                    metadata, metadata_path, args,
                ))
            except Exception as exc:
                row["status"] = "error"
                row["error"] = str(exc)
                print(f"error processing {relative}: {exc}", file=sys.stderr)
                if args.stop_on_error:
                    rows.append(row)
                    _write_summary_csv(summary_path, rows)
                    raise RuntimeError(f"Batch stopped while processing {relative}: {exc}") from exc
            rows.append(row)
            # Preserve completed results even if a later input fails.
            _write_summary_csv(summary_path, rows)
    finally:
        glfw.destroy_window(window)
        glfw.terminate()

    succeeded = sum(row["status"] == "ok" for row in rows)
    print(f"Completed {succeeded}/{len(rows)} files; summary: {summary_path}")
    return summary_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="COMSOL text export or folder of text exports")
    parser.add_argument(
        "--output-dir", type=Path,
        help="batch output folder (default: INPUT/powder_analysis_results)",
    )
    parser.add_argument("--pattern", default="*.txt", help="batch filename pattern (default: *.txt)")
    parser.add_argument("--recursive", action="store_true", help="search input subfolders recursively")
    parser.add_argument(
        "--stop-on-error", action="store_true",
        help="stop at the first bad input instead of recording the error and continuing",
    )
    parser.add_argument(
        "--filter-outliers", action=argparse.BooleanOptionalAction, default=True,
        help="discard coordinate outliers before calculating bounds (default: enabled)",
    )
    parser.add_argument(
        "--outlier-mad-threshold", type=float, default=3.0, metavar="N",
        help="robust coordinate-outlier threshold in scaled MADs (default: 3.0)",
    )
    parser.add_argument(
        "--coordination-tolerance-mm", type=float, default=0.02, metavar="MM",
        help="maximum surface gap counted as contact; overlaps count (default: 0.02 mm)",
    )
    parser.add_argument(
        "--spreading-roi-filter-width-mm", type=float, default=10.0, metavar="MM",
        help=("Gaussian low-pass FWHM applied only to the mean powder-height profile "
              "for ROI detection (default: 10 mm)"),
    )
    parser.add_argument("--width", type=int, default=1600, help="window width (default: 1600)")
    parser.add_argument("--height", type=int, default=900, help="window height (default: 900)")
    parser.add_argument("--title", default="GPU Powder Renderer")
    parser.add_argument("--color", choices=("uniform", "diameter", "contacts", "z"), default="diameter")
    parser.add_argument("--diameter-scale", type=float, default=1.0, help="visual diameter multiplier")
    parser.add_argument("--background", type=float, nargs=3, metavar=("R", "G", "B"), default=(0.025, 0.03, 0.045))
    parser.add_argument("--msaa", type=int, choices=(0, 2, 4, 8), default=4, help="multisample antialiasing")
    parser.add_argument("--vsync", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--screenshot", type=Path, help="save the initial view (press S to save again)")
    parser.add_argument("--exit-after-screenshot", action="store_true", help="close after writing --screenshot")
    parser.add_argument(
        "--heightmap", type=Path,
        help="output prefix for exact .npy heights, 16-bit .png preview, and .json metadata",
    )
    parser.add_argument(
        "--heightmap-transverse-pixels", type=int, default=256, metavar="N",
        help=("fixed pixels transverse to spreading; spreading pixels are calculated "
              "from the same physical resolution (default: 256)"),
    )
    parser.add_argument(
        "--heightmap-only", action="store_true",
        help="write the heightmap without opening the interactive viewer",
    )
    parser.add_argument(
        "--spread-axis", choices=("x", "y"), default="x",
        help="source-coordinate spreading direction used for roughness (default: x)",
    )
    parser.add_argument(
        "--transverse-occupancy", type=float, default=0.90,
        help="occupancy threshold for automatic transverse bounds (default: 0.90)",
    )
    parser.add_argument(
        "--active-fraction", type=float, default=0.50,
        help="minimum occupied transverse fraction included in roughness (default: 0.50)",
    )
    parser.add_argument(
        "--substrate-height-mm", type=float, default=0.0,
        help="height assigned to exposed substrate within the bed region (default: 0 mm)",
    )
    parser.add_argument(
        "--density-stack-height-factor", type=float, default=1.0, metavar="F",
        help="require density-hull height > F times maximum particle diameter (default: 1.0)",
    )
    parser.add_argument(
        "--density-hull-directions", type=int, default=128, metavar="N",
        help="support directions used to approximate circular convex hulls (default: 128)",
    )
    parser.add_argument(
        "--density-workers", type=int, default=0, metavar="N",
        help=("processes for clipped-hull density; 0 = auto (up to 16, leaving one CPU "
              "thread free), 1 = serial (default: 0)"),
    )
    args = parser.parse_args(argv)
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.diameter_scale <= 0:
        parser.error("--diameter-scale must be positive")
    if args.heightmap_transverse_pixels <= 0:
        parser.error("--heightmap-transverse-pixels must be positive")
    if args.outlier_mad_threshold <= 0:
        parser.error("--outlier-mad-threshold must be positive")
    if args.coordination_tolerance_mm < 0:
        parser.error("--coordination-tolerance-mm must be nonnegative")
    if args.spreading_roi_filter_width_mm <= 0.0:
        parser.error("--spreading-roi-filter-width-mm must be positive")
    if args.heightmap_only and args.heightmap is None and not args.input.is_dir():
        parser.error("--heightmap-only requires --heightmap OUTPUT")
    if not 0.0 < args.transverse_occupancy <= 1.0:
        parser.error("--transverse-occupancy must be in (0, 1]")
    if not 0.0 < args.active_fraction <= 1.0:
        parser.error("--active-fraction must be in (0, 1]")
    if args.density_stack_height_factor <= 0.0:
        parser.error("--density-stack-height-factor must be positive")
    if args.density_hull_directions < 16:
        parser.error("--density-hull-directions must be at least 16")
    if args.density_workers < 0:
        parser.error("--density-workers must be nonnegative")
    if any(not 0.0 <= x <= 1.0 for x in args.background):
        parser.error("--background components must lie between 0 and 1")
    return args


def main() -> int:
    try:
        args = parse_args()
        if not args.input.exists():
            raise RuntimeError(f"Input does not exist: {args.input}")
        if args.input.is_dir():
            if args.heightmap is not None:
                raise RuntimeError("Use --output-dir, not --heightmap, in directory mode.")
            run_batch(args)
        else:
            if args.output_dir is not None:
                raise RuntimeError("--output-dir is only valid when INPUT is a directory.")
            run(args)
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
