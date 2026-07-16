"""Generate dimensioned AprilTag 36h11 images for documents and printing.

The requested tag edge is the complete outer black-border edge.  SVG files are
the most reliable choice for Word because their physical dimensions are stored
in millimetres.  PNG files use 100 pixels/mm (2540 dpi) metadata so the marker
edge also imports at the requested size when the document does not rescale it.
"""

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


MM_PER_INCH = 25.4
DEFAULT_PIXELS_PER_MM = 100
TAG_CELLS = 8  # 6x6 payload plus the one-cell black border on every side.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create exact-size AprilTag 36h11 SVG and PNG images."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs") / "apriltags_10_62mm",
    )
    parser.add_argument("--tag-edge-mm", type=float, default=10.62)
    parser.add_argument("--ids", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--pixels-per-mm", type=int, default=DEFAULT_PIXELS_PER_MM
    )
    return parser.parse_args()


def marker_image(dictionary: object, tag_id: int, side_pixels: int) -> np.ndarray:
    return cv2.aruco.generateImageMarker(
        dictionary, int(tag_id), int(side_pixels), borderBits=1
    )


def marker_cells(dictionary: object, tag_id: int) -> np.ndarray:
    """Return the exact 8x8 marker, where True means a black cell."""
    image = marker_image(dictionary, tag_id, TAG_CELLS)
    return image < 128


def svg_marker_rectangles(
    cells: np.ndarray, x_mm: float, y_mm: float, tag_edge_mm: float
) -> List[str]:
    cell_mm = tag_edge_mm / float(TAG_CELLS)
    rectangles = []
    for row in range(TAG_CELLS):
        for col in range(TAG_CELLS):
            if cells[row, col]:
                rectangles.append(
                    '<rect x="%.6f" y="%.6f" width="%.6f" height="%.6f"/>'
                    % (
                        x_mm + col * cell_mm,
                        y_mm + row * cell_mm,
                        cell_mm,
                        cell_mm,
                    )
                )
    return rectangles


def write_individual_svg(
    path: Path, cells: np.ndarray, tag_id: int, tag_edge_mm: float
) -> None:
    quiet_mm = 2.0 * tag_edge_mm / float(TAG_CELLS)
    canvas_mm = tag_edge_mm + 2.0 * quiet_mm
    content = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%.6fmm" height="%.6fmm" '
        'viewBox="0 0 %.6f %.6f" shape-rendering="crispEdges">'
        % (canvas_mm, canvas_mm, canvas_mm, canvas_mm),
        "<title>AprilTag 36h11 ID %d, outer black border %.2f mm</title>"
        % (tag_id, tag_edge_mm),
        '<rect width="100%" height="100%" fill="white"/>',
        '<g fill="black">',
    ]
    content.extend(svg_marker_rectangles(cells, quiet_mm, quiet_mm, tag_edge_mm))
    content.extend(["</g>", "</svg>"])
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def write_individual_png(
    path: Path,
    dictionary: object,
    tag_id: int,
    tag_edge_mm: float,
    pixels_per_mm: int,
) -> Tuple[int, int]:
    marker_pixels = int(round(tag_edge_mm * pixels_per_mm))
    quiet_mm = 2.0 * tag_edge_mm / float(TAG_CELLS)
    quiet_pixels = int(round(quiet_mm * pixels_per_mm))
    marker = marker_image(dictionary, tag_id, marker_pixels)
    canvas_side = marker_pixels + 2 * quiet_pixels
    canvas = np.full((canvas_side, canvas_side), 255, dtype=np.uint8)
    canvas[
        quiet_pixels : quiet_pixels + marker_pixels,
        quiet_pixels : quiet_pixels + marker_pixels,
    ] = marker
    dpi = float(pixels_per_mm) * MM_PER_INCH
    Image.fromarray(canvas, mode="L").save(path, dpi=(dpi, dpi))
    return marker_pixels, quiet_pixels


def sheet_positions() -> Sequence[Tuple[int, float, float]]:
    """Tag id index and marker top-left positions in a 64x48 mm sheet."""
    return (
        (0, 6.69, 4.0),
        (1, 26.69, 4.0),
        (2, 46.69, 4.0),
        (3, 16.69, 23.0),
        (4, 36.69, 23.0),
    )


def write_sheet_svg(
    path: Path,
    dictionary: object,
    tag_ids: Sequence[int],
    tag_edge_mm: float,
) -> None:
    width_mm = 64.0
    height_mm = 48.0
    content = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="64mm" height="48mm" '
        'viewBox="0 0 64 48" shape-rendering="crispEdges">',
        "<title>AprilTag 36h11 IDs 1 to 5, outer black borders 10.62 mm</title>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for index, x_mm, y_mm in sheet_positions():
        if index >= len(tag_ids):
            continue
        tag_id = tag_ids[index]
        cells = marker_cells(dictionary, tag_id)
        content.append('<g fill="black">')
        content.extend(svg_marker_rectangles(cells, x_mm, y_mm, tag_edge_mm))
        content.append("</g>")
        content.append(
            '<text x="%.6f" y="%.6f" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="2.2" fill="black">ID %d</text>'
            % (x_mm + tag_edge_mm / 2.0, y_mm + tag_edge_mm + 3.0, tag_id)
        )
    content.extend(
        [
            '<g fill="none" stroke="black" stroke-width="0.25">',
            '<line x1="6" y1="43" x2="26" y2="43"/>',
            '<line x1="6" y1="41.8" x2="6" y2="44.2"/>',
            '<line x1="26" y1="41.8" x2="26" y2="44.2"/>',
            "</g>",
            '<text x="28" y="43.8" font-family="Arial, sans-serif" '
            'font-size="2.1" fill="black">20.00 mm print check</text>',
            '<text x="32" y="47" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="1.8" fill="black">'
            "Every outer black-border edge = %.2f mm; print at 100%%</text>"
            % tag_edge_mm,
            "</svg>",
        ]
    )
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def put_centered_text(
    image: np.ndarray,
    text: str,
    center_x: int,
    y: int,
    font_scale: float,
    thickness: int,
) -> None:
    size, _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    cv2.putText(
        image,
        text,
        (center_x - size[0] // 2, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        0,
        thickness,
        cv2.LINE_AA,
    )


def write_sheet_png(
    path: Path,
    dictionary: object,
    tag_ids: Sequence[int],
    tag_edge_mm: float,
    pixels_per_mm: int,
) -> None:
    width_pixels = int(round(64.0 * pixels_per_mm))
    height_pixels = int(round(48.0 * pixels_per_mm))
    image = np.full((height_pixels, width_pixels), 255, dtype=np.uint8)
    marker_pixels = int(round(tag_edge_mm * pixels_per_mm))
    for index, x_mm, y_mm in sheet_positions():
        if index >= len(tag_ids):
            continue
        tag_id = tag_ids[index]
        x = int(round(x_mm * pixels_per_mm))
        y = int(round(y_mm * pixels_per_mm))
        marker = marker_image(dictionary, tag_id, marker_pixels)
        image[y : y + marker_pixels, x : x + marker_pixels] = marker
        put_centered_text(
            image,
            "ID %d" % tag_id,
            x + marker_pixels // 2,
            y + marker_pixels + int(round(2.6 * pixels_per_mm)),
            0.65 * pixels_per_mm / 10.0,
            max(1, int(round(0.18 * pixels_per_mm))),
        )

    x1 = int(round(6.0 * pixels_per_mm))
    x2 = int(round(26.0 * pixels_per_mm))
    y = int(round(43.0 * pixels_per_mm))
    thickness = max(1, int(round(0.25 * pixels_per_mm)))
    cap = int(round(1.2 * pixels_per_mm))
    cv2.line(image, (x1, y), (x2, y), 0, thickness, cv2.LINE_AA)
    cv2.line(image, (x1, y - cap), (x1, y + cap), 0, thickness, cv2.LINE_AA)
    cv2.line(image, (x2, y - cap), (x2, y + cap), 0, thickness, cv2.LINE_AA)
    cv2.putText(
        image,
        "20.00 mm print check",
        (int(round(28.0 * pixels_per_mm)), int(round(43.8 * pixels_per_mm))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55 * pixels_per_mm / 10.0,
        0,
        max(1, int(round(0.16 * pixels_per_mm))),
        cv2.LINE_AA,
    )
    put_centered_text(
        image,
        "Every outer black-border edge = %.2f mm; print at 100%%" % tag_edge_mm,
        width_pixels // 2,
        int(round(47.0 * pixels_per_mm)),
        0.48 * pixels_per_mm / 10.0,
        max(1, int(round(0.14 * pixels_per_mm))),
    )
    dpi = float(pixels_per_mm) * MM_PER_INCH
    Image.fromarray(image, mode="L").save(path, dpi=(dpi, dpi))


def detected_ids(path: Path, dictionary: object) -> List[int]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    _corners, ids, _rejected = detector.detectMarkers(image)
    if ids is None:
        return []
    return sorted(int(value) for value in ids.reshape(-1))


def main() -> None:
    args = parse_args()
    if args.tag_edge_mm <= 0.0:
        raise SystemExit("--tag-edge-mm must be positive")
    if args.pixels_per_mm <= 0:
        raise SystemExit("--pixels-per-mm must be positive")
    if len(args.ids) > 5:
        raise SystemExit("the combined sheet supports at most five IDs")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker_pixels = int(round(args.tag_edge_mm * args.pixels_per_mm))
    quiet_pixels = 0

    for tag_id in args.ids:
        stem = "apriltag_36h11_id_%d_%.2fmm" % (tag_id, args.tag_edge_mm)
        cells = marker_cells(dictionary, tag_id)
        write_individual_svg(
            output_dir / (stem + ".svg"), cells, tag_id, args.tag_edge_mm
        )
        marker_pixels, quiet_pixels = write_individual_png(
            output_dir / (stem + ".png"),
            dictionary,
            tag_id,
            args.tag_edge_mm,
            args.pixels_per_mm,
        )

    sheet_stem = "apriltag_36h11_ids_%s_%.2fmm_sheet" % (
        "_".join(str(tag_id) for tag_id in args.ids),
        args.tag_edge_mm,
    )
    sheet_svg = output_dir / (sheet_stem + ".svg")
    sheet_png = output_dir / (sheet_stem + ".png")
    write_sheet_svg(sheet_svg, dictionary, args.ids, args.tag_edge_mm)
    write_sheet_png(
        sheet_png,
        dictionary,
        args.ids,
        args.tag_edge_mm,
        args.pixels_per_mm,
    )

    verification = {
        "family": "36h11",
        "ids": args.ids,
        "outer_black_border_mm": args.tag_edge_mm,
        "pixels_per_mm": args.pixels_per_mm,
        "png_dpi": args.pixels_per_mm * MM_PER_INCH,
        "marker_pixels": marker_pixels,
        "individual_quiet_zone_pixels_per_side": quiet_pixels,
        "individual_png_detected_ids": {},
        "sheet_png_detected_ids": detected_ids(sheet_png, dictionary),
    }
    for tag_id in args.ids:
        stem = "apriltag_36h11_id_%d_%.2fmm" % (tag_id, args.tag_edge_mm)
        verification["individual_png_detected_ids"][str(tag_id)] = detected_ids(
            output_dir / (stem + ".png"), dictionary
        )
    with (output_dir / "verification.json").open("w", encoding="utf-8") as handle:
        json.dump(verification, handle, indent=2)
        handle.write("\n")

    expected_ids = sorted(args.ids)
    if verification["sheet_png_detected_ids"] != expected_ids:
        raise RuntimeError(
            "combined sheet decoded %s, expected %s"
            % (verification["sheet_png_detected_ids"], expected_ids)
        )
    for tag_id in args.ids:
        if verification["individual_png_detected_ids"][str(tag_id)] != [tag_id]:
            raise RuntimeError("individual tag %d did not decode correctly" % tag_id)

    print("wrote %s" % output_dir)
    print("verified AprilTag IDs %s" % expected_ids)
    print(
        "outer black border: %.2f mm = %d pixels at %.0f dpi"
        % (
            args.tag_edge_mm,
            marker_pixels,
            args.pixels_per_mm * MM_PER_INCH,
        )
    )


if __name__ == "__main__":
    main()
