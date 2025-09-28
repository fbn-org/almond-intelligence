# almond_annotation_tool.py
# -----------------------
# This is the tool we used to annotate almond positions in images for our training and test data.
#
# It was actually written completely by ChatGPT-Agent! Which is very impressive
#

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    from PIL import Image, ImageDraw, ImageTk
except ImportError as e:
    raise SystemExit(
        "Pillow is required for this tool. Install it with 'pip install pillow'."
    ) from e

import tkinter as tk


@dataclass
class ImageAnnotation:
    """Holds annotation points for a single image.

    Attributes
    ----------
    points: List of (x, y) tuples in pixel coordinates relative to the
        original image.
    """

    points: List[Tuple[float, float]]


class AlmondAnnotator:
    """Graphical tool for placing point annotations on images.

    Parameters
    ----------
    image_dir: Path to the directory containing images to annotate.  Any
        file with an extension of ``.jpg``, ``.jpeg``, ``.png``,
        ``.bmp``, ``.tif`` or ``.tiff`` (case‑insensitive) will be
        included.
    output_file: Path to the JSON file where annotations will be saved.
    dot_size: Radius of the annotation dots in pixels (display scale).  The
        dots will be drawn with a diameter of ``dot_size * 2`` on the
        canvas and a similar radius when exporting annotated copies.
    dot_color: Fill colour for the annotation dots.  A hex string (e.g.
        ``'#39FF14'`` for neon green) or any Tkinter colour name may be
        supplied.
    window_size: Tuple of (width, height) specifying the maximum size
        of the image display area.  Images larger than this will be
        scaled down proportionally to fit while preserving their
        aspect ratio.
    """

    def __init__(
        self,
        image_dir: str,
        output_file: str = "annotations.json",
        dot_size: int = 5,
        dot_color: str = "#39FF14",
        window_size: Tuple[int, int] = (1000, 700),
    ) -> None:
        # Validate and collect image paths
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in sorted(os.listdir(image_dir))
            if os.path.splitext(f)[1].lower() in supported_exts
        ]
        if not self.image_paths:
            raise FileNotFoundError(
                f"No supported image files were found in '{image_dir}'."
            )

        self.output_file = output_file
        self.dot_size = dot_size
        self.dot_color = dot_color
        self.window_width, self.window_height = window_size
        self.current_index = 0
        # Track rotation (0=0°, 1=90° clockwise, 2=180°, 3=270°)
        self.rotation = 0
        # Mapping of image basename to ImageAnnotation
        self.annotations: Dict[str, ImageAnnotation] = {}
        # Load any existing annotations
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Convert lists to tuples for internal consistency
                for k, v in data.items():
                    self.annotations[k] = ImageAnnotation(
                        points=[(float(x), float(y)) for x, y in v]
                    )
            except Exception:
                # If the JSON cannot be parsed just start fresh
                pass
        # Tkinter setup
        self.root = tk.Tk()
        self.root.title("Almond Annotation Tool")
        # Canvas for displaying images and dots
        self.canvas = tk.Canvas(self.root, cursor="cross")
        self.canvas.pack(side=tk.TOP, expand=True, fill=tk.BOTH)
        # Control buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.prev_btn = tk.Button(btn_frame, text="Prev", command=self.prev_image)
        self.prev_btn.pack(side=tk.LEFT)
        self.next_btn = tk.Button(btn_frame, text="Next", command=self.next_image)
        self.next_btn.pack(side=tk.LEFT)
        self.undo_btn = tk.Button(btn_frame, text="Undo", command=self.undo_point)
        self.undo_btn.pack(side=tk.LEFT)
        self.clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear_points)
        self.clear_btn.pack(side=tk.LEFT)
        self.save_btn = tk.Button(btn_frame, text="Save", command=self.save_annotations)
        self.save_btn.pack(side=tk.LEFT)
        self.export_btn = tk.Button(
            btn_frame, text="Export", command=self.export_annotated_image
        )
        self.export_btn.pack(side=tk.LEFT)
        # Rotate button to rotate current image by 90 degrees clockwise
        self.rotate_btn = tk.Button(
            btn_frame, text="Rotate", command=self.rotate_image
        )
        self.rotate_btn.pack(side=tk.LEFT)

        # Bind click event to add points
        self.canvas.bind("<Button-1>", self.add_point)

        self.root.bind("<Return>", lambda event: self.next_image())

        # Internal state for currently displayed image
        self.display_image = None  # PIL Image at display size
        self.tk_image = None  # Tkinter PhotoImage
        self.scale_x = 1.0
        self.scale_y = 1.0
        # Store original width and height for current image
        self.orig_w = 0
        self.orig_h = 0
        self.dot_ids: List[int] = []  # Canvas IDs for drawn dots
        # Load first image
        self.load_image()
        # Start the Tkinter main loop
        self.root.mainloop()

    def load_image(self) -> None:
        """Load the current image applying current rotation and redraw annotations.

        This method loads the image at ``self.current_index`` without
        modifying the rotation state.  It computes scaling factors to
        fit the rotated image within the window and draws any existing
        annotations transformed to the rotated coordinate system.
        """
        path = self.image_paths[self.current_index]
        # Load original image and store its dimensions
        original = Image.open(path)
        self.orig_w, self.orig_h = original.size
        # Apply rotation to obtain the orientation currently selected
        rotated = self._apply_rotation(original)
        rot_w, rot_h = rotated.size
        # Determine scaling factor to fit the window
        max_w, max_h = self.window_width, self.window_height
        scale = min(max_w / rot_w, max_h / rot_h, 1.0)
        disp_w, disp_h = int(rot_w * scale), int(rot_h * scale)
        # Scaling factors from display coordinates to rotated image coordinates
        self.scale_x = rot_w / disp_w if disp_w != 0 else 1.0
        self.scale_y = rot_h / disp_h if disp_h != 0 else 1.0
        # Resize rotated image for display
        self.display_image = rotated.resize((disp_w, disp_h), Image.LANCZOS)
        # Convert to Tk PhotoImage
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        # Configure canvas size
        self.canvas.config(width=disp_w, height=disp_h)
        # Clear canvas and draw image
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        # Draw existing annotations transformed to rotated/display coordinates
        self.dot_ids.clear()
        basename = os.path.basename(path)
        if basename in self.annotations:
            for x_o, y_o in self.annotations[basename].points:
                # Convert original coordinates to rotated coordinates
                x_r, y_r = self._orig_to_rotated(x_o, y_o)
                # Convert rotated coordinates to display coordinates
                disp_x = x_r / self.scale_x
                disp_y = y_r / self.scale_y
                dot_id = self.canvas.create_oval(
                    disp_x - self.dot_size,
                    disp_y - self.dot_size,
                    disp_x + self.dot_size,
                    disp_y + self.dot_size,
                    fill=self.dot_color,
                    outline="",
                )
                self.dot_ids.append(dot_id)
        # Update window title with progress info
        self.root.title(
            f"Almond Annotation Tool – {basename} ({self.current_index + 1}/{len(self.image_paths)})"
        )

    # --- Rotation helpers ----------------------------------------------------
    def _apply_rotation(self, image: Image.Image) -> Image.Image:
        """Return a copy of ``image`` rotated according to ``self.rotation``.

        Rotation is applied in 90° increments clockwise.  The mapping
        between the rotation value and the applied transformation is:

        ``0`` – no rotation
        ``1`` – 90° clockwise
        ``2`` – 180°
        ``3`` – 270° clockwise (or 90° counter‑clockwise)

        PIL's transpose method is used for speed and to automatically
        handle dimension swapping.
        """
        if self.rotation == 0:
            return image
        elif self.rotation == 1:
            # 90° clockwise is equivalent to ROTATE_270 (90° counter‑clockwise is ROTATE_90)
            return image.transpose(Image.ROTATE_270)
        elif self.rotation == 2:
            return image.transpose(Image.ROTATE_180)
        elif self.rotation == 3:
            # 270° clockwise (90° counter‑clockwise)
            return image.transpose(Image.ROTATE_90)
        else:
            # Should never happen
            return image

    def _orig_to_rotated(self, x_o: float, y_o: float) -> Tuple[float, float]:
        """Convert original image coordinates to rotated image coordinates.

        Parameters
        ----------
        x_o, y_o: Coordinates in the original image coordinate system.

        Returns
        -------
        x_r, y_r: Coordinates in the rotated image coordinate system.
        """
        # Width and height of the original image
        w, h = self.orig_w, self.orig_h
        if self.rotation == 0:
            return x_o, y_o
        elif self.rotation == 1:
            # 90° clockwise: (x_o, y_o) -> (x_r, y_r) = (h - 1 - y_o, x_o)
            return (h - 1 - y_o, x_o)
        elif self.rotation == 2:
            # 180°: (x_o, y_o) -> (w - 1 - x_o, h - 1 - y_o)
            return (w - 1 - x_o, h - 1 - y_o)
        elif self.rotation == 3:
            # 270° clockwise: (x_o, y_o) -> (y_o, w - 1 - x_o)
            return (y_o, w - 1 - x_o)
        else:
            return x_o, y_o

    def _rotated_to_original(self, x_r: float, y_r: float) -> Tuple[float, float]:
        """Convert rotated image coordinates back to original image coordinates.

        Parameters
        ----------
        x_r, y_r: Coordinates in the rotated image coordinate system.

        Returns
        -------
        x_o, y_o: Coordinates in the original image coordinate system.
        """
        w, h = self.orig_w, self.orig_h
        if self.rotation == 0:
            return x_r, y_r
        elif self.rotation == 1:
            # 90° clockwise: inverse of (x_r, y_r) = (h - 1 - y_o, x_o)
            # solves to: x_o = y_r, y_o = (h - 1) - x_r
            return (y_r, (h - 1) - x_r)
        elif self.rotation == 2:
            # 180°: inverse of (x_r, y_r) = (w - 1 - x_o, h - 1 - y_o)
            # solves to: x_o = (w - 1) - x_r, y_o = (h - 1) - y_r
            return ((w - 1) - x_r, (h - 1) - y_r)
        elif self.rotation == 3:
            # 270° clockwise: inverse of (x_r, y_r) = (y_o, w - 1 - x_o)
            # solves to: x_o = (w - 1) - y_r, y_o = x_r
            return ((w - 1) - y_r, x_r)
        else:
            return x_r, y_r

    def add_point(self, event: tk.Event) -> None:
        """Callback for left mouse clicks; adds a point at the clicked location."""
        # Compute coordinates relative to rotated image size
        x_r = event.x * self.scale_x
        y_r = event.y * self.scale_y
        # Convert rotated coordinates back to original image coordinates
        x_orig, y_orig = self._rotated_to_original(x_r, y_r)
        basename = os.path.basename(self.image_paths[self.current_index])
        # Append annotation
        ann = self.annotations.setdefault(basename, ImageAnnotation(points=[]))
        ann.points.append((x_orig, y_orig))
        # Draw on canvas at click location
        dot_id = self.canvas.create_oval(
            event.x - self.dot_size,
            event.y - self.dot_size,
            event.x + self.dot_size,
            event.y + self.dot_size,
            fill=self.dot_color,
            outline="",
        )
        self.dot_ids.append(dot_id)

    def undo_point(self) -> None:
        """Remove the most recently added annotation for the current image."""
        basename = os.path.basename(self.image_paths[self.current_index])
        if basename in self.annotations and self.annotations[basename].points:
            # Remove last point
            self.annotations[basename].points.pop()
            # Remove from canvas
            if self.dot_ids:
                dot_id = self.dot_ids.pop()
                self.canvas.delete(dot_id)

    def clear_points(self) -> None:
        """Clear all annotations for the current image."""
        basename = os.path.basename(self.image_paths[self.current_index])
        if basename in self.annotations:
            self.annotations[basename].points.clear()
        # Remove all drawn dots from canvas
        for dot_id in self.dot_ids:
            self.canvas.delete(dot_id)
        self.dot_ids.clear()

    def save_annotations(self) -> None:
        """Write the annotations dictionary to the JSON file on disk."""
        # Convert dataclass to plain dict for JSON serialisation
        serialisable: Dict[str, List[List[float]]] = {
            k: [[float(x), float(y)] for x, y in v.points]
            for k, v in self.annotations.items()
        }
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2)
        # Notify user via title flash
        current = self.root.title()
        self.root.title("Saved!")
        # Schedule resetting the title after a short delay
        self.root.after(
            1000, lambda: self.root.title(current)
        )

    def export_annotated_image(self) -> None:
        """Save a copy of the current image with dots drawn onto it.

        The copy is saved as a PNG file with ``_annotated`` appended to the
        original filename.  The original image is never modified.  If
        there are no annotations for the current image this method
        returns without doing anything.
        """
        basename = os.path.basename(self.image_paths[self.current_index])
        if basename not in self.annotations or not self.annotations[basename].points:
            return
        # Open the original image at full resolution
        path = self.image_paths[self.current_index]
        original = Image.open(path).copy()
        draw = ImageDraw.Draw(original)
        # Draw each point as a filled circle; radius scaled relative to original
        # use a radius proportional to the dot_size scaled by the display scale
        # We approximate radius in original pixels by dot_size * scale_x
        # to maintain similar appearance
        radius_x = self.dot_size * self.scale_x
        radius_y = self.dot_size * self.scale_y
        for x, y in self.annotations[basename].points:
            left = x - radius_x
            top = y - radius_y
            right = x + radius_x
            bottom = y + radius_y
            draw.ellipse(
                (left, top, right, bottom), fill=self.dot_color, outline=None
            )
        # Construct new filename
        base, ext = os.path.splitext(path)
        out_path = f"{base}_annotated.png"
        original.save(out_path, format="PNG")
        # Inform user
        current = self.root.title()
        self.root.title(f"Exported {os.path.basename(out_path)}")
        self.root.after(1500, lambda: self.root.title(current))

    def next_image(self) -> None:
        """Advance to the next image, saving annotations first."""
        self.save_annotations()
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            # Reset rotation when moving to a different image
            self.rotation = 0
            self.load_image()

    def prev_image(self) -> None:
        """Go back to the previous image, saving annotations first."""
        self.save_annotations()
        if self.current_index > 0:
            self.current_index -= 1
            # Reset rotation when moving to a different image
            self.rotation = 0
            self.load_image()

    def rotate_image(self) -> None:
        """Rotate the current image 90° clockwise and redraw.

        Existing annotations remain in place but are re‑projected to the
        new orientation.  The rotation state is incremented and wraps
        around after a full rotation (four clicks).
        """
        # Increment rotation (0 -> 1 -> 2 -> 3 -> 0)
        self.rotation = (self.rotation + 1) % 4
        self.load_image()


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive tool for annotating almond positions in images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image_dir",
        help="Directory containing images to annotate. Supported extensions: .jpg, .jpeg, .png, .bmp, .tif, .tiff.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="annotations.json",
        help="JSON file to save annotations.",
    )
    parser.add_argument(
        "--dot-size",
        type=int,
        default=5,
        help="Radius of annotation dots in pixels.",
    )
    parser.add_argument(
        "--dot-colour",
        "--dot-color",
        default="#39FF14",
        help="Colour of annotation dots. Accepts Tkinter colour names or hex values.",
    )
    parser.add_argument(
        "--window-size",
        default="1000x700",
        help="Maximum width and height for displaying images, formatted as WIDTHxHEIGHT.",
    )
    args = parser.parse_args(argv)
    # Parse window size
    try:
        width_str, height_str = args.window_size.lower().split("x")
        args.window_size = (int(width_str), int(height_str))
    except Exception:
        parser.error("--window-size must be in the format WIDTHxHEIGHT, e.g. 1000x700")
    return args


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    AlmondAnnotator(
        image_dir=args.image_dir,
        output_file=args.output,
        dot_size=args.dot_size,
        dot_color=args.dot_colour,
        window_size=args.window_size,
    )


if __name__ == "__main__":
    main(sys.argv[1:])