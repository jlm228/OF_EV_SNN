"""Reusable video utilities for comparing SNN outputs (e.g. clean vs. attacked)."""

import cv2
import numpy as np


def combine_side_by_side(video_paths, labels, output_path, fps=None, label_height=40):
    """Concatenate videos horizontally into one .mp4, with a text label above each.

    Used to make attack-vs-clean flow comparisons viewable as a single file
    (rather than two separate videos), so any threat's ``plot_evolution`` output
    can be combined the same way.

    Parameters
    ----------
    video_paths : sequence of str
        Paths to input videos (read frame-by-frame; must share height/be readable
        by OpenCV). Panels are placed left to right in the given order.
    labels : sequence of str
        One label per video, drawn in the header bar above its panel.
    output_path : str
        Destination .mp4 path.
    fps : float, optional
        Output frame rate; defaults to the first input video's fps.
    label_height : int
        Height in pixels of the text header bar above each panel.
    """
    if len(video_paths) != len(labels):
        raise ValueError("video_paths and labels must have the same length.")

    caps = [cv2.VideoCapture(p) for p in video_paths]
    try:
        for p, cap in zip(video_paths, caps):
            if not cap.isOpened():
                raise RuntimeError(f"Could not open video: {p}")

        n_frames = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT) for cap in caps))
        if n_frames <= 0:
            raise RuntimeError("One or more input videos have no frames.")
        if fps is None:
            fps = caps[0].get(cv2.CAP_PROP_FPS) or 10.0

        widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in caps]
        heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in caps]
        panel_h = max(heights)
        total_w = sum(widths)
        out_h = panel_h + label_height

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (total_w, out_h), isColor=True)
        try:
            for _ in range(n_frames):
                panels = []
                for cap, w, h, label in zip(caps, widths, heights, labels):
                    ok, frame = cap.read()
                    if not ok:
                        frame = np.zeros((h, w, 3), dtype=np.uint8)
                    header = np.zeros((label_height, w, 3), dtype=np.uint8)
                    cv2.putText(header, label, (10, label_height - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                                cv2.LINE_AA)
                    panels.append(np.concatenate([header, frame], axis=0))
                out.write(np.concatenate(panels, axis=1))
        finally:
            out.release()
    finally:
        for cap in caps:
            cap.release()
