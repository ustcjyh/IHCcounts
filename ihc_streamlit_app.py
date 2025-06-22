# -*- coding: utf-8 -*-
"""
Streamlit App: IHC Brown-Positive Cell Quantification
====================================================

Upload multiple IHC slides, adjust brown HSV thresholds via sliders,
segment nuclei with watershed, and obtain per‑slide brown‑positive
cell counts. Outputs CSV and optional QC overlay ZIP.

Run locally:
    pip install streamlit opencv-python-headless numpy pandas
    streamlit run ihc_streamlit_app.py
"""

from __future__ import annotations
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile, zipfile, io
from pathlib import Path

# Default thresholds
HSV_THRESH_DEFAULT = ([10, 50, 20], [30, 255, 200])  # brown lower & upper
CELL_HSV = ([100, 20, 20], [140, 255, 255])          # blue/purple nuclei
DIST_THR = 0.4                                       # distance transform threshold

# ---------------- Core helpers ----------------
def overlay_mask(img: np.ndarray, mask: np.ndarray,
                 color=(0, 255, 0), alpha=0.5):
    overlay = img.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def generate_masks(hsv: np.ndarray,
                   hsv_ranges: dict[str, tuple[list[int], list[int]]]):
    masks = {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for label, (lo, hi) in hsv_ranges.items():
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        masks[label] = m
    return masks

def segment_cells(cell_mask: np.ndarray, dist_thr: float = DIST_THR):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(cell_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, dist_thr * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    cv2.watershed(cv2.cvtColor(cell_mask, cv2.COLOR_GRAY2BGR), markers)
    markers[markers == -1] = 0
    return markers

def cell_metrics(markers: np.ndarray, brown_mask: np.ndarray):
    total = positive = 0
    for lbl in np.unique(markers):
        if lbl <= 1:
            continue
        cell = markers == lbl
        if cell.sum() < 10:
            continue
        total += 1
        if np.logical_and(cell, brown_mask > 0).any():
            positive += 1
    percent = positive / total * 100 if total else 0
    return total, positive, round(percent, 2)

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="IHC Brown Counter", layout="wide")
st.title("🧬 IHC Brown‑Positive Cell Quantification")
st.markdown("Upload slides → adjust HSV → run → download CSV / QC.")

with st.sidebar:
    st.header("⚙️ Parameters")
    lo_def, hi_def = HSV_THRESH_DEFAULT
    h_min = st.slider("H min", 0, 179, lo_def[0])
    h_max = st.slider("H max", 0, 179, hi_def[0])
    s_min = st.slider("S min", 0, 255, lo_def[1])
    s_max = st.slider("S max", 0, 255, hi_def[1])
    v_min = st.slider("V min", 0, 255, lo_def[2])
    v_max = st.slider("V max", 0, 255, hi_def[2])
    dist_thr = st.slider("Distance transform threshold", 0.2, 0.6, DIST_THR, 0.05)
    export_qc = st.checkbox("Save QC overlays (ZIP)", True)

st.subheader("① Upload images (PNG/JPG/TIF or ZIP)")
files = st.file_uploader("Upload files", type=["png","jpg","jpeg","tif","tiff","zip"], accept_multiple_files=True)

if not files:
    st.stop()

tmpdir = Path(tempfile.mkdtemp())
image_paths: list[Path] = []
for f in files:
    if f.name.lower().endswith('.zip'):
        with zipfile.ZipFile(f) as zf:
            zf.extractall(tmpdir)
            image_paths.extend([tmpdir / p for p in zf.namelist() if (tmpdir / p).is_file()])
    else:
        p = tmpdir / f.name
        p.write_bytes(f.read())
        image_paths.append(p)

st.subheader("② Run analysis")
if st.button("Run analysis", type="primary"):
    rows = []
    csv_buf = io.StringIO()
    qc_zip_buf = io.BytesIO()
    qc_zip = zipfile.ZipFile(qc_zip_buf, 'w', zipfile.ZIP_DEFLATED)
    prog = st.progress(0.0)

    for idx, img_path in enumerate(image_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            st.warning(f"Skip unreadable {img_path.name}")
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_ranges = {
            'brown': ([h_min, s_min, v_min], [h_max, s_max, v_max]),
            'cell': CELL_HSV,
        }
        masks = generate_masks(hsv, hsv_ranges)
        markers = segment_cells(masks['cell'], dist_thr)
        total, pos, pct = cell_metrics(markers, masks['brown'])
        rows.append(dict(filename=img_path.name, total_cells=total,
                         brown_positive_cells=pos, positive_percent=pct))

        if export_qc:
            qc = img.copy()
            for lbl in np.unique(markers):
                if lbl <= 1:
                    continue
                cell = markers == lbl
                if cell.sum() < 10:
                    continue
                color = (0,255,0) if np.logical_and(cell, masks['brown']>0).any() else (0,0,255)
                qc[cell] = color
            qc = cv2.addWeighted(qc,0.4, img,0.6,0)
            _, buf = cv2.imencode('.png', qc)
            qc_zip.writestr(f"{img_path.stem}_overlay.png", buf.tobytes())

        prog.progress(idx/len(image_paths))

    df = pd.DataFrame(rows)
    df.to_csv(csv_buf, index=False)

    st.subheader("③ Results")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", csv_buf.getvalue(), "ihc_metrics.csv", "text/csv")
    if export_qc:
        qc_zip.close()
        st.download_button("Download QC ZIP", qc_zip_buf.getvalue(), "qc_overlays.zip", "application/zip")
    st.success("Done!")
