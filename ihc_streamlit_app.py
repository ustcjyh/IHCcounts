# -*- coding: utf-8 -*-
"""
Streamlit App: IHC Brown-Positive Cell Quantification (Live Preview)
====================================================================

* Upload PNG/JPG/TIF images *or* a ZIP archive.
* Sidebar HSV sliders adjust brown (DAB) thresholds and watershed distance threshold.
* Live preview: first image shows positive (green) vs negative (red) cell overlay
  with real‑time statistics as you move the sliders.
* Run batch analysis on all images to obtain CSV and optional QC overlay ZIP.

Run locally:
    pip install streamlit opencv-python-headless numpy pandas
    streamlit run ihc_streamlit_app.py
"""

from __future__ import annotations
import streamlit as st
import cv2, numpy as np, pandas as pd
import tempfile, zipfile, io
from pathlib import Path

# ---------------- Default thresholds ----------------
HSV_DEFAULT = ([10, 50, 20], [30, 255, 200])     # brown lower, upper
CELL_HSV    = ([100, 20, 20], [140, 255, 255])   # nuclei
DIST_THR_DF = 0.4                                # watershed distance threshold

# ---------------- Helper functions ------------------
def overlay_mask(img, mask, color=(0, 255, 0), alpha=0.5):
    overlay = img.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def generate_masks(hsv, hsv_ranges):
    masks = {}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for lab, (lo, hi) in hsv_ranges.items():
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        masks[lab] = m
    return masks

def segment_cells(cell_mask, dist_thr=DIST_THR_DF):
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opening = cv2.morphologyEx(cell_mask, cv2.MORPH_OPEN, ker, iterations=2)
    sure_bg = cv2.dilate(opening, ker, iterations=3)
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

def cell_metrics(markers, brown_mask):
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
    pct = positive / total * 100 if total else 0
    return total, positive, round(pct, 2)

# ---------------- Streamlit UI ----------------------
st.set_page_config(page_title="IHC Brown Counter", layout="wide")
st.title("🧬 IHC Brown‑Positive Cell Quantification")

st.markdown("Upload slides → adjust HSV thresholds → live preview → run batch analysis.")

# Sidebar parameters
with st.sidebar:
    st.header("⚙️ Thresholds")
    lo, hi = HSV_DEFAULT
    h_min = st.slider("H min", 0, 179, lo[0])
    h_max = st.slider("H max", 0, 179, hi[0])
    s_min = st.slider("S min", 0, 255, lo[1])
    s_max = st.slider("S max", 0, 255, hi[1])
    v_min = st.slider("V min", 0, 255, lo[2])
    v_max = st.slider("V max", 0, 255, hi[2])
    dist_thr = st.slider("Watershed dist‑thr", 0.2, 0.6, DIST_THR_DF, 0.05)
    export_qc = st.checkbox("Save QC overlay ZIP", value=True)

# File upload
st.subheader("① Upload images or ZIP")
files = st.file_uploader("Choose files", type=["png","jpg","jpeg","tif","tiff","zip"], accept_multiple_files=True)
if not files:
    st.stop()

tmpdir = Path(tempfile.mkdtemp())
img_paths: list[Path] = []
for f in files:
    if f.name.lower().endswith('.zip'):
        with zipfile.ZipFile(f) as zf:
            zf.extractall(tmpdir)
            img_paths += [tmpdir / p for p in zf.namelist() if (tmpdir / p).is_file()]
    else:
        p = tmpdir / f.name
        p.write_bytes(f.read())
        img_paths.append(p)
if not img_paths:
    st.warning("No valid images found")
    st.stop()

# Thumbnails
st.subheader("② Thumbnail preview (first 4)")
cols = st.columns(min(4, len(img_paths)))
for c, p in zip(cols, img_paths[:4]):
    img = cv2.imread(str(p))
    c.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=p.name, use_column_width=True)

# Live overlay preview
st.subheader("③ Live cell overlay (first image)")
first_img = cv2.imread(str(img_paths[0]))
if first_img is not None:
    hsv = cv2.cvtColor(first_img, cv2.COLOR_BGR2HSV)
    ranges = {'brown': ([h_min,s_min,v_min],[h_max,s_max,v_max]), 'cell': CELL_HSV}
    masks = generate_masks(hsv, ranges)
    markers = segment_cells(masks['cell'], dist_thr)
    tot, pos, pct = cell_metrics(markers, masks['brown'])
    overlay = first_img.copy()
    for lbl in np.unique(markers):
        if lbl<=1: continue
        reg = markers==lbl
        if reg.sum()<10: continue
        color = (0,255,0) if np.logical_and(reg, masks['brown']>0).any() else (0,0,255)
        overlay[reg] = color
    overlay = cv2.addWeighted(overlay,0.4, first_img,0.6,0)
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
             caption=f"Pos: {pos}/{tot} ({pct}%)", use_column_width=True)
else:
    st.error("First image unreadable")

st.divider()

# Run analysis
st.subheader("④ Run analysis on all images")
if st.button("Run analysis", type="primary"):
    rows = []
    csv_buf = io.StringIO()
    qc_buf = io.BytesIO()
    qc_zip = zipfile.ZipFile(qc_buf, 'w', zipfile.ZIP_DEFLATED)
    prog = st.progress(0.0)
    for idx, p in enumerate(img_paths, 1):
        img = cv2.imread(str(p))
        if img is None:
            st.warning(f"Skip unreadable {p.name}")
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        masks = generate_masks(hsv, ranges)
        markers = segment_cells(masks['cell'], dist_thr)
        t, posi, perc = cell_metrics(markers, masks['brown'])
        rows.append(dict(filename=p.name,total_cells=t,brown_positive_cells=posi,positive_percent=perc))
        if export_qc:
            qc = img.copy()
            for lbl in np.unique(markers):
                if lbl<=1: continue
                region = markers==lbl
                if region.sum()<10: continue
                col = (0,255,0) if np.logical_and(region, masks['brown']>0).any() else (0,0,255)
                qc[region] = col
            qc = cv2.addWeighted(qc,0.4,img,0.6,0)
            _,buf = cv2.imencode('.png', qc)
            qc_zip.writestr(f"{p.stem}_overlay.png", buf.tobytes())
        prog.progress(idx/len(img_paths))
    df = pd.DataFrame(rows)
    df.to_csv(csv_buf, index=False)
    st.dataframe(df, use_container_width=True)
    st.download_button("Download CSV", csv_buf.getvalue(), "ihc_metrics.csv", "text/csv")
    if export_qc:
        qc_zip.close()
        st.download_button("Download QC ZIP", qc_buf.getvalue(), "qc_overlays.zip", "application/zip")
    st.success("Analysis complete")
