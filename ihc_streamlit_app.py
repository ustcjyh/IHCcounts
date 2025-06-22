# ---------- 1. 依赖 ----------
import os, csv, zipfile
from io import BytesIO

import streamlit as st
import numpy as np
from skimage import io, measure, morphology
import matplotlib.pyplot as plt

# ---------- 2. 单张图片处理函数 ----------
def process_image(image_bytes, r_th, g_th, b_th, min_size):
    """返回 (label 数, 原图 ndarray, 二值/过滤后 mask)"""
    # 读入图片（Streamlit 上传的是 BytesIO）
    img = io.imread(image_bytes)
    if img.shape[-1] == 4:        # 有透明通道时去掉 Alpha
        img = img[:, :, :3]

    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # 紫红细胞的简单阈值：R > r_th, B > b_th, G < g_th
    mask = (r > r_th) & (b > b_th) & (g < g_th)

    # 去掉小杂点
    mask = morphology.remove_small_objects(mask, min_size=min_size)

    # 连通域计数
    labels = measure.label(mask)
    n_cells = labels.max()

    return n_cells, img, mask

# ---------- 3. Streamlit UI ----------
st.set_page_config(page_title="紫红染色细胞计数", layout="wide")
st.title("紫红染色细胞自动计数 (Streamlit)")

st.markdown(
    """
    **使用方法**  
    1. 先上传单张 `.tif/.png/.jpg` 查看效果；  
    2. 滑块微调阈值 & 最小面积直至计数/掩膜满意；  
    3. 如需批量处理，把所有图片打包成 **ZIP** 上传即可批量输出 CSV + 掩膜图。
    ---
    """
)

# --- 单张图片区 -----------------------------------------------------------
col1, col2 = st.columns([1, 2])
with col1:
    uploaded_img = st.file_uploader("▶ 上传单张图片", type=["tif", "png", "jpg"])

with col2:
    # 默认阈值：根据示例图大致经验值
    r_th   = st.slider("红通道下限 (R >)", 0, 255, 150, key="r_th")
    b_th   = st.slider("蓝通道下限 (B >)", 0, 255, 110, key="b_th")
    g_th   = st.slider("绿通道上限 (G <)", 0, 255, 120, key="g_th")
    min_sz = st.slider("最小目标面积 (像素)", 0, 2000, 80, 10, key="min_sz")

if uploaded_img:
    n, original, mask = process_image(uploaded_img, r_th, g_th, b_th, min_sz)

    st.success(f"检测到细胞 (连通域) 数量：**{n}**")
    # 画图
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original)
    ax[0].set_title("原图")
    ax[0].axis("off")
    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title(f"掩膜 / 细胞计数：{n}")
    ax[1].axis("off")
    st.pyplot(fig)

# --- 批量处理区 -----------------------------------------------------------
st.markdown("---")
st.subheader("批量 ZIP 处理")

zip_file = st.file_uploader("▶ 上传包含多张图片的 ZIP", type="zip")
csv_name = st.text_input("输出 CSV 文件名", "results.csv")

if zip_file and st.button("开始批量处理"):
    st.info("⏳ 正在解压并处理，请稍候……")
    tmp_dir = "tmp_uploaded_imgs"
    os.makedirs(tmp_dir, exist_ok=True)

    # 解压
    with zipfile.ZipFile(BytesIO(zip_file.read())) as zf:
        zf.extractall(tmp_dir)

    tif_list = [
        os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)
        if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))
    ]

    results = []
    for fp in tif_list:
        with open(fp, "rb") as f:
            n, img, msk = process_image(f, r_th, g_th, b_th, min_sz)
        results.append([os.path.basename(fp), n])

        # 保存掩膜示意图到同目录 (可选)
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(img);  ax[0].axis("off")
        ax[1].imshow(msk, cmap="gray"); ax[1].axis("off")
        plt.tight_layout()
        plt.savefig(fp + "_mask.png", dpi=150)
        plt.close(fig)

    # 写 CSV
    csv_path = os.path.join(tmp_dir, csv_name)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Cell_Count"])
        writer.writerows(results)

    st.success("✅ 批量完成！点击下方按钮下载结果：")
    with open(csv_path, "rb") as f:
        st.download_button(
            "📥 下载 CSV", f, file_name=csv_name, mime="text/csv"
        )
