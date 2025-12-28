import streamlit as st
from PIL import Image
import io
import numpy as np
import cv2
import random
import urllib.parse

st.markdown("""
<head>
<meta name="google-site-verification" content="UOB8ksPEWmyI2-x2bCPRcwEWxcQKJPbmeSt6mA_EjX4" />
</head>
""", unsafe_allow_html=True)

# ======================
# 画像処理本体
# ======================
def process_image(img, strength):
    # 完全無加工
    if strength == 0:
        return img.copy()

    MAX_SIZE = 1100
    EDGE_LOW = 80
    EDGE_HIGH = 160

    EDGE_NOISE_STD = 1.2 * strength
    GLOBAL_NOISE_STD = 0.4 * strength

    HSV_H_SHIFT = int(2 * strength)
    HSV_S_SHIFT = int(3 * strength)
    HSV_V_SHIFT = int(2 * strength)

    EDGE_SHIFT_RANGE = [-2, -1, 1, 2] if strength > 0.2 else [-1, 1]
    asym_std = 0.3 * strength

    img = img.convert("RGB")
    img.thumbnail((MAX_SIZE, MAX_SIZE), Image.LANCZOS)
    img_np = np.array(img)

    h, w, _ = img_np.shape

    # エッジ検出
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, EDGE_LOW, EDGE_HIGH)
    edge_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    perturbed = img_np.copy()

    # 輪郭破壊
    for y in range(h):
        for x in range(w):
            if edge_mask[y, x] > 0:
                dx = random.choice(EDGE_SHIFT_RANGE)
                dy = random.choice(EDGE_SHIFT_RANGE)
                nx = min(max(x + dx, 0), w - 1)
                ny = min(max(y + dy, 0), h - 1)
                perturbed[y, x] = img_np[ny, nx]

    # 輪郭限定ノイズ
    edge_noise = np.random.normal(0, EDGE_NOISE_STD, perturbed.shape).astype(np.float32)
    mask_3ch = np.repeat(edge_mask[:, :, None] > 0, 3, axis=2)
    perturbed = perturbed.astype(np.float32)
    perturbed[mask_3ch] += edge_noise[mask_3ch]

    # HSV揺らし
    hsv = cv2.cvtColor(np.clip(perturbed, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsv[:, :, 0] = np.clip(
        hsv[:, :, 0] + np.random.randint(-HSV_H_SHIFT, HSV_H_SHIFT + 1, (h, w)), 0, 179
    )
    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1] + np.random.randint(-HSV_S_SHIFT, HSV_S_SHIFT + 1, (h, w)), 0, 255
    )
    hsv[:, :, 2] = np.clip(
        hsv[:, :, 2] + np.random.randint(-HSV_V_SHIFT, HSV_V_SHIFT + 1, (h, w)), 0, 255
    )

    perturbed = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)

    # グローバルノイズ
    perturbed += np.random.normal(0, GLOBAL_NOISE_STD, perturbed.shape)

    # 非対称性
    if random.random() < 0.5:
        perturbed[:, :w//2] += np.random.normal(0, asym_std, perturbed[:, :w//2].shape)
    else:
        perturbed[:, w//2:] += np.random.normal(0, asym_std, perturbed[:, w//2:].shape)

    perturbed = np.clip(perturbed, 0, 255).astype(np.uint8)
    return Image.fromarray(perturbed)

# ======================
# GIF 1秒ループ用フレーム生成
# ======================
def make_gif_frames(base_img, strength, frames=8):
    imgs = []
    base_np = np.array(base_img).astype(np.float32)

    for _ in range(frames):
        frame = base_np.copy()
        # 知覚不可レベルの微差
        frame += np.random.normal(0, 0.15 * max(strength, 0.3), frame.shape)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        imgs.append(Image.fromarray(frame))

    return imgs

# ======================
# Streamlit UI
# ======================
st.title("静止画をそのままGIFまたはJPGに変換")

st.write("① 画像をアップロード → ② 出力形式を選択 → ③ ダウンロード")

uploaded = st.file_uploader(
    "画像をアップロードしてください",
    type=["png", "jpg", "jpeg", "webp", "gif"]
)

if not uploaded:
    st.info("画像を1枚アップロードしてください。")

output_format = st.radio(
    "出力形式",
    ["gif", "jpg"],
    horizontal=True
)

strength = st.slider(
    "加工の強さ（0 = 見た目そのまま）",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05
)

st.caption("※ 静止画の雰囲気を保ちたい場合は低めがおすすめです")

tweet_text = st.text_input(
    "X投稿文（任意）",
    value="投稿文（任意）"
)

if uploaded:
    img = Image.open(uploaded)
    if img.format == "GIF":
        img.seek(0)

    result = process_image(img, strength)
    st.image(result, caption=f"プレビュー（1秒ループGIF用・強度 {strength}）")

    buf = io.BytesIO()

    if output_format == "jpg":
        result.save(buf, format="JPEG", quality=75, subsampling=2)
        mime = "image/jpeg"
        ext = "jpg"
    else:
        if strength == 0:
            frames = [img.copy()]
            frames[0].save(
                buf,
                format="GIF",
                save_all=True,
                duration=1000,
                loop=0
            )
        else:
            frames = make_gif_frames(result, strength, frames=8)
            frames[0].save(
                buf,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=125,
                loop=0,
                disposal=2
            )

        mime = "image/gif"
        ext = "gif"

    st.download_button(
        "画像をダウンロード",
        data=buf.getvalue(),
        file_name=f"output.{ext}",
        mime=mime
    )

    intent_url = "https://twitter.com/intent/tweet?text=" + urllib.parse.quote(tweet_text)

    st.markdown(f"""
    <a href="{intent_url}" target="_blank">
        <button style="
            padding:10px 18px;
            font-size:16px;
            background-color:#1DA1F2;
            color:white;
            border:none;
            border-radius:6px;
            cursor:pointer;">
            Xに投稿する
        </button>
    </a>
    """, unsafe_allow_html=True)

    st.caption("※ Xのログイン情報は取得されません（公式投稿画面が開きます）")


