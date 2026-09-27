
import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from skimage.metrics import structural_similarity as ssim

st.set_page_config(page_title="Poster Similarity Judge", page_icon="🧠", layout="wide")

st.title("🧠 Brainiacs — Poster Similarity Judge")
st.caption("Compare a challenge image with a participant-generated poster using multiple visual metrics.")

@st.cache_resource(show_spinner=False)
def load_clip():
    try:
        import torch
        from transformers import CLIPProcessor, CLIPModel
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        return model, processor
    except Exception as e:
        return None, None

def read_cv(pil):
    rgb = np.array(pil.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def resize_pair(a, b, size=(512, 512)):
    a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, size, interpolation=cv2.INTER_AREA)
    return a, b

def structural_score(a, b):
    a, b = resize_pair(a, b)
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return float(ssim(ga, gb, data_range=255)) * 100

def color_score(a, b):
    a, b = resize_pair(a, b)
    ah = cv2.calcHist([a], [0,1], None, [32,32], [0,256,0,256])
    bh = cv2.calcHist([b], [0,1], None, [32,32], [0,256,0,256])
    cv2.normalize(ah, ah)
    cv2.normalize(bh, bh)
    corr = cv2.compareHist(ah, bh, cv2.HISTCMP_CORREL)
    return float(np.clip((corr + 1) / 2, 0, 1) * 100)

def feature_score(a, b):
    a, b = resize_pair(a, b)
    g1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=1200)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
        return 0.0, 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(d1, d2, k=2)
    good = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.72*n.distance]
    # Saturating score: enough strong matches gives a high structural/feature score.
    score = 100 * (1 - np.exp(-len(good) / 35))
    return float(np.clip(score, 0, 100)), len(good)

def clip_score(pil_a, pil_b):
    model, processor = load_clip()
    if model is None:
        return None
    import torch
    inputs = processor(images=[pil_a.convert("RGB"), pil_b.convert("RGB")], return_tensors="pt")
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        sim = float((feats[0] @ feats[1]).item())
    # CLIP cosine similarity typically lies around 0..1 for related images.
    return float(np.clip(sim * 100, 0, 100))

def overall(semantic, structural, color, feature):
    # Semantic gets the largest weight because generated posters can look different
    # while depicting the same concept. Feature/structure helps detect composition.
    if semantic is None:
        return 0.45*structural + 0.20*color + 0.35*feature
    return 0.50*semantic + 0.20*structural + 0.10*color + 0.20*feature

def verdict(score):
    if score >= 85: return "Very high similarity"
    if score >= 70: return "High similarity"
    if score >= 55: return "Moderate similarity"
    if score >= 40: return "Low similarity"
    return "Very low similarity"

original_file = st.file_uploader("1. Upload the ORIGINAL challenge image", type=["png","jpg","jpeg","webp"])
poster_file = st.file_uploader("2. Upload the PARTICIPANT'S generated poster", type=["png","jpg","jpeg","webp"])

if original_file and poster_file:
    original = Image.open(io.BytesIO(original_file.getvalue()))
    poster = Image.open(io.BytesIO(poster_file.getvalue()))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Original")
        st.image(original, use_container_width=True)
    with c2:
        st.subheader("Participant poster")
        st.image(poster, use_container_width=True)

    with st.spinner("Analysing both images..."):
        a = read_cv(original)
        b = read_cv(poster)
        s_struct = structural_score(a, b)
        s_color = color_score(a, b)
        s_feature, match_count = feature_score(a, b)
        s_semantic = clip_score(original, poster)
        final = overall(s_semantic, s_struct, s_color, s_feature)

    st.divider()
    st.subheader("Similarity report")

    cols = st.columns(4)
    cols[0].metric("Semantic / AI", "N/A" if s_semantic is None else f"{s_semantic:.1f}%")
    cols[1].metric("Structure / SSIM", f"{s_struct:.1f}%")
    cols[2].metric("Color", f"{s_color:.1f}%")
    cols[3].metric("Feature matches", f"{s_feature:.1f}%")

    st.progress(int(round(final)))
    st.markdown(f"### Overall similarity: **{final:.1f}%**")
    st.info(verdict(final))
    st.write(f"ORB strong feature matches detected: **{match_count}**")

    st.markdown("### Suggested judging use")
    st.write(
        "Use the overall score as a judging aid, not as the only criterion. "
        "For an AI-poster challenge, semantic similarity should matter more than exact pixels."
    )

    st.download_button(
        "Download score as TXT",
        data=(
            f"Brainiacs Poster Similarity Report\n"
            f"Overall: {final:.1f}%\n"
            f"Semantic/AI: {'N/A' if s_semantic is None else f'{s_semantic:.1f}%'}\n"
            f"Structure/SSIM: {s_struct:.1f}%\n"
            f"Color: {s_color:.1f}%\n"
            f"Feature: {s_feature:.1f}%\n"
            f"ORB matches: {match_count}\n"
            f"Verdict: {verdict(final)}\n"
        ),
        file_name="similarity_report.txt",
        mime="text/plain",
    )
else:
    st.info("Upload both images to start the comparison.")
    st.markdown("""
    **How this tool scores the poster**
    - **Semantic / AI similarity (50%)** — whether both images represent similar visual concepts.
    - **Structure / SSIM (20%)** — broad layout and luminance similarity.
    - **Color (10%)** — similarity of dominant color distribution.
    - **Feature matching (20%)** — matching visual details, shapes, and keypoints.
    """)
