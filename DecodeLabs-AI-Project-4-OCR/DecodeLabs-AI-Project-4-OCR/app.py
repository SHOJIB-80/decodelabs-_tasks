"""Optional Streamlit UI: python -m streamlit run app.py"""
import cv2
import streamlit as st

from ocr_engine import decode_image_bytes, get_image_limits, recognize

st.set_page_config(page_title="VisionText AI | DecodeLabs P4", layout="wide", page_icon="🔎")
st.title("VisionText AI")
st.caption("DecodeLabs Project 4 · Optical Character Recognition")
st.write(
    "Upload a printed-text image to extract readable text. Word confidence filtering "
    "defaults to 80%. Tesseract confidence is an estimate, not independently measured accuracy."
)

try:
    max_upload_bytes, _ = get_image_limits()
except ValueError as exc:
    st.error(str(exc))
    st.stop()
max_upload_mb = max(1, max_upload_bytes // (1024 * 1024))

left, right = st.columns([2, 1])
with left:
    upload = st.file_uploader(
        "Choose JPG or PNG image",
        type=["png", "jpg", "jpeg"],
        max_upload_size=max_upload_mb,
        help=f"Maximum file size: {max_upload_mb} MB. Decoded dimensions are also limited.",
    )
with right:
    threshold = st.slider("Minimum word confidence (%)", 0, 100, 80)
    psm = st.selectbox(
        "Page layout",
        [6, 3, 11, 7],
        format_func=lambda x: {
            6: "Block of text",
            3: "Automatic",
            11: "Sparse text",
            7: "Single text line",
        }[x],
    )

if upload:
    try:
        raw = upload.getvalue()
        img = decode_image_bytes(raw)
        height, width = img.shape[:2]
        result = recognize(img, min_confidence=threshold, psm=psm)
    except (RuntimeError, ValueError, OSError) as exc:
        st.error(str(exc))
    else:
        st.caption(
            f"Input: {upload.name} · {len(raw) / (1024 * 1024):.2f} MB · "
            f"{width:,} × {height:,} px"
        )
        a, b = st.columns(2)
        with a:
            st.subheader("Original image")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.subheader("Preprocessed image")
            st.image(result["preprocessed"], clamp=True, use_container_width=True)
        with b:
            st.subheader("Detected text regions")
            st.image(cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB), use_container_width=True)
            st.metric("Words passing threshold", result["accepted_word_count"])
            st.metric(
                "Average accepted-word confidence",
                f"{result['mean_accepted_confidence']:.1f}%",
            )
            st.metric("OCR processing time", f"{result['processing_time_seconds']:.2f} s")
            st.caption(
                f"Method: {result['variant']} · Tesseract's confidence is not a validated "
                "measure of recognition accuracy."
            )
        st.subheader("Extracted text")
        st.code(result["text"] or "No text passed the threshold", language=None)
        st.download_button(
            "Download extracted text",
            result["text"],
            file_name="extracted_text.txt",
            mime="text/plain",
        )
else:
    st.info("Upload a sample image to begin, or use samples/sample_document.png.")
