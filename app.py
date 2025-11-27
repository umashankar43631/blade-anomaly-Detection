import json
import os
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List, Optional

import google.generativeai as genai
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

# Load environment variables from local files if present
load_dotenv()
load_dotenv(".env.local")

st.set_page_config(
    page_title="BladeGuard AI – Streamlit",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_STYLES = """
<style>
    .main { background-color: #f8fafc; }
    section[data-testid="stSidebar"] { background: #0f172a; }
    .reportview-container .main .block-container { padding-top: 2rem; }
    .blade-header { display: flex; align-items: center; gap: 0.75rem; }
    .pill { border-radius: 999px; padding: 0.2rem 0.65rem; font-weight: 600; font-size: 0.75rem; display: inline-flex; align-items: center; gap: 0.4rem; }
    .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 16px; padding: 1.25rem; box-shadow: 0 10px 30px -18px rgba(0,0,0,0.25); }
    .muted { color: #6b7280; font-size: 0.9rem; }
    .label { font-size: 0.7rem; letter-spacing: 0.08em; text-transform: uppercase; color: #475569; font-weight: 700; }
</style>
"""
st.markdown(CUSTOM_STYLES, unsafe_allow_html=True)

SEVERITY_COLORS = {
    "Critical": {"stroke": (220, 38, 38), "bg": "#fee2e2", "text": "#991b1b"},
    "High": {"stroke": (234, 88, 12), "bg": "#ffedd5", "text": "#9a3412"},
    "Medium": {"stroke": (202, 138, 4), "bg": "#fef08a", "text": "#854d0e"},
    "Low": {"stroke": (37, 99, 235), "bg": "#dbeafe", "text": "#1d4ed8"},
    "None": {"stroke": (22, 163, 74), "bg": "#dcfce7", "text": "#166534"},
}

DETECTION_CAPABILITIES = [
    "Surface cracks & laminate fractures",
    "Leading edge erosion (paint vs laminate)",
    "Lightning damages (tip & laminate)",
    "Core material defects",
    "Environmental wear (dust/erosion)",
]

SYSTEM_PROMPT = """
      You are a World-Class Wind Turbine Structural Engineer and Certified Blade Inspector with 20 years of experience in Non-Destructive Testing (NDT) and Computer Vision Analysis.
      
      Your task is to analyze the provided image (which may be a standard photo or a binocular/stereo inspection image) of a wind turbine blade.
      
      You must rigorously detect, classify, and LOCALIZE faults with extreme precision.
      
      Step 1: Scrutinize the entire image for any anomalies.
      Step 2: Classify each anomaly into one of the following specific types:
      1. Surface cracks (Paint/Gelcoat)
      2. Leading Edge erosion (Only Paint/Gelcoat)
      3. Leading Edge Erosion (Laminate) - *Critical*
      4. Laminate Crack - *Critical*
      5. Lightning Defect (Laminate Level) - *Critical*
      6. Lightning Defect (Tip Opened) - *Critical*
      7. Laminate defect (Till Core material / Through Laminate) - *Critical*
      8. Surface level erosion
      9. Dust accumulation
      
      Step 3: For EVERY detected defect, you MUST provide a precise bounding box (ymin, xmin, ymax, xmax) normalized to a 1000x1000 scale.
      - 0,0 is the top-left corner.
      - 1000,1000 is the bottom-right corner.
      - The box should tightly enclose the visible defect.
      
      Guidelines:
      - If the image contains binocular views (two similar images side-by-side), detect defects in ALL views where they are visible. Treat the image as a single canvas for coordinates.
      - Be conservative with "Critical" severity; reserve it for structural compromises.
      - If the blade appears healthy, explicitly state "No Defects Detected" and provide a high condition score.
      - Provide a maintenance recommendation for every detected issue.
"""

RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "hasDefects": {"type": "boolean"},
        "bladeConditionScore": {
            "type": "number",
            "description": "A score from 0 to 100 representing the overall health of the blade. 100 is perfect, 0 is destroyed.",
        },
        "summary": {"type": "string", "description": "A concise executive summary of the inspection findings."},
        "defects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "Must be one of: Surface Crack (Paint/Gelcoat), Leading Edge Erosion (Paint/Gelcoat), Leading Edge Erosion (Laminate), Laminate Crack, Lightning Defect (Laminate Level), Lightning Defect (Tip Opened), Laminate Defect (Till Core/Through Laminate), Surface Level Erosion, Dust Accumulation, or Other.",
                    },
                    "severity": {"type": "string", "description": "Low, Medium, High, or Critical"},
                    "confidence": {"type": "number", "description": "Confidence score 0-100"},
                    "location": {"type": "string", "description": "Where on the blade this is located (e.g., Tip, Root, Trailing Edge, Leading Edge)"},
                    "description": {"type": "string", "description": "Detailed visual description of the specific defect found."},
                    "recommendation": {"type": "string", "description": "Actionable maintenance recommendation."},
                    "boundingBox": {
                        "type": "object",
                        "description": "Precise bounding box of the defect with coordinates normalized to 1000 (0-1000 scale).",
                        "properties": {
                            "ymin": {"type": "number", "description": "Top Y coordinate (0-1000)"},
                            "xmin": {"type": "number", "description": "Left X coordinate (0-1000)"},
                            "ymax": {"type": "number", "description": "Bottom Y coordinate (0-1000)"},
                            "xmax": {"type": "number", "description": "Right X coordinate (0-1000)"},
                        },
                        "required": ["ymin", "xmin", "ymax", "xmax"],
                    },
                },
                "required": ["type", "severity", "confidence", "location", "description", "recommendation", "boundingBox"],
            },
        },
    },
    "required": ["hasDefects", "bladeConditionScore", "summary", "defects"],
}


def _get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    if not key:
        raise RuntimeError("Add GEMINI_API_KEY or API_KEY to your environment to run analysis.")
    return key


def _extract_response_text(response: Any) -> str:
    if getattr(response, "text", None):
        return response.text
    for candidate in getattr(response, "candidates", []):
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if parts is None:
            parts = getattr(candidate, "parts", [])
        for part in parts:
            if getattr(part, "text", None):
                return part.text
    raise RuntimeError("No response text returned from Gemini.")


def run_gemini_analysis(image_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    genai.configure(api_key=_get_api_key())
    model = genai.GenerativeModel(model_name="gemini-3-pro-preview")

    response = model.generate_content(
        [
            {"text": SYSTEM_PROMPT},
            {"inline_data": {"mime_type": mime_type, "data": image_bytes}},
        ],
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": RESPONSE_SCHEMA,
            "temperature": 0.1,
        },
    )

    raw_text = _extract_response_text(response)
    parsed = json.loads(raw_text)
    parsed["timestamp"] = datetime.utcnow().isoformat()
    return parsed


def _get_severity_color(severity: str) -> Dict[str, Any]:
    return SEVERITY_COLORS.get(severity, {"stroke": (59, 130, 246), "bg": "#e5e7eb", "text": "#0f172a"})


def draw_annotations(image_bytes: bytes, defects: List[Dict[str, Any]]) -> Image.Image:
    base_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    overlay = base_image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = ImageFont.load_default()
    width, height = base_image.size

    for defect in defects:
        bbox: Optional[Dict[str, float]] = defect.get("boundingBox")
        if not bbox:
            continue

        left = max(0, min(width, int((bbox["xmin"] / 1000) * width)))
        top = max(0, min(height, int((bbox["ymin"] / 1000) * height)))
        right = max(0, min(width, int((bbox["xmax"] / 1000) * width)))
        bottom = max(0, min(height, int((bbox["ymax"] / 1000) * height)))

        color = _get_severity_color(str(defect.get("severity", "Low")).title())
        stroke_color = color["stroke"]
        draw.rectangle([left, top, right, bottom], outline=stroke_color + (255,), width=4)

        label = str(defect.get("type", "Defect"))
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_x = left + 4
        label_y = top + 4
        draw.rectangle(
            [label_x - 2, label_y - 2, label_x + text_width + 6, label_y + text_height + 4],
            fill=stroke_color + (200,),
        )
        draw.text((label_x + 2, label_y + 1), label, fill=(255, 255, 255, 255), font=font)

    return overlay


def _render_header():
    st.markdown(
        """
        <div class="blade-header">
            <div style="background:#2563eb; color:white; padding:0.6rem; border-radius:12px;">🌀</div>
            <div>
                <div style="font-size:1.4rem; font-weight:800; color:#0f172a;">BladeGuard AI</div>
                <div class="muted">Automated Turbine Inspection System</div>
            </div>
            <div class="pill" style="background:#ecfdf3; color:#16a34a; margin-left:auto; border:1px solid #bbf7d0;">
                ✅ System Operational
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_detection_capabilities():
    st.markdown("### Detection Capabilities")
    st.write("The model can localize and classify the following:")
    for item in DETECTION_CAPABILITIES:
        st.markdown(f"- {item}")


def _render_ready_state():
    st.markdown(
        """
        <div class="card" style="text-align:center;">
            <div style="font-size:2.8rem;">📷</div>
            <h3 style="margin:0.3rem 0;">Ready for Inspection</h3>
            <p class="muted" style="max-width:480px; margin:0 auto;">
                Upload an image of a wind turbine blade to begin automated structural analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_summary(result: Dict[str, Any]):
    score = float(result.get("bladeConditionScore", 0))
    severity = "green" if score >= 80 else "amber" if score >= 50 else "red"
    score_color = {"green": "#16a34a", "amber": "#d97706", "red": "#dc2626"}[severity]

    with st.container():
        st.markdown("#### Executive Summary")
        cols = st.columns([2, 5, 3])
        with cols[0]:
            st.markdown(
                f"""
                <div class="card" style="text-align:center;">
                    <div class="label">Health Score</div>
                    <div style="font-size:2.6rem; font-weight:800; color:{score_color}; line-height:1;">{score:.0f}</div>
                    <div class="muted">/ 100</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.markdown(
                f"""
                <div class="card" style="height:100%;">
                    <div class="label">Summary</div>
                    <p style="margin-top:0.25rem; color:#0f172a; font-weight:600;">{result.get("summary", "")}</p>
                    <div class="muted" style="margin-top:0.5rem;">{result.get("timestamp")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.markdown(
                """
                <div class="card">
                    <div class="label">Diagnostics</div>
                    <ul style="padding-left:1rem; color:#334155;">
                        <li>Confidence-weighted bounding boxes</li>
                        <li>Actionable maintenance guidance</li>
                        <li>Scores reflect defect impact</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_defects(defects: List[Dict[str, Any]]):
    st.markdown("#### Detailed Findings")
    if not defects:
        st.success("No defects detected. The blade appears to be in excellent structural condition.")
        return

    for defect in defects:
        severity = str(defect.get("severity", "Low")).title()
        color = _get_severity_color(severity)
        confidence = float(defect.get("confidence", 0))

        st.markdown(
            f"""
            <div class="card" style="margin-bottom:0.75rem;">
                <div style="display:flex; gap:0.6rem; align-items:flex-start;">
                    <div class="pill" style="background:{color['bg']}; color:{color['text']}; border:1px solid rgba(0,0,0,0.05);">{severity}</div>
                    <div>
                        <div style="font-weight:700; color:#0f172a; font-size:1.05rem;">{defect.get("type", "Defect")}</div>
                        <div class="muted">Location: {defect.get("location", "Unknown")}</div>
                    </div>
                    <div style="margin-left:auto;" class="muted">Confidence: {confidence:.0f}%</div>
                </div>
                <p style="margin:0.8rem 0; color:#1f2937;">{defect.get("description", "")}</p>
                <div style="font-weight:700; color:#0f172a;">Recommendation</div>
                <p style="margin:0.25rem 0 0; color:#1f2937;">{defect.get("recommendation", "")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(max(confidence / 100, 0), 1))


def _render_image_panel(image_bytes: bytes, defects: List[Dict[str, Any]]):
    annotated = draw_annotations(image_bytes, defects)
    st.markdown("#### Annotated Image")
    st.image(annotated, use_column_width=True, caption="Detected defects with precise bounding boxes (0-1000 normalized grid).")


def main():
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "image_bytes" not in st.session_state:
        st.session_state.image_bytes = None
    if "mime_type" not in st.session_state:
        st.session_state.mime_type = None

    _render_header()
    st.write("")

    left, right = st.columns([5, 7], gap="large")

    with left:
        st.markdown("### Image Input")
        uploaded = st.file_uploader(
            "Upload a blade image (JPG, PNG, WEBP up to 10MB)",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=False,
        )

        if uploaded:
            image_bytes = uploaded.getvalue()
            mime_type = uploaded.type or "image/jpeg"
            st.image(image_bytes, caption="Preview", use_column_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                run_clicked = st.button("Run AI Analysis", type="primary")
            with col_b:
                clear_clicked = st.button("Clear Selection")

            if clear_clicked:
                st.session_state.analysis_result = None
                st.session_state.image_bytes = None
                st.session_state.mime_type = None
                st.experimental_rerun()

            if run_clicked:
                try:
                    with st.spinner("Analyzing blade surface..."):
                        result = run_gemini_analysis(image_bytes, mime_type)
                    st.session_state.analysis_result = result
                    st.session_state.image_bytes = image_bytes
                    st.session_state.mime_type = mime_type
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Analysis failed: {exc}")

        else:
            _render_ready_state()

        st.divider()
        _render_detection_capabilities()

    with right:
        result = st.session_state.analysis_result
        if result and st.session_state.image_bytes:
            _render_summary(result)
            _render_image_panel(st.session_state.image_bytes, result.get("defects", []))
            _render_defects(result.get("defects", []))
        else:
            _render_ready_state()


if __name__ == "__main__":
    main()
