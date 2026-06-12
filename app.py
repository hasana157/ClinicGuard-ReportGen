"""
Medical Report Generation — Streamlit Web Application

A professional demo interface for uploading chest X-ray images and generating
grounded medical reports with visual evidence (Grad-CAM heatmaps).

Run with: streamlit run app.py
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import os
import sys
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Medical Report AI — Zero-Hallucination Radiology",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Custom CSS — Premium Dark Theme
# ============================================================================

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.05rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Section cards */
    .section-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.8rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .section-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }
    
    .section-card h3 {
        color: #a8edea;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        letter-spacing: -0.01em;
    }
    
    /* Metric cards */
    .metric-container {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #0a0a1a 0%, #1a1a3e 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        flex: 1;
        min-width: 150px;
        border: 1px solid rgba(168, 237, 234, 0.15);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: rgba(168, 237, 234, 0.4);
        box-shadow: 0 0 20px rgba(168, 237, 234, 0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.3rem;
    }
    
    /* Finding badges */
    .finding-positive {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
        box-shadow: 0 2px 8px rgba(238, 90, 36, 0.3);
    }
    
    .finding-uncertain {
        background: linear-gradient(135deg, #ffa726 0%, #ff7043 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
        box-shadow: 0 2px 8px rgba(255, 167, 38, 0.3);
    }
    
    .finding-normal {
        background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
        box-shadow: 0 2px 8px rgba(67, 160, 71, 0.3);
    }
    
    /* Report text area */
    .report-box {
        background: #0a0a1a;
        border: 1px solid rgba(168, 237, 234, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        line-height: 1.7;
        white-space: pre-wrap;
    }
    
    .report-box .section-header {
        color: #a8edea;
        font-weight: 700;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 1rem;
    }
    
    /* Evidence table */
    .evidence-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .evidence-table th {
        background: linear-gradient(135deg, #302b63, #24243e);
        color: #a8edea;
        padding: 0.8rem 1rem;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .evidence-table td {
        padding: 0.7rem 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        color: #ccc;
        font-size: 0.9rem;
    }
    
    /* Confidence bar */
    .confidence-bar-bg {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        height: 10px;
        overflow: hidden;
    }
    
    .confidence-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* Status badges */
    .status-grounded {
        color: #66bb6a;
        font-weight: 600;
    }
    
    .status-hallucinated {
        color: #ff6b6b;
        font-weight: 600;
    }
    
    /* Sidebar */
    .sidebar-info {
        background: rgba(168, 237, 234, 0.05);
        border: 1px solid rgba(168, 237, 234, 0.15);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .sidebar-info p {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.85rem;
        margin: 0.3rem 0;
    }
    
    /* Upload area */
    .stFileUploader > div {
        border-radius: 12px;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Divider */
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(168, 237, 234, 0.3), transparent);
        margin: 1.5rem 0;
        border: none;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_resource
def load_model():
    """Load the medical report generation pipeline."""
    try:
        from src.vision_encoder import GroundedVisionEncoder, preprocess_for_model
        from src.grounding_module import GradCAMGrounding
        from src.report_generator import ConstrainedReportGenerator
        from src.config import get_config

        config = get_config()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load vision encoder
        encoder = GroundedVisionEncoder(config=config.vision)
        encoder.to(device)
        encoder.eval()
        
        # Load checkpoint if available
        checkpoint_path = os.path.join(config.training.checkpoint_dir, "best_model.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            encoder.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
            st.sidebar.success("✅ Loaded fine-tuned model")
        else:
            st.sidebar.info("ℹ️ Using pre-trained weights (no fine-tuning)")
        
        # Create grounding module
        grounding = GradCAMGrounding(model=encoder, config=config.gradcam)
        
        # Create report generator
        generator = ConstrainedReportGenerator(
            encoder=encoder,
            grounding_module=grounding,
            config=config.report,
        )
        
        return encoder, grounding, generator, device, config
    except Exception as e:
        st.error(f"⚠️ Model loading failed: {str(e)}")
        return None, None, None, None, None


def create_confidence_bar(confidence: float, label: str) -> str:
    """Create an HTML confidence bar."""
    if confidence >= 0.75:
        color = "linear-gradient(90deg, #ff6b6b, #ee5a24)"
        status = "HIGH"
    elif confidence >= 0.50:
        color = "linear-gradient(90deg, #ffa726, #ff7043)"
        status = "MODERATE"
    else:
        color = "linear-gradient(90deg, #66bb6a, #43a047)"
        status = "LOW"
    
    width = int(confidence * 100)
    return f"""
    <div style="margin: 0.5rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="color: #ccc; font-size: 0.85rem; font-weight: 500;">{label}</span>
            <span style="color: #a8edea; font-size: 0.85rem; font-weight: 600;">{confidence:.1%} ({status})</span>
        </div>
        <div class="confidence-bar-bg">
            <div class="confidence-bar-fill" style="width: {width}%; background: {color};"></div>
        </div>
    </div>
    """


def format_report_html(report_text: str) -> str:
    """Format report text with styled section headers."""
    sections = report_text.split("\n\n")
    html_parts = []
    
    for section in sections:
        lines = section.strip().split("\n")
        if not lines:
            continue
        
        # Check if first line is a section header
        first_line = lines[0].strip().upper()
        if any(header in first_line for header in ["INDICATION", "COMPARISON", "FINDINGS", "IMPRESSION"]):
            header_html = f'<div class="section-header">{lines[0].strip()}</div>'
            body_html = "\n".join(lines[1:]) if len(lines) > 1 else ""
            html_parts.append(f"{header_html}\n{body_html}")
        else:
            html_parts.append(section)
    
    return "\n\n".join(html_parts)


def run_demo_mode(uploaded_image: Image.Image):
    """Run in demo mode when models can't be loaded — show the UI with simulated results."""
    
    # Simulated predictions for demo
    demo_predictions = {
        "Cardiomegaly": 0.87,
        "Effusion": 0.72,
        "Atelectasis": 0.45,
        "Infiltration": 0.38,
        "Pneumonia": 0.22,
        "Consolidation": 0.15,
        "Edema": 0.12,
        "Pneumothorax": 0.08,
        "Emphysema": 0.06,
        "Fibrosis": 0.05,
        "Hernia": 0.03,
        "Mass": 0.04,
        "Nodule": 0.07,
        "Pleural_Thickening": 0.09,
    }
    
    demo_report = """FINDINGS:
Cardiomegaly is identified with the cardiac silhouette measuring above normal limits. The cardiothoracic ratio is increased.

The costophrenic angles show blunting bilaterally, which may represent small bilateral pleural effusions. Clinical correlation is recommended.

The lungs are otherwise clear without focal consolidation, pneumothorax, or mass lesion. The mediastinal contours are within normal limits. No acute osseous abnormality is identified.

IMPRESSION:
1. Cardiomegaly.
2. Possible small bilateral pleural effusions — recommend clinical correlation.
3. No acute cardiopulmonary process otherwise identified."""

    demo_evidence = [
        {"Claim": "Cardiomegaly is identified", "Source": "Visual", "Reference": "GradCAM region [45,60,180,200]", "Confidence": 0.87, "Status": "✅ Grounded"},
        {"Claim": "Pleural effusions", "Source": "Visual", "Reference": "GradCAM region [20,150,100,220]", "Confidence": 0.72, "Status": "⚠️ Uncertain"},
        {"Claim": "Lungs are otherwise clear", "Source": "Visual", "Reference": "Global assessment", "Confidence": 0.91, "Status": "✅ Grounded"},
        {"Claim": "No pneumothorax", "Source": "Visual", "Reference": "GradCAM absence", "Confidence": 0.92, "Status": "✅ Grounded"},
        {"Claim": "No mass lesion", "Source": "Visual", "Reference": "GradCAM absence", "Confidence": 0.96, "Status": "✅ Grounded"},
    ]
    
    return demo_predictions, demo_report, demo_evidence


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main Streamlit application."""
    
    # ---- Header ----
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Medical Report AI</h1>
        <p>Zero-hallucination radiology report generation with grounded visual evidence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ---- Sidebar ----
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        st.markdown("""
        <div class="sidebar-info">
            <p><strong>Model:</strong> DenseNet121 (torchxrayvision)</p>
            <p><strong>Pre-trained:</strong> MIMIC-CXR / CheXpert</p>
            <p><strong>Grounding:</strong> Grad-CAM</p>
            <p><strong>Generation:</strong> Template-based constrained</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Thresholds")
        positive_threshold = st.slider(
            "Positive finding threshold",
            min_value=0.0, max_value=1.0, value=0.75, step=0.05,
            help="Minimum confidence to assert a finding is present"
        )
        uncertain_threshold = st.slider(
            "Uncertain finding threshold",
            min_value=0.0, max_value=1.0, value=0.50, step=0.05,
            help="Below positive but above this → hedged language"
        )
        
        st.markdown("### Optional Inputs")
        patient_history = st.text_area(
            "Patient History",
            placeholder="e.g., 65-year-old male with shortness of breath...",
            height=100,
        )
        prior_report = st.text_area(
            "Prior Report",
            placeholder="Paste previous radiology report for comparison...",
            height=100,
        )
        
        st.markdown("---")
        st.markdown("""
        <div class="sidebar-info">
            <p>🔬 <strong>Architecture</strong></p>
            <p>• DenseNet121 backbone (7.5M params)</p>
            <p>• Grad-CAM visual grounding</p>
            <p>• Constrained template generation</p>
            <p>• Hallucination detection & refusal</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"<p style='color: rgba(255,255,255,0.3); font-size: 0.75rem; text-align: center;'>v1.0.0 | {datetime.now().strftime('%Y-%m-%d')}</p>", unsafe_allow_html=True)
    
    # ---- Main Content ----
    col_upload, col_results = st.columns([1, 1.5])
    
    with col_upload:
        st.markdown("""
        <div class="section-card">
            <h3>📤 Upload Chest X-Ray</h3>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload a chest X-ray image (PNG, JPG, DICOM)",
            type=["png", "jpg", "jpeg", "dcm"],
            help="Supported: Frontal (PA/AP) chest radiographs",
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Chest X-Ray", use_container_width=True)
            
            # Image info
            st.markdown(f"""
            <div style="background: rgba(168,237,234,0.05); padding: 0.8rem; border-radius: 8px; margin-top: 0.5rem;">
                <p style="color: #a8edea; font-size: 0.85rem; margin: 0;">
                    📐 Size: {image.size[0]}×{image.size[1]} | 
                    📁 Format: {uploaded_file.type} | 
                    💾 {uploaded_file.size / 1024:.1f} KB
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="border: 2px dashed rgba(168,237,234,0.2); border-radius: 12px; padding: 3rem; text-align: center;">
                <p style="color: rgba(255,255,255,0.4); font-size: 1.1rem;">📷 Drag & drop a chest X-ray here</p>
                <p style="color: rgba(255,255,255,0.25); font-size: 0.85rem;">Supported: PNG, JPG, JPEG</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col_results:
        if uploaded_file is not None:
            # Try loading real model, fall back to demo
            with st.spinner("🔄 Analyzing chest X-ray..."):
                model_result = load_model()
                
                if model_result[0] is not None:
                    encoder, grounding, generator, device, config = model_result
                    try:
                        from src.vision_encoder import preprocess_for_model
                        
                        # Run real inference
                        image_tensor = preprocess_for_model(image).to(device)
                        
                        with torch.no_grad():
                            output = encoder(image_tensor)
                        
                        predictions = {}
                        from src.config import PATHOLOGY_LABELS
                        probs = output["probabilities"][0].cpu().numpy()
                        print(f"DEBUG: Probabilities - min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
                        print(f"DEBUG: Probs array: {probs}")
                        for i, label in enumerate(PATHOLOGY_LABELS):
                            predictions[label] = float(probs[i])
                        
                        # Generate report
                        report_output = generator.generate(
                            image=image,
                            patient_history=patient_history if patient_history else None,
                            prior_report=prior_report if prior_report else None,
                        )
                        report_text = report_output.report_text
                        evidence = [
                            {
                                "Claim": e.get("generated_claim", ""),
                                "Source": e.get("source_type", ""),
                                "Reference": e.get("source_reference", ""),
                                "Confidence": e.get("confidence_score", 0),
                                "Status": "✅ Grounded" if not e.get("hallucinated", False) else "❌ Hallucinated",
                            }
                            for e in report_output.evidence_log
                        ]
                        
                    except Exception as e:
                        st.warning(f"⚠️ Real inference failed ({str(e)[:80]}...), showing demo results")
                        predictions, report_text, evidence = run_demo_mode(image)
                else:
                    predictions, report_text, evidence = run_demo_mode(image)
            
            # ---- Classification Results ----
            st.markdown("""
            <div class="section-card">
                <h3>🔍 Classification Results</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Findings badges
            positive_findings = {k: v for k, v in predictions.items() if v >= positive_threshold}
            uncertain_findings = {k: v for k, v in predictions.items() if uncertain_threshold <= v < positive_threshold}
            
            if positive_findings:
                badges_html = ""
                for finding, conf in sorted(positive_findings.items(), key=lambda x: -x[1]):
                    badges_html += f'<span class="finding-positive">{finding} ({conf:.0%})</span> '
                st.markdown(f"**Positive Findings:** {badges_html}", unsafe_allow_html=True)
            
            if uncertain_findings:
                badges_html = ""
                for finding, conf in sorted(uncertain_findings.items(), key=lambda x: -x[1]):
                    badges_html += f'<span class="finding-uncertain">{finding} ({conf:.0%})</span> '
                st.markdown(f"**Uncertain Findings:** {badges_html}", unsafe_allow_html=True)
            
            if not positive_findings and not uncertain_findings:
                st.markdown('<span class="finding-normal">Normal Study — No significant findings</span>', unsafe_allow_html=True)
            
            # Confidence bars
            st.markdown("<br>", unsafe_allow_html=True)
            top_findings = sorted(predictions.items(), key=lambda x: -x[1])[:8]
            bars_html = ""
            for label, conf in top_findings:
                bars_html += create_confidence_bar(conf, label)
            st.markdown(bars_html, unsafe_allow_html=True)
            
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            
            # ---- Generated Report ----
            st.markdown("""
            <div class="section-card">
                <h3>📋 Generated Medical Report</h3>
            </div>
            """, unsafe_allow_html=True)
            
            formatted = format_report_html(report_text)
            st.markdown(f'<div class="report-box">{formatted}</div>', unsafe_allow_html=True)
            
            # Download button
            st.download_button(
                label="📥 Download Report (TXT)",
                data=report_text,
                file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )
            
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            
            # ---- Metrics ----
            num_grounded = sum(1 for e in evidence if "Grounded" in e.get("Status", ""))
            total_claims = len(evidence) if evidence else 1
            hallucination_count = sum(1 for e in evidence if "Hallucinated" in e.get("Status", ""))
            
            grounding_rate = num_grounded / total_claims if total_claims > 0 else 1.0
            hallucination_rate = hallucination_count / total_claims if total_claims > 0 else 0.0
            avg_confidence = np.mean([e["Confidence"] for e in evidence]) if evidence else 0.0
            
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-card">
                    <div class="metric-value">{grounding_rate:.0%}</div>
                    <div class="metric-label">Grounding Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{hallucination_rate:.0%}</div>
                    <div class="metric-label">Hallucination Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_confidence:.0%}</div>
                    <div class="metric-label">Avg Confidence</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(positive_findings)}</div>
                    <div class="metric-label">Findings Detected</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            
            # ---- Evidence Log ----
            st.markdown("""
            <div class="section-card">
                <h3>🔗 Grounding Evidence Log</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if evidence:
                evidence_df = pd.DataFrame(evidence)
                st.dataframe(
                    evidence_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Confidence": st.column_config.ProgressColumn(
                            "Confidence",
                            min_value=0,
                            max_value=1,
                            format="%.2f",
                        ),
                    },
                )
                
                # Download evidence CSV
                csv = evidence_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Evidence Log (CSV)",
                    data=csv,
                    file_name=f"evidence_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
        else:
            # No image uploaded yet
            st.markdown("""
            <div class="section-card" style="text-align: center; padding: 4rem 2rem;">
                <h3>👈 Upload a Chest X-Ray to Begin</h3>
                <p style="color: rgba(255,255,255,0.5);">
                    The AI will analyze the image, classify pathologies,<br>
                    generate a grounded medical report, and provide evidence logs.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # ---- Architecture Overview (bottom section) ----
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-card">
        <h3>🏗️ System Architecture</h3>
    </div>
    """, unsafe_allow_html=True)
    
    arch_cols = st.columns(4)
    
    with arch_cols[0]:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔬</div>
            <div style="color: #a8edea; font-weight: 600; margin-bottom: 0.3rem;">Vision Encoder</div>
            <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem;">DenseNet121<br>Pre-trained on CheXpert/MIMIC<br>7.5M parameters</div>
        </div>
        """, unsafe_allow_html=True)
    
    with arch_cols[1]:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
            <div style="color: #a8edea; font-weight: 600; margin-bottom: 0.3rem;">Grad-CAM Grounding</div>
            <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem;">Attention-based<br>Region attribution<br>Per-finding heatmaps</div>
        </div>
        """, unsafe_allow_html=True)
    
    with arch_cols[2]:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📝</div>
            <div style="color: #a8edea; font-weight: 600; margin-bottom: 0.3rem;">Constrained Generation</div>
            <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem;">Template-based<br>Confidence-gated<br>Refusal mechanism</div>
        </div>
        """, unsafe_allow_html=True)
    
    with arch_cols[3]:
        st.markdown("""
        <div class="metric-card">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🛡️</div>
            <div style="color: #a8edea; font-weight: 600; margin-bottom: 0.3rem;">Hallucination Guard</div>
            <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem;">Factuality verification<br>Evidence cross-referencing<br>Zero-tolerance policy</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: rgba(255,255,255,0.2); font-size: 0.8rem;">
        Medical Report AI v1.0.0 | Zero-Hallucination Radiology Report Generation<br>
        ⚠️ This tool is for research/educational purposes only. Not for clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
