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
    page_title="ClinicGuard-ReportGen",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main {
    background-color: #0f172a;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    font-weight: 700;
}

.stAlert {
    border-radius: 12px;
}

div[data-testid="stDataFrame"] {
    border-radius: 12px;
}

h1, h2, h3 {
    letter-spacing: -0.02em;
}
</style>
""", unsafe_allow_html=True)


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


st.markdown("""
<style>
:root {
    --cg-blue: #0D47A1;
    --cg-cyan: #00BCD4;
    --cg-teal: #14B8A6;
    --cg-green: #2E7D32;
    --cg-red: #C62828;
    --cg-amber: #F57C00;
    --cg-bg: #0F1419;
    --cg-elevated: #1A1E26;
    --cg-card: #232A35;
    --cg-card-2: #1B2330;
    --cg-border: #3F4B5C;
    --cg-text: #E8EEF5;
    --cg-muted: #9CA3AF;
    --cg-dim: #6B7280;
    --cg-shadow: 0 18px 48px rgba(0, 0, 0, 0.28);
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 15% 0%, rgba(13, 71, 161, 0.22), transparent 34rem),
        radial-gradient(circle at 92% 8%, rgba(20, 184, 166, 0.14), transparent 30rem),
        var(--cg-bg) !important;
    color: var(--cg-text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", sans-serif;
}

.block-container {
    max-width: 1480px;
    padding-top: 1.5rem;
    padding-bottom: 2.5rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #151C26 0%, #111821 100%);
    border-right: 1px solid rgba(63, 75, 92, 0.9);
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label {
    color: var(--cg-text) !important;
}

[data-testid="stSidebar"] button:not(:disabled),
[data-testid="stSidebar"] button:not(:disabled) *,
[data-testid="stSidebar"] [data-testid^="stBaseButton"]:not(:disabled),
[data-testid="stSidebar"] [data-testid^="stBaseButton"]:not(:disabled) * {
    color: var(--cg-text) !important;
    -webkit-text-fill-color: var(--cg-text) !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] button:not(:disabled) svg,
[data-testid="stSidebar"] button:not(:disabled) svg * {
    color: var(--cg-cyan) !important;
    fill: var(--cg-cyan) !important;
    stroke: var(--cg-cyan) !important;
}

.cg-hero {
    position: relative;
    overflow: hidden;
    padding: 2.25rem 2.35rem;
    border-radius: 14px;
    border: 1px solid rgba(0, 188, 212, 0.22);
    background:
        linear-gradient(135deg, rgba(13, 71, 161, 0.92), rgba(26, 30, 38, 0.96) 56%, rgba(20, 184, 166, 0.22)),
        linear-gradient(180deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.01));
    box-shadow: var(--cg-shadow);
    animation: cgSlideDown 0.35s ease-out both;
}

.cg-hero::after {
    content: "";
    position: absolute;
    inset: auto 0 0 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 188, 212, 0.65), transparent);
}

.cg-eyebrow {
    margin: 0 0 0.65rem 0;
    color: rgba(232, 238, 245, 0.74);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.cg-hero h1 {
    margin: 0;
    color: #FFFFFF;
    font-size: clamp(2.1rem, 4vw, 3.25rem);
    font-weight: 800;
    letter-spacing: -0.02em;
}

.cg-hero-subtitle {
    max-width: 760px;
    margin: 0.75rem 0 1.4rem 0;
    color: rgba(232, 238, 245, 0.8);
    font-size: 1.02rem;
    line-height: 1.65;
}

.cg-hero-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    align-items: center;
}

.cg-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.48rem 0.74rem;
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.14);
    background: rgba(255, 255, 255, 0.07);
    color: var(--cg-text);
    font-size: 0.82rem;
    font-weight: 700;
}

.cg-pill-alert {
    border-color: rgba(198, 40, 40, 0.48);
    background: rgba(198, 40, 40, 0.16);
}

.cg-workflow {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
    margin: 1rem 0 1.4rem 0;
}

.cg-step {
    padding: 0.85rem 0.95rem;
    border-radius: 10px;
    border: 1px solid rgba(63, 75, 92, 0.75);
    background: rgba(35, 42, 53, 0.72);
}

.cg-step strong {
    display: block;
    color: var(--cg-cyan);
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.cg-step span {
    display: block;
    margin-top: 0.28rem;
    color: var(--cg-muted);
    font-size: 0.86rem;
}

.cg-card,
.section-card,
.metric-card,
.sidebar-info,
.report-box {
    border-radius: 12px !important;
    border: 1px solid rgba(63, 75, 92, 0.78) !important;
    background:
        linear-gradient(145deg, rgba(35, 42, 53, 0.96), rgba(26, 30, 38, 0.96)) !important;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18) !important;
}

.section-card {
    padding: 1.15rem 1.25rem !important;
    margin-bottom: 1rem !important;
}

.section-card h3 {
    color: var(--cg-text) !important;
    font-size: 1rem !important;
}

.cg-file-meta,
.cg-mode-card,
.cg-ready-card,
.cg-empty-card {
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid rgba(63, 75, 92, 0.78);
    background: rgba(35, 42, 53, 0.72);
}

.cg-file-meta {
    margin-top: 0.75rem;
    color: var(--cg-muted);
    font-size: 0.88rem;
}

.cg-file-meta strong {
    color: var(--cg-text);
}

.cg-mode-card {
    margin-bottom: 1rem;
    border-left: 4px solid var(--cg-cyan);
}

.cg-mode-demo {
    border-left-color: var(--cg-red);
}

.cg-mode-live {
    border-left-color: var(--cg-green);
}

.cg-mode-card strong {
    display: block;
    color: var(--cg-text);
    font-size: 0.95rem;
    margin-bottom: 0.2rem;
}

.cg-mode-card span {
    color: var(--cg-muted);
    font-size: 0.88rem;
}

.cg-stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
    gap: 0.75rem;
    margin: 0.85rem 0 0.35rem 0;
}

.cg-stat-card {
    padding: 0.95rem;
    border-radius: 12px;
    border: 1px solid rgba(63, 75, 92, 0.78);
    background: linear-gradient(145deg, rgba(35, 42, 53, 0.98), rgba(24, 32, 43, 0.98));
}

.cg-stat-label {
    color: var(--cg-muted);
    font-size: 0.73rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.cg-stat-value {
    margin-top: 0.35rem;
    color: var(--cg-cyan);
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: 0;
}

.cg-stat-note {
    color: var(--cg-dim);
    font-size: 0.78rem;
    margin-top: 0.15rem;
}

.report-box {
    background: #111821 !important;
    color: var(--cg-text) !important;
    font-size: 1rem !important;
    line-height: 1.75 !important;
    max-height: 520px;
    overflow-y: auto;
    padding: 1.25rem !important;
    border-left: 4px solid var(--cg-cyan) !important;
}

.report-box .section-header {
    color: var(--cg-cyan) !important;
}

.cg-dropzone {
    border: 2px dashed rgba(0, 188, 212, 0.35);
    border-radius: 12px;
    padding: 2.4rem 1.25rem;
    text-align: center;
    background: rgba(0, 188, 212, 0.04);
}

.cg-dropzone p:first-child {
    color: var(--cg-text);
    font-size: 1rem;
    font-weight: 700;
}

.cg-dropzone p:last-child {
    color: var(--cg-muted);
    font-size: 0.85rem;
}

.finding-positive,
.finding-normal {
    background: linear-gradient(135deg, var(--cg-green), #43A047) !important;
}

.finding-uncertain {
    background: linear-gradient(135deg, var(--cg-amber), #FF9800) !important;
}

.finding-refused {
    background: linear-gradient(135deg, var(--cg-red), #E53935);
}

.confidence-bar-bg {
    background: rgba(107, 114, 128, 0.25) !important;
    height: 8px !important;
}

.confidence-bar-fill {
    transition: width 0.65s ease-out !important;
}

.gradient-divider {
    background: linear-gradient(90deg, transparent, rgba(0, 188, 212, 0.28), transparent) !important;
    margin: 1.25rem 0 !important;
}

button {
    border-radius: 10px !important;
    font-weight: 700 !important;
    min-height: 46px !important;
    padding: 0.72rem 1.1rem !important;
    color: var(--cg-text) !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease !important;
}

button:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 8px 22px rgba(0, 188, 212, 0.16);
}

.stButton > button,
div[data-testid="stButton"] button,
[data-testid="stBaseButton-primary"] {
    width: 100% !important;
    border: 0 !important;
    background: linear-gradient(135deg, #00BCD4 0%, #00ACC1 100%) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 15px rgba(0, 188, 212, 0.4) !important;
    font-size: 1rem !important;
}

.stButton > button *,
div[data-testid="stButton"] button *,
[data-testid="stBaseButton-primary"] * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

.stButton > button:hover:not(:disabled),
div[data-testid="stButton"] button:hover:not(:disabled),
[data-testid="stBaseButton-primary"]:hover:not(:disabled) {
    background: linear-gradient(135deg, #00ACC1 0%, #0097A7 100%) !important;
    box-shadow: 0 8px 24px rgba(0, 188, 212, 0.55) !important;
}

.stButton > button:disabled,
div[data-testid="stButton"] button:disabled,
[data-testid="stBaseButton-primary"]:disabled {
    opacity: 0.55 !important;
    color: rgba(255, 255, 255, 0.82) !important;
    cursor: not-allowed !important;
}

.stDownloadButton > button,
div[data-testid="stDownloadButton"] button {
    width: 100% !important;
    border: 2px solid var(--cg-cyan) !important;
    background: rgba(0, 188, 212, 0.12) !important;
    color: var(--cg-cyan) !important;
    box-shadow: none !important;
}

.stDownloadButton > button *,
div[data-testid="stDownloadButton"] button * {
    color: var(--cg-cyan) !important;
    fill: var(--cg-cyan) !important;
    -webkit-text-fill-color: var(--cg-cyan) !important;
}

.stDownloadButton > button:hover:not(:disabled),
div[data-testid="stDownloadButton"] button:hover:not(:disabled) {
    background: var(--cg-cyan) !important;
    color: #FFFFFF !important;
    box-shadow: 0 6px 18px rgba(0, 188, 212, 0.34) !important;
}

.stDownloadButton > button:hover:not(:disabled) *,
div[data-testid="stDownloadButton"] button:hover:not(:disabled) * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

button:focus-visible,
a:focus-visible,
input:focus-visible,
textarea:focus-visible {
    outline: 3px solid #0EA5E9 !important;
    outline-offset: 2px !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(63, 75, 92, 0.78);
    border-radius: 12px;
    min-height: 260px;
    overflow: hidden;
}

[data-testid="stImage"] img {
    max-height: 360px;
    object-fit: contain;
    border-radius: 12px;
    border: 1px solid rgba(63, 75, 92, 0.9);
    background: #0B1118;
}

[data-testid="stHorizontalBlock"] > [data-testid="column"] {
    min-width: 0;
}

@media (min-width: 1024px) {
    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        max-height: calc(100vh - 245px);
        overflow-y: auto;
        padding-right: 0.35rem;
    }
}

[data-testid="stMetricValue"] {
    color: var(--cg-cyan);
}

[data-testid="stExpander"] {
    border-color: rgba(63, 75, 92, 0.78) !important;
    border-radius: 12px !important;
    background: rgba(35, 42, 53, 0.45);
}

[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: rgba(35, 42, 53, 0.82) !important;
    border: 1px solid rgba(63, 75, 92, 0.92) !important;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.16) !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpander"] summary *,
[data-testid="stSidebar"] [data-testid="stExpander"] button,
[data-testid="stSidebar"] [data-testid="stExpander"] button *,
[data-testid="stSidebar"] [data-testid="stExpander"] [role="button"],
[data-testid="stSidebar"] [data-testid="stExpander"] [role="button"] * {
    color: var(--cg-text) !important;
    fill: var(--cg-cyan) !important;
    stroke: var(--cg-cyan) !important;
    -webkit-text-fill-color: var(--cg-text) !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover,
[data-testid="stSidebar"] [data-testid="stExpander"] button:hover,
[data-testid="stSidebar"] [data-testid="stExpander"] [role="button"]:hover {
    background: rgba(0, 188, 212, 0.10) !important;
    border-color: rgba(0, 188, 212, 0.42) !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] p,
[data-testid="stSidebar"] [data-testid="stExpander"] label,
[data-testid="stSidebar"] [data-testid="stExpander"] span {
    color: var(--cg-text) !important;
    -webkit-text-fill-color: var(--cg-text) !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] [data-testid="stExpander"] svg,
[data-testid="stSidebar"] [data-testid="stExpander"] svg * {
    color: var(--cg-cyan) !important;
    fill: var(--cg-cyan) !important;
    stroke: var(--cg-cyan) !important;
    opacity: 1 !important;
}

@keyframes cgSlideDown {
    from { opacity: 0; transform: translateY(-12px); }
    to { opacity: 1; transform: translateY(0); }
}

@media (max-width: 900px) {
    .cg-workflow,
    .cg-stat-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 640px) {
    .cg-hero {
        padding: 1.35rem;
    }
    .cg-workflow,
    .cg-stat-grid {
        grid-template-columns: 1fr;
    }
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

@media (prefers-reduced-motion: reduce) {
    * {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
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
        checkpoint_path = os.path.join(config.training.checkpoint_dir, "best_model.pt")

        if not os.path.exists(checkpoint_path):
            st.sidebar.warning("No fine-tuned checkpoint found. Demo Mode will be used.")
            return None, None, None, None, config
        
        # Load vision encoder
        encoder = GroundedVisionEncoder(config=config.vision)
        encoder.to(device)
        encoder.eval()
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=False)
        st.sidebar.success("✅ Loaded fine-tuned model")
        
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
        color = "linear-gradient(90deg, #2E7D32, #43A047)"
        status = "HIGH"
    elif confidence >= 0.50:
        color = "linear-gradient(90deg, #F57C00, #FF9800)"
        status = "MODERATE"
    else:
        color = "linear-gradient(90deg, #C62828, #E53935)"
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


def render_hero() -> None:
    """Render the premium product hero and workflow summary."""
    st.markdown("""
    <section class="cg-hero">
        <p class="cg-eyebrow">Evidence-grounded medical AI prototype</p>
        <h1>ClinicGuard-ReportGen</h1>
        <p class="cg-hero-subtitle">
            Radiology-style report generation with claim-level evidence, controlled refusal,
            and hallucination-aware audit trails.
        </p>
        <div class="cg-hero-row">
            <span class="cg-pill">Status: ready for analysis</span>
            <span class="cg-pill cg-pill-alert">Research prototype - not for clinical diagnosis</span>
        </div>
    </section>
    <div class="cg-workflow">
        <div class="cg-step"><strong>01 Upload</strong><span>Chest X-ray image</span></div>
        <div class="cg-step"><strong>02 Context</strong><span>Optional history or prior</span></div>
        <div class="cg-step"><strong>03 Generate</strong><span>Constrained report text</span></div>
        <div class="cg-step"><strong>04 Audit</strong><span>Evidence log and refusals</span></div>
    </div>
    """, unsafe_allow_html=True)


def render_mode_card(is_demo_mode: bool) -> None:
    """Render live/demo mode status."""
    if is_demo_mode:
        st.markdown("""
        <div class="cg-mode-card cg-mode-demo">
            <strong>Demo Mode</strong>
            <span>Simulated outputs are shown because real model inference failed or the pipeline is unavailable.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="cg-mode-card cg-mode-live">
            <strong>Live Model Mode</strong>
            <span>Outputs were generated using the loaded model and report-generation pipeline.</span>
        </div>
        """, unsafe_allow_html=True)


def render_stat_grid(total: int, supported: int, hallucinated: int, refused: int, avg_confidence: float) -> None:
    """Render premium claim audit KPI cards."""
    hallucination_rate = hallucinated / total if total else 0.0
    st.markdown(f"""
    <div class="cg-stat-grid">
        <div class="cg-stat-card">
            <div class="cg-stat-label">Total claims</div>
            <div class="cg-stat-value">{total}</div>
            <div class="cg-stat-note">Generated claim rows</div>
        </div>
        <div class="cg-stat-card">
            <div class="cg-stat-label">Supported</div>
            <div class="cg-stat-value">{supported}</div>
            <div class="cg-stat-note">Grounded or source-backed</div>
        </div>
        <div class="cg-stat-card">
            <div class="cg-stat-label">Refused</div>
            <div class="cg-stat-value">{refused}</div>
            <div class="cg-stat-note">Below threshold</div>
        </div>
        <div class="cg-stat-card">
            <div class="cg-stat-label">Hallucinated</div>
            <div class="cg-stat-value">{hallucinated}</div>
            <div class="cg-stat-note">Unsupported generated claims</div>
        </div>
        <div class="cg-stat-card">
            <div class="cg-stat-label">Hallucination rate</div>
            <div class="cg-stat-value">{hallucination_rate:.0%}</div>
            <div class="cg-stat-note">Avg confidence {avg_confidence:.0%}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


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
    render_hero()
    
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
        
        with st.expander("Advanced Settings"):
            positive_threshold = st.slider(
                "Assert finding if confidence >=",
                min_value=0.0,
                max_value=1.0,
                value=0.75,
                step=0.05,
                help="Minimum confidence to assert a finding is present",
            )
            uncertain_threshold = st.slider(
                "Mark uncertain if confidence >=",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                step=0.05,
                help="Below positive but above this threshold uses hedged language",
            )
        
        st.markdown("### Case Inputs")
        with st.expander("Patient Context", expanded=True):
            patient_age = st.number_input(
                "Age (optional)",
                min_value=0,
                max_value=120,
                value=0,
                step=1,
                help="Leave at 0 if age is unknown or not relevant.",
            )
            patient_sex = st.selectbox(
                "Sex",
                ["Not specified", "Female", "Male", "Other / not listed"],
            )
            smoking_status = st.selectbox(
                "Smoking status",
                ["Not specified", "Never", "Former", "Current", "Unknown"],
            )
            clinical_notes = st.text_area(
                "Clinical Notes",
                placeholder="e.g., shortness of breath, follow-up pleural effusion...",
                height=100,
            )

        with st.expander("Prior Report", expanded=False):
            prior_report = st.text_area(
                "Prior Report",
                placeholder="Paste previous radiology report for comparison...",
                height=120,
            )

        patient_history_parts = []
        if patient_age:
            patient_history_parts.append(f"Age: {patient_age}")
        if patient_sex != "Not specified":
            patient_history_parts.append(f"Sex: {patient_sex}")
        if smoking_status != "Not specified":
            patient_history_parts.append(f"Smoking status: {smoking_status}")
        if clinical_notes.strip():
            patient_history_parts.append(clinical_notes.strip())
        patient_history = " | ".join(patient_history_parts) if patient_history_parts else None
        
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
    image = None
    if "run_analysis" not in st.session_state:
        st.session_state.run_analysis = False

    col_left, col_right = st.columns([0.35, 0.65], gap="large")
    
    with col_left:
        st.markdown("""
        <div class="section-card">
            <h3>📤 Upload Chest X-Ray</h3>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Upload a chest X-ray image (PNG or JPG)",
            type=["png", "jpg", "jpeg"],
            help="Supported: Frontal (PA/AP) chest radiographs in PNG/JPG format",
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Chest X-Ray", width="stretch")
            
            # Image info
            st.markdown(f"""
            <div class="cg-file-meta">
                <strong>File ready</strong><br>
                Size: {image.size[0]} x {image.size[1]} px | Format: {uploaded_file.type} |
                Weight: {uploaded_file.size / 1024:.1f} KB
            </div>
            """, unsafe_allow_html=True)
        else:
            st.session_state.run_analysis = False
            st.markdown("""
            <div class="cg-dropzone">
                <p style="color: rgba(255,255,255,0.4); font-size: 1.1rem;">📷 Drag & drop a chest X-ray here</p>
                <p style="color: rgba(255,255,255,0.25); font-size: 0.85rem;">Supported: PNG, JPG, JPEG</p>
            </div>
            """, unsafe_allow_html=True)

        image_status = "Ready" if uploaded_file is not None else "Waiting for upload"
        context_status = "Provided" if patient_history else "Not provided"
        prior_status = "Provided" if prior_report.strip() else "Not provided"
        st.markdown(f"""
        <div class="section-card">
            <h3>Case status</h3>
            <p><strong>Image:</strong> {image_status}</p>
            <p><strong>Patient context:</strong> {context_status}</p>
            <p><strong>Prior report:</strong> {prior_status}</p>
        </div>
        """, unsafe_allow_html=True)

        generate_clicked = st.button(
            "Generate report",
            type="primary",
            width="stretch",
            disabled=uploaded_file is None,
        )
        if generate_clicked:
            st.session_state.run_analysis = True
    
    with col_right:
        if uploaded_file is not None and st.session_state.run_analysis:
            # Try loading real model, fall back to demo
            is_demo_mode = False
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
                        is_demo_mode = True
                        st.warning(f"⚠️ Real inference failed ({str(e)[:80]}...), showing demo results")
                        predictions, report_text, evidence = run_demo_mode(image)
                else:
                    is_demo_mode = True
                    predictions, report_text, evidence = run_demo_mode(image)
            
            # Findings and audit summaries
            positive_findings = {k: v for k, v in predictions.items() if v >= positive_threshold}
            uncertain_findings = {k: v for k, v in predictions.items() if uncertain_threshold <= v < positive_threshold}
            refused_claims = [
                {
                    "Finding": finding,
                    "Reason": f"Confidence {conf:.2f} below uncertainty threshold",
                    "Decision": "Not generated",
                }
                for finding, conf in predictions.items()
                if conf < uncertain_threshold
            ]
            total_claims = len(evidence)
            supported_claims = sum(1 for e in evidence if "Hallucinated" not in e.get("Status", ""))
            hallucinated_claims = sum(1 for e in evidence if "Hallucinated" in e.get("Status", ""))
            hallucination_rate = hallucinated_claims / total_claims if total_claims else 0.0
            avg_confidence = np.mean([e["Confidence"] for e in evidence]) if evidence else 0.0

            render_mode_card(is_demo_mode)

            # ---- Generated Report ----
            st.subheader("Generated report")
            formatted = format_report_html(report_text)
            st.markdown(f'<div class="report-box">{formatted}</div>', unsafe_allow_html=True)

            st.download_button(
                label="Download report (TXT)",
                data=report_text,
                file_name=f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

            # ---- Evidence Visualization ----
            st.subheader("Evidence visualization")
            if positive_findings:
                badges_html = ""
                for finding, conf in sorted(positive_findings.items(), key=lambda x: -x[1]):
                    badges_html += f'<span class="finding-positive">{finding} ({conf:.0%})</span> '
                st.markdown("**Model-supported findings:**")
                st.markdown(badges_html, unsafe_allow_html=True)

            if uncertain_findings:
                badges_html = ""
                for finding, conf in sorted(uncertain_findings.items(), key=lambda x: -x[1]):
                    badges_html += f'<span class="finding-uncertain">{finding} ({conf:.0%})</span> '
                st.markdown("**Low-confidence / uncertain findings:**")
                st.markdown(badges_html, unsafe_allow_html=True)

            if not positive_findings and not uncertain_findings:
                st.markdown(
                    '<span class="finding-normal">No high-confidence abnormality detected by this prototype.</span>',
                    unsafe_allow_html=True,
                )

            st.markdown("**Top model probabilities:**")
            top_findings = sorted(predictions.items(), key=lambda x: -x[1])[:8]
            bars_html = ""
            for label, conf in top_findings:
                bars_html += create_confidence_bar(conf, label)
            st.markdown(bars_html, unsafe_allow_html=True)

            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

            st.subheader("Claim-level evidence log")
            if evidence:
                evidence_df = pd.DataFrame(evidence)
                st.dataframe(
                    evidence_df,
                    width="stretch",
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

                csv = evidence_df.to_csv(index=False)
                st.download_button(
                    label="Download evidence log (CSV)",
                    data=csv,
                    file_name=f"evidence_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            else:
                st.info("No evidence rows were produced for this case.")

            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

            st.subheader("Claim audit metrics")
            render_stat_grid(
                total=total_claims,
                supported=supported_claims,
                hallucinated=hallucinated_claims,
                refused=len(refused_claims),
                avg_confidence=float(avg_confidence),
            )
            st.caption(
                "Hallucination rate = unsupported generated claims / total generated claims."
            )

            with st.expander(
                f"Refused / not generated claims ({len(refused_claims)})",
                expanded=bool(refused_claims),
            ):
                st.caption(
                    "Refusal is protective: claims below the uncertainty threshold are not generated as findings."
                )
                if refused_claims:
                    st.dataframe(pd.DataFrame(refused_claims), width="stretch", hide_index=True)
                else:
                    st.success("No refused claims in this case.")
        elif uploaded_file is not None:
            st.markdown("""
            <div class="section-card" style="text-align: center; padding: 4rem 2rem;">
                <h3>Ready to generate</h3>
                <p style="color: rgba(255,255,255,0.5);">
                    Add optional history or a prior report, then click Generate report. The report,
                    evidence log, metrics, and refused claims will stay in this right pane.
                </p>
            </div>
            """, unsafe_allow_html=True)
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

    with st.expander("System Architecture"):
        st.markdown("""
        **1. Vision Encoder:** DenseNet121 predicts pathology probabilities.

        **2. Grounding:** Grad-CAM/evidence references connect claims to visual support.

        **3. Generation:** Template-based constrained report generation.

        **4. Safety Layer:** Confidence thresholds decide assert / uncertain / refuse.

        **5. Audit:** Evidence log verifies each generated claim.
        """)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; color: rgba(255,255,255,0.2); font-size: 0.8rem;">
        ClinicGuard-ReportGen v1.0.0 | Evidence-Grounded Radiology Report Generation<br>
        ⚠️ This tool is for research/educational purposes only. Not for clinical diagnosis.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
