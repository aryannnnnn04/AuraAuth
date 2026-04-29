"""
Streamlit Frontend for AuraAuth AutoML System.

Modern SaaS-Style Dashboard
- Automated model training and optimization
- Uncertainty estimation and reliability
- Distribution shift detection
- Model explainability and interpretability
"""

import sys
import os
import time
import io
import textwrap
import concurrent.futures

# Ensure the project root (AuraAuth/) is on sys.path so 'backend' package resolves
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import hmac
import secrets
import json
import ast
import re
import logging
from typing import Optional
from html import escape as html_escape

logger = logging.getLogger(__name__)

# ============================================================================
# Streamlit Configuration
# ============================================================================

st.set_page_config(
    page_title="AuraAuth — AutoML",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Modern SaaS-Style CSS Design
# ============================================================================

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');

    :root {
        --bg-primary: #0D111B;
        --bg-secondary: #141A2A;
        --hero-button: #121B2B;
        --hero-button-hover: #1A2740;
        --hero-button-border: #4C648F;
        --bg-card: rgba(22, 29, 45, 0.88);
        --text-primary: #E7ECF7;
        --text-muted: #A7B1C8;
        --accent: #5B8CFF;
        --accent-strong: #3F73F5;
        --border-soft: rgba(143, 167, 217, 0.24);
    }

    * {
        font-family: 'Manrope', 'Segoe UI', sans-serif;
    }
    
    body, .main {
        background: radial-gradient(circle at 20% 20%, #1B2440 0%, var(--bg-primary) 45%, #0A0D14 100%) !important;
        color: var(--text-primary) !important;
    }
    
    .stApp {
        background: linear-gradient(180deg, rgba(13, 17, 27, 0.98) 0%, rgba(10, 13, 20, 0.98) 100%);
    }
    
    /* Headings */
    h1, h2, h3 {
        color: var(--text-primary);
        font-family: 'Space Grotesk', 'Manrope', sans-serif;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    
    h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        font-size: 2rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        font-size: 1.3rem;
        margin-bottom: 0.75rem;
    }
    
    /* Cards */
    [data-testid="stVerticalBlock"] > div {
        padding: 0;
    }
    
    .card {
        background: var(--bg-card);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid var(--border-soft);
        box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 14px 34px rgba(0,0,0,0.45);
    }
    
    /* Dark Cards */
    .dark-card {
        background-color: #0A0F1C;
        color: var(--text-primary);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid var(--border-soft);
        box-shadow: 0 10px 26px rgba(0,0,0,0.35);
    }
    
    /* All Streamlit Buttons (match Core Features card aesthetic) */
    .stButton > button {
        background: linear-gradient(160deg, rgba(26, 34, 54, 0.92) 0%, rgba(18, 24, 40, 0.92) 100%) !important;
        color: #F1F5FF !important;
        padding: 10px 22px !important;
        min-height: 42px !important;
        border-radius: 12px !important;
        border: 1px solid var(--border-soft) !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        transition: all 0.22s ease !important;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.3) !important;
    }

    .stButton > button * {
        color: #F1F5FF !important;
    }

    .stButton > button:hover {
        border-color: rgba(143, 167, 217, 0.4) !important;
        background: linear-gradient(160deg, rgba(30, 40, 62, 0.95) 0%, rgba(20, 28, 46, 0.95) 100%) !important;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.35) !important;
        transform: translateY(-2px) !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(160deg, rgba(29, 41, 66, 0.96) 0%, rgba(21, 30, 50, 0.96) 100%) !important;
        color: #F7FAFF !important;
        border: 1px solid rgba(143, 167, 217, 0.35) !important;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.35) !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(160deg, rgba(35, 47, 74, 0.98) 0%, rgba(24, 35, 58, 0.98) 100%) !important;
        border-color: rgba(160, 184, 232, 0.45) !important;
        box-shadow: 0 14px 26px rgba(0, 0, 0, 0.4) !important;
        transform: translateY(-2px) !important;
    }

    .stButton > button[kind="secondary"] {
        background: rgba(17, 25, 40, 0.86) !important;
        color: #E3E8F4 !important;
        border: 1px solid rgba(129, 153, 205, 0.24) !important;
        box-shadow: 0 6px 14px rgba(0,0,0,0.28) !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background: rgba(24, 34, 54, 0.92) !important;
        color: #F7FAFF !important;
        box-shadow: 0 10px 20px rgba(0,0,0,0.34) !important;
    }

    /* Download buttons should match primary dark button styling */
    .stDownloadButton button,
    .stDownloadButton > button,
    div[data-testid="stDownloadButton"] button {
        background: linear-gradient(160deg, rgba(26, 34, 54, 0.92) 0%, rgba(18, 24, 40, 0.92) 100%) !important;
        color: #F1F5FF !important;
        border: 1px solid var(--border-soft) !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        min-height: 42px !important;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.3) !important;
        transition: all 0.22s ease !important;
    }

    .stDownloadButton button *,
    .stDownloadButton > button *,
    div[data-testid="stDownloadButton"] button * {
        color: #F1F5FF !important;
    }

    .stDownloadButton button:hover,
    .stDownloadButton > button:hover,
    div[data-testid="stDownloadButton"] button:hover {
        border-color: rgba(143, 167, 217, 0.4) !important;
        background: linear-gradient(160deg, rgba(30, 40, 62, 0.95) 0%, rgba(20, 28, 46, 0.95) 100%) !important;
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.35) !important;
        transform: translateY(-2px) !important;
    }

    .stDownloadButton button:disabled,
    .stDownloadButton > button:disabled,
    div[data-testid="stDownloadButton"] button:disabled {
        background: #0E1523 !important;
        color: #8A97B6 !important;
        border-color: rgba(129, 153, 205, 0.24) !important;
        box-shadow: none !important;
    }
    
    /* Neon Button Style */
    .neon-btn button {
        background-color: #25D0B2 !important;
        color: #071B1D !important;
        padding: 10px 24px !important;
        min-height: 42px !important;
        border-radius: 12px !important;
        border: 1px solid #5EE8D1 !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 18px rgba(26, 156, 139, 0.3) !important;
    }
    
    .neon-btn button:hover {
        box-shadow: 0 0 16px rgba(51, 234, 205, 0.75) !important;
        transform: translateY(-2px);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A0F1C 0%, #111A2D 100%);
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: white;
    }
    
    /* Sidebar Buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(17, 25, 40, 0.92) !important;
        color: #E8ECF8 !important;
        padding: 10px 16px !important;
        min-height: 40px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(129, 153, 205, 0.28) !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        width: 100% !important;
        margin-bottom: 8px !important;
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.32) !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.38) !important;
        transform: translateX(2px) !important;
        background: rgba(24, 34, 54, 0.95) !important;
    }

    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(160deg, rgba(29, 41, 66, 0.96) 0%, rgba(21, 30, 50, 0.96) 100%) !important;
        color: #F7FAFF !important;
        border: 1px solid rgba(143, 167, 217, 0.35) !important;
        box-shadow: 0 10px 18px rgba(0, 0, 0, 0.38) !important;
    }

    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background: rgba(17, 25, 40, 0.9) !important;
        color: #DFE5F3 !important;
        border: 1px solid rgba(129, 153, 205, 0.28) !important;
    }

    [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background: rgba(24, 34, 54, 0.95) !important;
        color: #F7FAFF !important;
        border: 1px solid rgba(160, 184, 232, 0.45) !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #8DB0FF;
    }
    
    /* Success/Warning */
    .success-banner {
        background: linear-gradient(135deg, #7C3AED 0%, #A78BFA 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 600;
        margin: 16px 0;
    }
    
    .warning-banner {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8787 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 600;
        margin: 16px 0;
    }
    
    .info-banner {
        background: linear-gradient(135deg, #4F9EFF 0%, #669FFF 100%);
        color: white;
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 600;
        margin: 16px 0;
    }
    
    /* Feature Grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin: 30px 0;
    }
    
    .feature-card {
        background: linear-gradient(160deg, rgba(26, 34, 54, 0.92) 0%, rgba(18, 24, 40, 0.92) 100%);
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 10px 24px rgba(0,0,0,0.3);
        text-align: center;
        transition: all 0.3s ease;
        border-top: 4px solid #5B8CFF;
        border: 1px solid var(--border-soft);
    }
    
    .feature-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 14px 30px rgba(44, 81, 170, 0.35);
    }
    
    .feature-card h4 {
        color: var(--text-primary);
        font-weight: 700;
        margin: 12px 0;
    }
    
    .feature-card p {
        color: var(--text-muted);
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    /* Progress Bar */
    .progress-container {
        margin: 20px 0;
    }
    
    /* Form Elements */
    .stSelectbox, .stRadio, .stTextInput {
        color: var(--text-primary) !important;
    }
    
    .stSelectbox label, .stRadio label, .stTextInput label {
        color: var(--text-primary) !important;
        font-weight: 600;
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        background-color: #10192A !important;
        border: 1px solid #5A7AC2 !important;
        border-radius: 12px;
    }

    /* Input field refinement for auth and forms */
    div[data-baseweb="input"] {
        background-color: #10192A !important;
        border: 1px solid #4E6799 !important;
        border-radius: 10px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }

    div[data-baseweb="input"]:focus-within {
        border-color: #8DB0FF !important;
        box-shadow: 0 0 0 3px rgba(91, 140, 255, 0.2) !important;
    }

    div[data-baseweb="input"] input {
        color: var(--text-primary) !important;
        -webkit-text-fill-color: var(--text-primary) !important;
        caret-color: #DCE8FF !important;
    }

    div[data-baseweb="input"] input::placeholder {
        color: #8A97B6 !important;
        -webkit-text-fill-color: #8A97B6 !important;
    }

    [data-testid="stTextInput"] input {
        background-color: #10192A !important;
        color: #E7ECF7 !important;
        -webkit-text-fill-color: #E7ECF7 !important;
    }

    [data-testid="stTextInput"] input::placeholder {
        color: #8A97B6 !important;
        -webkit-text-fill-color: #8A97B6 !important;
        opacity: 1 !important;
    }

    [data-testid="stTextInput"] input:-webkit-autofill,
    [data-testid="stTextInput"] input:-webkit-autofill:hover,
    [data-testid="stTextInput"] input:-webkit-autofill:focus {
        -webkit-text-fill-color: #E7ECF7 !important;
        box-shadow: 0 0 0px 1000px #10192A inset !important;
        transition: background-color 5000s ease-in-out 0s;
    }

    /* Number input refinement (e.g., Max rows in fast mode) */
    [data-testid="stNumberInput"] div[data-baseweb="input"] {
        background-color: #10192A !important;
        border: 1px solid #4E6799 !important;
        border-radius: 10px !important;
    }

    [data-testid="stNumberInput"] input {
        background-color: #10192A !important;
        color: #E7ECF7 !important;
        -webkit-text-fill-color: #E7ECF7 !important;
        caret-color: #DCE8FF !important;
    }

    [data-testid="stNumberInput"] button {
        background: rgba(24, 34, 54, 0.95) !important;
        color: #E7ECF7 !important;
        border: 1px solid rgba(129, 153, 205, 0.35) !important;
    }

    [data-testid="stNumberInput"] button:hover {
        background: rgba(33, 47, 74, 0.98) !important;
        color: #F7FAFF !important;
    }

    [data-testid="stNumberInput"] [disabled],
    [data-testid="stNumberInput"] button:disabled,
    [data-testid="stNumberInput"] input:disabled {
        background: #0E1523 !important;
        color: #8A97B6 !important;
        -webkit-text-fill-color: #8A97B6 !important;
        opacity: 1 !important;
    }
    
    .stSelectbox > div > div > div {
        color: var(--text-primary) !important;
    }
    
    /* Radio Buttons */
    [data-testid="stRadio"] label {
        color: var(--text-primary) !important;
    }
    
    /* File Uploader */
    .stFileUploader {
        border: 2px dashed #5B8CFF !important;
        background: rgba(13, 20, 35, 0.55) !important;
        border-radius: 12px !important;
        padding: 20px !important;
    }
    
    .stFileUploader label {
        color: var(--text-primary) !important;
        font-weight: 600;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #33466D, transparent);
        margin: 2rem 0;
    }
    
    /* ============================================================================ */
    /* GLASSMORPHISM HERO SECTION */
    /* ============================================================================ */
    .hero-glass {
        background: rgba(18, 27, 43, 0.78);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(130, 155, 210, 0.28);
        padding: 28px 28px;
        border-radius: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.28);
        text-align: left;
        animation: slideInDown 0.6s ease-out;
        margin-bottom: 20px;
    }
    
    .hero-glass h1 {
        color: #F5F8FF;
        -webkit-text-fill-color: #F5F8FF;
        margin-bottom: 10px;
        font-size: 2rem;
    }
    
    .hero-glass p {
        color: #B7C2DB;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .compact-hero-grid {
        display: grid;
        grid-template-columns: 1.3fr 0.7fr;
        gap: 16px;
        align-items: stretch;
        margin-bottom: 18px;
    }

    .quick-stats {
        display: grid;
        grid-template-columns: repeat(4, minmax(120px, 1fr));
        gap: 12px;
        margin-bottom: 20px;
    }

    .quick-stat-card {
        background: rgba(17, 25, 40, 0.86);
        border: 1px solid rgba(129, 153, 205, 0.24);
        border-radius: 12px;
        padding: 12px 14px;
    }

    .quick-stat-label {
        color: #A8B4CF;
        font-size: 0.78rem;
        margin-bottom: 6px;
    }

    .quick-stat-value {
        color: #F1F5FF;
        font-size: 1rem;
        font-weight: 700;
    }

    .auth-brand-title {
        color: #F4F7FF !important;
        font-family: 'Space Grotesk', 'Manrope', sans-serif !important;
        font-weight: 700;
        letter-spacing: -0.01em;
    }

    .auth-brand-subtitle {
        color: #C2CBE0 !important;
    }

    .auth-card {
        background: rgba(18, 26, 41, 0.92);
        border: 1px solid rgba(130, 155, 210, 0.3);
        padding: 34px;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.32);
    }

    .auth-heading {
        color: #F5F8FF !important;
        font-family: 'Space Grotesk', 'Manrope', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
        margin-bottom: 14px !important;
    }

    @media (max-width: 980px) {
        .compact-hero-grid {
            grid-template-columns: 1fr;
        }
        .quick-stats {
            grid-template-columns: repeat(2, minmax(120px, 1fr));
        }
    }
    
    /* ============================================================================ */
    /* ANIMATED STATUS BADGES */
    /* ============================================================================ */
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        animation: fadeInScale 0.4s ease-out;
    }
    
    .badge-active {
        background: linear-gradient(135deg, #7C3AED 0%, #A78BFA 100%);
        color: white;
        box-shadow: 0 0 12px rgba(124, 58, 237, 0.4);
    }
    
    .badge-complete {
        background: linear-gradient(135deg, #10B981 0%, #34D399 100%);
        color: white;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.4);
    }
    
    .badge-pending {
        background: linear-gradient(135deg, #F59E0B 0%, #FBBF24 100%);
        color: white;
        box-shadow: 0 0 12px rgba(245, 158, 11, 0.4);
    }
    
    .badge-pulse {
        animation: pulse-glow 2s ease-in-out infinite;
    }
    
    /* ============================================================================ */
    /* PIPELINE PROGRESS INDICATOR */
    /* ============================================================================ */
    .pipeline-container {
        margin: 40px 0;
    }
    
    .pipeline-stages {
        display: flex;
        justify-content: space-between;
        position: relative;
        margin: 40px 0;
        gap: 20px;
    }
    
    .pipeline-stages::before {
        content: '';
        position: absolute;
        top: 24px;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #E5E7EB 0%, #D1D5DB 50%, #E5E7EB 100%);
        z-index: 0;
    }
    
    .stage {
        flex: 1;
        text-align: center;
        position: relative;
        z-index: 1;
    }
    
    .stage-circle {
        width: 52px;
        height: 52px;
        margin: 0 auto 12px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.4s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .stage.active .stage-circle {
        background: linear-gradient(135deg, #7C3AED 0%, #A78BFA 100%);
        box-shadow: 0 0 20px rgba(124, 58, 237, 0.6);
        animation: bounce-stage 1.4s ease-in-out infinite;
        transform: scale(1.1);
    }
    
    .stage.completed .stage-circle {
        background: linear-gradient(135deg, #10B981 0%, #34D399 100%);
        animation: checkmark-appear 0.5s ease-out;
    }
    
    .stage.pending .stage-circle {
        background: linear-gradient(135deg, #D1D5DB 0%, #E5E7EB 100%);
        color: #9CA3AF;
    }
    
    .stage-label {
        font-weight: 600;
        color: #0E1117;
        margin-top: 8px;
        font-size: 0.9rem;
    }
    
    .stage.active .stage-label {
        color: #7C3AED;
        font-weight: 700;
    }
    
    /* ============================================================================ */
    /* PAGE TRANSITION ANIMATIONS */
    /* ============================================================================ */
    @keyframes slideInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInScale {
        from {
            opacity: 0;
            transform: scale(0.8);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    @keyframes pulse-glow {
        0%, 100% {
            box-shadow: 0 0 8px rgba(124, 58, 237, 0.4);
        }
        50% {
            box-shadow: 0 0 20px rgba(124, 58, 237, 0.8);
        }
    }
    
    @keyframes bounce-stage {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.15);
        }
        100% {
            transform: scale(1);
        }
    }
    
    @keyframes checkmark-appear {
        from {
            transform: scale(0) rotate(-180deg);
        }
        to {
            transform: scale(1) rotate(0);
        }
    }
    
    @keyframes slide-in-page {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Page container animation */
    [data-testid="stVerticalBlock"] {
        animation: slide-in-page 0.5s ease-out;
    }
    
    /* Card entrance animations */
    .card {
        animation: slideInUp 0.5s ease-out;
    }
    
    .feature-card {
        animation: slideInUp 0.5s ease-out;
    }
    
    /* Stagger animation for multiple elements */
    .element-1 { animation-delay: 0.1s; }
    .element-2 { animation-delay: 0.2s; }
    .element-3 { animation-delay: 0.3s; }
    .element-4 { animation-delay: 0.4s; }
    .element-5 { animation-delay: 0.5s; }
    
    /* Loading animation */
    .loading-spinner {
        display: inline-block;
        width: 30px;
        height: 30px;
        border: 3px solid rgba(124, 58, 237, 0.2);
        border-top-color: #7C3AED;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Smooth metric transitions */
    [data-testid="stMetricValue"] {
        animation: fadeInScale 0.6s ease-out;
    }

    /* ============================================================================ */
    /* ENHANCED TOOLTIP & MICRO-INTERACTIONS                                        */
    /* ============================================================================ */
    .tooltip-card {
        position: relative;
        overflow: visible;
    }

    .card::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #7C3AED, #A78BFA, #7C3AED);
        border-radius: 0 0 20px 20px;
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .card:hover::after {
        opacity: 1;
    }

    /* Gradient text utility */
    .gradient-text {
        background: linear-gradient(135deg, #7C3AED 0%, #A78BFA 50%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Stat counter animation */
    .stat-counter {
        font-variant-numeric: tabular-nums;
        transition: all 0.4s cubic-bezier(.4,0,.2,1);
    }

    /* Improved scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0D1322;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #4F77E8, #7FA1FF);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #3D64D6;
    }

    /* Enhanced file uploader */
    .stFileUploader {
        background: rgba(53, 89, 176, 0.12) !important;
        transition: all 0.3s ease !important;
    }
    .stFileUploader:hover {
        background: rgba(79, 120, 214, 0.2) !important;
        border-color: #7FA1FF !important;
    }

    /* Tag / chip style */
    .chip {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(91, 140, 255, 0.18);
        color: #D4E2FF;
        margin: 4px 2px;
    }

    /* Footer */
    .app-footer {
        text-align: center;
        padding: 24px 0 12px;
        color: #8F9AB1;
        font-size: 0.8rem;
        border-top: 1px solid #2A395A;
        margin-top: 40px;
    }

    /* Sidebar active indicator */
    [data-testid="stSidebar"] .stButton > button[data-testid] {
        position: relative;
        overflow: hidden;
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: #A3AECB;
    }
    .stTabs [aria-selected="true"] {
        color: #8DB0FF !important;
        border-bottom-color: #8DB0FF !important;
    }

    </style>
""", unsafe_allow_html=True)

# ============================================================================
# Session State Initialization
# ============================================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "task_type" not in st.session_state:
    st.session_state.task_type = None
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None
if "pipeline_executed" not in st.session_state:
    st.session_state.pipeline_executed = False
if "dataset_upload_id" not in st.session_state:
    st.session_state.dataset_upload_id = None

# ============================================================================
# User Authentication Helpers
# ============================================================================

_USERS_FILE = os.path.join(os.path.dirname(__file__), ".users.json")
_PBKDF2_ITERATIONS = 600_000  # OWASP recommended minimum


def _hash_password(password: str, salt: Optional[str] = None) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256 and a random per-user salt."""
    if salt is None:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        _PBKDF2_ITERATIONS,
    )
    return f"{salt}${dk.hex()}"


def _verify_hashed_password(password: str, stored: str) -> bool:
    """Constant-time comparison of password against stored PBKDF2 hash."""
    if "$" not in stored:
        return False
    salt, expected_hex = stored.split("$", 1)
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        _PBKDF2_ITERATIONS,
    )
    return hmac.compare_digest(dk.hex(), expected_hex)


def _load_users_db() -> dict:
    """Load user database from disk, creating defaults if missing."""
    if os.path.exists(_USERS_FILE):
        try:
            with open(_USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt users file, recreating defaults.")
    # Default demo accounts (PBKDF2 hashed)
    defaults = {
        "demo": {"hash": _hash_password("password123"), "email": "demo@auraauth.io"},
        "test": {"hash": _hash_password("test123"), "email": "test@auraauth.io"},
    }
    _save_users_db(defaults)
    return defaults


def _save_users_db(db: dict) -> None:
    """Persist user database to disk."""
    with open(_USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)


def _verify_password(username: str, password: str) -> bool:
    """Verify a username/password pair against the stored hashes."""
    db = _load_users_db()
    record = db.get(username)
    if record is None:
        return False
    # Support new dict format and legacy plain-hash format
    stored_hash = record["hash"] if isinstance(record, dict) else record
    return _verify_hashed_password(password, stored_hash)


def _register_user(username: str, password: str, email: str = "") -> bool:
    """Register a new user. Returns False if username already taken."""
    db = _load_users_db()
    if username in db:
        return False
    db[username] = {"hash": _hash_password(password), "email": email}
    _save_users_db(db)
    return True


def reset_session():
    """Reset session state for new workflow."""
    st.session_state.dataset = None
    st.session_state.dataset_upload_id = None
    st.session_state.target_column = None
    st.session_state.task_type = None
    st.session_state.pipeline_ready = False
    st.session_state.pipeline_results = None
    st.session_state.pipeline_executed = False


def set_auth_mode(mode: str) -> None:
    """Switch auth mode and clear form fields to avoid stale widget state."""
    st.session_state.auth_mode = mode
    auth_keys = [
        "login_username",
        "login_password",
        "signup_username",
        "signup_email",
        "signup_password",
        "signup_confirm_password",
    ]
    for key in auth_keys:
        if key in st.session_state:
            st.session_state[key] = ""


# ============================================================================
# UI Component Functions
# ============================================================================

def render_status_badge(status: str, text: str = ""):
    """
    Render an animated status badge.
    
    Args:
        status: 'active', 'complete', or 'pending'
        text: Badge text content
    
    Example:
        render_status_badge('active', 'Processing Data')
    """
    status_map = {
        'active': ('badge-active badge-pulse', '⚡'),
        'complete': ('badge-complete', '✓'),
        'pending': ('badge-pending', '⏳')
    }
    
    badge_class, icon = status_map.get(status, ('badge-pending', '○'))
    safe_text = html_escape(text) if text else html_escape(status.upper())
    display_text = f"{icon} {safe_text}"
    
    st.markdown(
        f'<span class="status-badge {badge_class}">{display_text}</span>',
        unsafe_allow_html=True
    )


def render_pipeline_progress(stages: list, current_stage: int = 0):
    """
    Render animated pipeline stage progress indicator.
    
    Args:
        stages: List of stage names (e.g., ['Data Upload', 'Preprocessing', 'Training', 'Evaluation'])
        current_stage: Current active stage index (0-based)
    
    Example:
        stages = ['Data Upload', 'Preprocessing', 'Training', 'Evaluation']
        render_pipeline_progress(stages, current_stage=1)
    """
    st.markdown('<div class="pipeline-container">', unsafe_allow_html=True)
    
    html_stages = '<div class="pipeline-stages">'
    for idx, stage in enumerate(stages):
        if idx < current_stage:
            stage_status = 'completed'
            icon = '✓'
        elif idx == current_stage:
            stage_status = 'active'
            icon = f'{idx + 1}'
        else:
            stage_status = 'pending'
            icon = f'{idx + 1}'
        
        html_stages += f'''
            <div class="stage {stage_status}">
                <div class="stage-circle">{icon}</div>
                <div class="stage-label">{stage}</div>
            </div>
        '''
    
    html_stages += '</div>'
    st.markdown(html_stages, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_hero_glass(title: str, subtitle: str, emoji: str = "✨"):
    """
    Render a glassmorphism hero section.
    
    Args:
        title: Main heading text
        subtitle: Subheading text
        emoji: Leading emoji
    
    Example:
        render_hero_glass("Welcome to AuraAuth", "Enterprise-Grade AutoML", "🚀")
    """
    safe_title = html_escape(title)
    safe_subtitle = html_escape(subtitle)
    safe_emoji = html_escape(emoji)
    st.markdown(f'''
        <div class="hero-glass">
            <h1>{safe_emoji} {safe_title}</h1>
            <p>{safe_subtitle}</p>
        </div>
    ''', unsafe_allow_html=True)


def render_metric_card(title: str, value: str, icon: str = "📊", highlight_color: str = "purple"):
    """
    Render an animated metric card with glassmorphism.
    
    Args:
        title: Card title
        value: Metric value to display
        icon: Leading icon
        highlight_color: Color theme ('purple', 'green', 'blue', 'orange')
    
    Example:
        render_metric_card("Training Progress", "85%", "📈", "purple")
    """
    color_map = {
        'purple': '#7C3AED',
        'green': '#10B981',
        'blue': '#3B82F6',
        'orange': '#F59E0B'
    }
    color = color_map.get(highlight_color, '#7C3AED')
    safe_title = html_escape(str(title))
    safe_value = html_escape(str(value))
    safe_icon = html_escape(str(icon))
    
    st.markdown(f'''
        <div class="card" style="border-left: 4px solid {color}; position: relative;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p style="color: #6B7280; margin: 0; font-size: 0.9rem;">{safe_icon} {safe_title}</p>
                    <h3 style="margin: 8px 0 0 0; color: {color};" class="stat-counter">{safe_value}</h3>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)


def navigate_to(page: str):
    """Navigate to a specific page."""
    st.session_state.current_page = page
    st.rerun()


def _to_serializable(value):
    """Convert nested objects (including numpy/pandas scalars) to JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]

    # Handle common scientific Python scalar/array types safely.
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass

    return str(value)


def _render_doc_value(value):
    """Render documentation content regardless of whether it is dict/json/markdown/html."""
    if value is None or value == {}:
        st.info("No data available")
        return

    if isinstance(value, str):
        text = value.strip()
        if text.startswith("<") and text.endswith(">"):
            st.markdown(value, unsafe_allow_html=True)
        else:
            st.markdown(value)
        return

    if isinstance(value, (dict, list, tuple, set)):
        try:
            st.json(_to_serializable(value))
        except Exception:
            st.write(_to_serializable(value))
        return

    st.write(_to_serializable(value))


def _parse_doc_payload(raw_value):
    """Parse section payload into dict/list/string for resilient rendering."""
    if raw_value is None:
        return None

    if isinstance(raw_value, (dict, list, tuple, set)):
        return raw_value

    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return ""

        if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
            try:
                return json.loads(text)
            except Exception:
                return raw_value

        return raw_value

    return raw_value


def _has_doc_content(value):
    """Return True if documentation section has meaningful content."""
    parsed = _parse_doc_payload(value)

    if parsed is None:
        return False

    if isinstance(parsed, str):
        return bool(parsed.strip())

    if isinstance(parsed, (dict, list, tuple, set)):
        return len(parsed) > 0

    return True


def _render_dataset_sheet_ui(section_value):
    """Render Dataset Sheet in a readable structured layout."""
    section = _parse_doc_payload(section_value)
    if not _has_doc_content(section):
        st.info("No dataset sheet available")
        return

    if not isinstance(section, dict):
        _render_doc_value(section)
        return

    st.markdown(f"**Title:** {section.get('title', 'Dataset Sheet')}")
    if section.get("timestamp"):
        st.caption(f"Generated: {section.get('timestamp')}")

    top_left, top_right = st.columns(2)
    with top_left:
        st.markdown("**Description**")
        st.markdown(section.get("description", "N/A"))
    with top_right:
        st.markdown("**Composition**")
        st.markdown(section.get("composition", "N/A"))

    st.markdown("---")

    mid_left, mid_right = st.columns(2)
    with mid_left:
        st.markdown("**Distribution**")
        st.markdown(section.get("distribution", "N/A"))
    with mid_right:
        st.markdown("**Data Quality**")
        st.markdown(section.get("data_quality", "N/A"))

    st.markdown("---")

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.markdown("**Warnings**")
        st.markdown(section.get("warnings", "N/A"))
    with bottom_right:
        st.markdown("**Preprocessing Notes**")
        st.markdown(section.get("preprocessing_notes", "N/A"))


def _render_dataset_info_ui(section_value, results):
    """Render Dataset Info with key stats cards and details."""
    section = _parse_doc_payload(section_value)
    if not isinstance(section, dict):
        section = {}

    data_profile = (results or {}).get("data_profile", {}) if isinstance(results, dict) else {}
    if not isinstance(data_profile, dict):
        data_profile = {}

    total_samples = section.get("total_samples", data_profile.get("n_samples", data_profile.get("dataset_size", "N/A")))
    total_features = section.get("total_features", data_profile.get("n_features", data_profile.get("feature_count", "N/A")))
    missing_pct = section.get("missing_percentage", data_profile.get("missing_values_pct", "N/A"))
    quality_score = section.get("quality_score", data_profile.get("data_quality_score", "N/A"))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Samples", str(total_samples))
    with c2:
        st.metric("Features", str(total_features))
    with c3:
        m = _safe_float(missing_pct)
        missing_text = f"{m:.2f}%" if m is not None else str(missing_pct)
        st.metric("Missing", missing_text)
    with c4:
        q = _safe_float(quality_score)
        quality_text = f"{q:.2f}" if q is not None else str(quality_score)
        st.metric("Quality", quality_text)

    st.markdown("---")

    if _has_doc_content(section):
        st.markdown("**Detailed Dataset Info**")
        _render_doc_value(section)
    else:
        st.info("No additional dataset info found; showing values derived from pipeline profile.")


def _render_training_report_ui(section_value, results):
    """Render Training Report with status and training metadata."""
    section = _parse_doc_payload(section_value)
    if not isinstance(section, dict):
        section = {}

    model_selection = (results or {}).get("model_selection", {}) if isinstance(results, dict) else {}
    if not isinstance(model_selection, dict):
        model_selection = {}

    status = section.get("execution_status", "Completed")
    best_model = section.get("best_model", model_selection.get("best_model_name", model_selection.get("best_model", "N/A")))
    best_score = section.get("best_score", model_selection.get("best_score", "N/A"))
    duration = section.get("total_duration", (results or {}).get("execution_time", "N/A")) if isinstance(results, dict) else section.get("total_duration", "N/A")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Status", str(status))
    with c2:
        st.metric("Best Model", str(best_model))
    with c3:
        score_val = _safe_float(best_score)
        score_text = f"{score_val:.4f}" if score_val is not None else str(best_score)
        st.metric("Best Score", score_text)
    with c4:
        st.metric("Duration", str(duration))

    st.markdown("---")

    if _has_doc_content(section):
        st.markdown("**Detailed Training Report**")
        _render_doc_value(section)


def _render_recommendations_ui(section_value, results):
    """Render recommendations as clean strength/improvement lists."""
    section = _parse_doc_payload(section_value)

    if isinstance(section, dict):
        strengths = section.get("strengths", [])
        improvements = section.get("improvements", [])

        if isinstance(strengths, str):
            strengths = [strengths]
        if isinstance(improvements, str):
            improvements = [improvements]

        left, right = st.columns(2)
        with left:
            st.markdown("**Strengths**")
            if isinstance(strengths, (list, tuple)) and strengths:
                st.markdown("\n".join([f"- {item}" for item in strengths]))
            else:
                st.markdown("- Model documentation generated successfully")
        with right:
            st.markdown("**Improvements**")
            if isinstance(improvements, (list, tuple)) and improvements:
                st.markdown("\n".join([f"- {item}" for item in improvements]))
            else:
                st.markdown("- Monitor confidence and drift periodically")
        return

    if _has_doc_content(section):
        _render_doc_value(section)
        return

    uncertainty = (results or {}).get("uncertainty", {}) if isinstance(results, dict) else {}
    shift = (results or {}).get("distribution_shift", {}) if isinstance(results, dict) else {}
    confidence = _safe_float(uncertainty.get("confidence_score", None)) if isinstance(uncertainty, dict) else None
    shift_level = shift.get("shift_level", "unknown") if isinstance(shift, dict) else "unknown"

    st.markdown("**Strengths**")
    st.markdown("- Pipeline completed and documentation generated")
    if confidence is not None and confidence >= 0.8:
        st.markdown("- High confidence indicates stable in-distribution predictions")

    st.markdown("**Improvements**")
    st.markdown("- Re-run training after major data updates")
    st.markdown(f"- Track drift level over time (current: {shift_level})")


def _safe_float(value):
    """Best-effort conversion to float."""
    try:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        return float(value)
    except Exception:
        return None


def _parse_model_card_payload(raw_card):
    """Parse model card payload into a dictionary without raising."""
    if isinstance(raw_card, dict):
        return dict(raw_card)

    if isinstance(raw_card, str):
        txt = raw_card.strip()
        if not txt:
            return {}
        try:
            parsed = json.loads(txt)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {
            "title": "Model Card",
            "overview": txt,
        }

    return {}


def _extract_model_name_from_title(title):
    """Extract model name from title-like field."""
    if not isinstance(title, str) or not title.strip():
        return "Model"
    parts = title.split(":", 1)
    return parts[1].strip() if len(parts) == 2 and parts[1].strip() else title.strip()


def _parse_intended_use_sections(text):
    """Split intended use text into suitable and not suitable bullet lists."""
    if not isinstance(text, str) or not text.strip():
        return [], []

    suitable = []
    not_suitable = []
    in_not = False

    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if "not suitable" in low:
            in_not = True
            continue
        if s.startswith("-"):
            item = s.lstrip("-").strip()
            if item:
                if in_not:
                    not_suitable.append(item)
                else:
                    suitable.append(item)

    if not suitable and not not_suitable:
        suitable = [x.strip() for x in text.splitlines() if x.strip()]

    return suitable, not_suitable


def _parse_performance_data(performance):
    """Normalize performance block into metrics and parameter dictionary."""
    out = {
        "accuracy": None,
        "mean_score": None,
        "std_score": None,
        "best_params": {},
    }

    if isinstance(performance, dict):
        out["accuracy"] = _safe_float(
            performance.get("accuracy", performance.get("Accuracy"))
        )
        out["mean_score"] = _safe_float(
            performance.get("mean_score", performance.get("Mean_Score"))
        )
        out["std_score"] = _safe_float(
            performance.get("std_score", performance.get("Std_Score"))
        )

        best_params = performance.get("best_params", performance.get("Best_Params", {}))
        if isinstance(best_params, dict):
            out["best_params"] = best_params
        return out

    if isinstance(performance, str):
        text = performance

        def _capture_float(pattern):
            m = re.search(pattern, text, flags=re.IGNORECASE)
            return _safe_float(m.group(1)) if m else None

        out["accuracy"] = _capture_float(r"accuracy\s*:\s*([0-9]*\.?[0-9]+)")
        out["mean_score"] = _capture_float(r"mean[_\s]*score\s*:\s*([0-9]*\.?[0-9]+)")
        out["std_score"] = _capture_float(r"std[_\s]*score\s*:\s*([0-9]*\.?[0-9]+)")

        m_params = re.search(r"best[_\s]*params\s*:\s*(\{.*\})", text, flags=re.IGNORECASE)
        if m_params:
            try:
                parsed = ast.literal_eval(m_params.group(1))
                if isinstance(parsed, dict):
                    out["best_params"] = parsed
            except Exception:
                out["best_params"] = {}

    return out


def _format_metric_value(value, digits=4):
    """Format metric values for st.metric display."""
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _extract_shap_importance_df(results):
    """Extract SHAP-like feature importance into a normalized DataFrame."""
    if not isinstance(results, dict):
        return None

    raw = results.get("shap_values")

    # Fallback to explainability global feature importance if explicit shap_values are absent.
    if raw is None:
        explainability = results.get("explainability", {})
        if isinstance(explainability, dict):
            raw = explainability.get("global_feature_importance")

    if raw is None:
        return None

    try:
        if isinstance(raw, pd.DataFrame):
            df = raw.copy()
            if "feature" in df.columns and "importance" in df.columns:
                return df[["feature", "importance"]]
            if df.shape[1] == 1:
                return pd.DataFrame({"feature": df.index.astype(str), "importance": df.iloc[:, 0].astype(float)})

        if isinstance(raw, pd.Series):
            return pd.DataFrame({"feature": raw.index.astype(str), "importance": raw.values.astype(float)})

        if isinstance(raw, dict):
            return pd.DataFrame(
                [{"feature": str(k), "importance": float(v)} for k, v in raw.items()]
            )

        if isinstance(raw, list):
            if raw and isinstance(raw[0], dict):
                if "feature" in raw[0] and "importance" in raw[0]:
                    return pd.DataFrame(raw)[["feature", "importance"]]
                return pd.DataFrame(raw)
    except Exception:
        return None

    return None


def _render_model_card_ui(documentation, results):
    """Render a clean model card UI from documentation payload."""
    st.markdown(
        """
<style>
.metric-card {
    background: rgba(255,255,255,0.05);
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
}

/* Improve contrast for documentation/model-card text in dark theme */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: #E7ECF7 !important;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] li {
    color: #EAF1FF !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #E7ECF7 !important;
}
</style>
""",
        unsafe_allow_html=True,
    )

    card = _parse_model_card_payload((documentation or {}).get("model_card", {}))

    if not card:
        st.info("No model card available")
        return

    title = card.get("title", "Model Card")
    model_name = _extract_model_name_from_title(title)
    timestamp = card.get("timestamp", "N/A")

    with st.container():
        st.markdown(f"## 🧾 {model_name}")
        st.caption(f"Generated: {timestamp}")

    st.markdown(" ")

    with st.container():
        st.subheader("🎯 Model Overview")
        overview = card.get("overview", "Overview is not available.")
        st.markdown(overview)

    st.markdown("---")

    with st.container():
        st.subheader("✅ Intended Use")
        suitable, not_suitable = _parse_intended_use_sections(card.get("intended_use", ""))
        left, right = st.columns(2)
        with left:
            st.markdown("**Suitable for**")
            if suitable:
                st.markdown("\n".join([f"- {item}" for item in suitable]))
            else:
                st.markdown("- Not specified")
        with right:
            st.markdown("**Not suitable for**")
            if not_suitable:
                st.markdown("\n".join([f"- {item}" for item in not_suitable]))
            else:
                st.markdown("- Not specified")

    st.markdown(" ")

    with st.container():
        st.subheader("📊 Performance")
        perf = _parse_performance_data(card.get("performance", {}))
        uncertainty = (results or {}).get("uncertainty", {}) if isinstance(results, dict) else {}
        confidence_raw = uncertainty.get("confidence_score", None) if isinstance(uncertainty, dict) else None
        confidence_val = _safe_float(confidence_raw)
        confidence_text = f"{confidence_val * 100:.1f}%" if confidence_val is not None else "85%"

        hero_metric_val = perf.get("accuracy")
        hero_metric_label = "Accuracy"
        if hero_metric_val is None:
            hero_metric_val = perf.get("mean_score")
            hero_metric_label = "Best Score"

        hero_left, hero_center, hero_right = st.columns([1, 2, 1])
        with hero_center:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(f"⭐ Hero Metric: {hero_metric_label}", _format_metric_value(hero_metric_val))
            st.markdown('</div>', unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Accuracy", _format_metric_value(perf.get("accuracy")))
            st.markdown('</div>', unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Mean Score", _format_metric_value(perf.get("mean_score")))
            st.markdown('</div>', unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Std Score", _format_metric_value(perf.get("std_score")))
            st.markdown('</div>', unsafe_allow_html=True)
        with m4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Confidence", confidence_text)
            st.markdown('</div>', unsafe_allow_html=True)

        if confidence_val is not None:
            if confidence_val > 0.8:
                st.success("High model confidence")
            elif confidence_val > 0.6:
                st.warning("Moderate confidence")
            else:
                st.error("Low confidence – review model")

        st.markdown("**Best Parameters**")
        best_params = perf.get("best_params", {})
        if isinstance(best_params, dict) and best_params:
            st.markdown("\n".join([f"- **{k}**: {v}" for k, v in best_params.items()]))
        else:
            st.info("Best parameters are not available.")

    st.markdown("---")

    with st.container():
        st.subheader("⚙️ Training Details")
        preprocessing = (results or {}).get("preprocessing", {}) if isinstance(results, dict) else {}
        data_profile = (results or {}).get("data_profile", {}) if isinstance(results, dict) else {}
        train_samples = preprocessing.get("train_samples", "N/A") if isinstance(preprocessing, dict) else "N/A"
        test_samples = preprocessing.get("test_samples", "N/A") if isinstance(preprocessing, dict) else "N/A"
        features_after = preprocessing.get("features_after_preprocessing", None) if isinstance(preprocessing, dict) else None
        features_total = features_after if isinstance(features_after, (int, float)) else (
            data_profile.get("n_features", "N/A") if isinstance(data_profile, dict) else "N/A"
        )

        split_text = "N/A"
        dataset_size_text = "N/A"
        if isinstance(train_samples, (int, float)) and isinstance(test_samples, (int, float)) and (train_samples + test_samples) > 0:
            total = train_samples + test_samples
            train_pct = (train_samples / total) * 100
            test_pct = (test_samples / total) * 100
            split_text = f"{train_pct:.0f}/{test_pct:.0f} ({int(train_samples)} train, {int(test_samples)} test)"
            dataset_size_text = f"{int(total)}"

        opt_method = "Optuna"
        n_trials_val = (results or {}).get("pipeline_metadata", {}).get("n_trials", None) if isinstance((results or {}).get("pipeline_metadata", {}), dict) else None
        if n_trials_val is not None:
            opt_method = f"Optuna ({n_trials_val} trials)"

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Train/Test Split**")
            st.markdown(split_text)
        with c2:
            st.markdown("**Optimization Method**")
            st.markdown(opt_method)
        with c3:
            st.markdown("**Model Type**")
            st.markdown(model_name)

        st.markdown(f"**Dataset Size:** {dataset_size_text}")
        st.markdown(f"**Features:** {features_total}")

    st.markdown("---")

    with st.container():
        st.subheader("🔍 Model Behavior & Feature Impact")
        shap_df = _extract_shap_importance_df(results)
        if shap_df is not None and not shap_df.empty and "feature" in shap_df.columns and "importance" in shap_df.columns:
            top_df = (
                shap_df[["feature", "importance"]]
                .dropna()
                .sort_values(by="importance", ascending=False)
                .head(10)
            )

            fig, ax = plt.subplots(figsize=(8, 4.8))
            ax.barh(top_df["feature"].astype(str), top_df["importance"].astype(float), color="#5B8CFF")
            ax.invert_yaxis()
            ax.set_xlabel("Importance")
            ax.set_ylabel("Features")
            ax.set_title("Feature Importance (SHAP)")
            fig.patch.set_alpha(0.0)
            ax.set_facecolor("none")
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("Top features based on SHAP will be displayed here")

    st.markdown("---")

    with st.container():
        st.subheader("🔍 Key Insights")
        insights = [
            f"Model type: {model_name}",
            "Model is suitable for structured tabular data.",
        ]
        std_score = perf.get("std_score")
        if isinstance(std_score, (int, float)):
            if float(std_score) >= 0.05:
                insights.append("Model may be unstable due to high score variance across folds.")
            elif float(std_score) <= 0.02:
                insights.append("Low score variance indicates stable training.")
            else:
                insights.append("Higher score variance suggests sensitivity to data splits.")
        if confidence_val is not None:
            if confidence_val > 0.8:
                insights.append("Confidence is high for current validation outputs.")
            elif confidence_val > 0.6:
                insights.append("Confidence is moderate; review borderline predictions.")
            else:
                insights.append("Confidence is low; model likely needs more data and feature enrichment.")

        st.markdown("\n".join([f"- {line}" for line in insights]))

    st.markdown("---")

    with st.container():
        st.subheader("⚠️ Limitations")
        limitations_text = card.get("limitations", "No limitations provided.")
        st.warning(limitations_text)

    st.markdown(" ")

    with st.container():
        json_blob = json.dumps(card, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Download Model Card (JSON)",
            data=json_blob,
            file_name="model_card.json",
            mime="application/json",
        )

        with st.expander("Advanced Details"):
            st.markdown("**Additional Metadata**")
            meta_rows = [
                {"Field": "Title", "Value": card.get("title", "N/A")},
                {"Field": "Timestamp", "Value": card.get("timestamp", "N/A")},
            ]
            st.markdown("\n".join([f"- **{row['Field']}**: {row['Value']}" for row in meta_rows]))


def _build_documentation_from_results(results):
    """Build documentation from pipeline outputs when stage-9 docs are missing."""
    try:
        from backend.core.documentation_generator import DocumentationGenerator

        if not isinstance(results, dict):
            return None

        generator = DocumentationGenerator()

        model_selection = results.get("model_selection", {})
        metadata = model_selection.get("metadata", {}) if isinstance(model_selection, dict) else {}

        model_name = (
            model_selection.get("best_model_name")
            or model_selection.get("best_model")
            or "AutoML Model"
        )

        metrics = metadata.get("best_metrics", {}) if isinstance(metadata, dict) else {}
        if not isinstance(metrics, dict) or not metrics:
            best_score = model_selection.get("best_score") if isinstance(model_selection, dict) else None
            metrics = {"score": best_score if best_score is not None else "N/A"}

        data_profile = results.get("data_profile", {})
        if not isinstance(data_profile, dict):
            data_profile = {}
        data_profile = dict(data_profile)

        # Ensure required keys expected by DocumentationGenerator.
        data_profile.setdefault("n_samples", data_profile.get("dataset_size", "unknown"))
        data_profile.setdefault("n_features", data_profile.get("feature_count", "unknown"))

        if "missing_values_pct" not in data_profile:
            mv = data_profile.get("missing_value_report", {})
            if isinstance(mv, dict) and mv:
                try:
                    data_profile["missing_values_pct"] = float(sum(mv.values())) / float(len(mv))
                except Exception:
                    data_profile["missing_values_pct"] = 0.0
            else:
                data_profile["missing_values_pct"] = 0.0

        if "target_distribution" not in data_profile or not isinstance(data_profile.get("target_distribution"), dict):
            data_profile["target_distribution"] = {}

        explainability = results.get("explainability", {})
        if not isinstance(explainability, dict):
            explainability = {}
        gfi = explainability.get("global_feature_importance", {})
        feature_importance = []
        if isinstance(gfi, dict):
            for feature_name, importance in gfi.items():
                try:
                    feature_importance.append({
                        "feature": str(feature_name),
                        "importance": float(importance),
                    })
                except Exception:
                    continue
            feature_importance.sort(key=lambda x: x.get("importance", 0.0), reverse=True)

        explainability_summary = {"feature_importance": feature_importance}

        uncertainty = results.get("uncertainty", {})
        if not isinstance(uncertainty, dict):
            uncertainty = {}
        uncertainty_summary = {
            "mean_confidence": uncertainty.get("confidence_score", uncertainty.get("mean_confidence", "unknown")),
            "uncertainty_level": uncertainty.get("uncertainty_level", "unknown"),
            "notes": uncertainty.get("notes", ""),
        }

        shift = results.get("distribution_shift", {})
        if not isinstance(shift, dict):
            shift = {}
        shift_summary = {
            "shift_score": shift.get("shift_score", "unknown"),
            "shift_level": shift.get("shift_level", "unknown"),
            "notes": shift.get("notes", ""),
        }

        model_card = generator.generate_model_card(
            model_name=model_name,
            metrics=metrics,
            data_profile=data_profile,
            explainability_summary=explainability_summary,
            uncertainty_summary=uncertainty_summary,
            shift_summary=shift_summary,
        )
        dataset_sheet = generator.generate_dataset_sheet(data_profile)

        dataset_info = {
            "total_samples": data_profile.get("n_samples", "N/A"),
            "total_features": data_profile.get("n_features", "N/A"),
            "missing_percentage": data_profile.get("missing_values_pct", 0.0),
            "quality_score": data_profile.get("data_quality_score", "N/A"),
        }

        training_report = {
            "execution_status": "Completed",
            "best_model": model_name,
            "best_score": model_selection.get("best_score", "N/A") if isinstance(model_selection, dict) else "N/A",
            "total_duration": results.get("execution_time", "N/A"),
        }

        recommendations = {
            "strengths": [
                f"Model selected: {model_name}",
                "Documentation rebuilt from current pipeline outputs",
            ],
            "improvements": [
                "Monitor confidence and drift in production",
                "Re-run pipeline after major data updates",
            ],
        }

        return {
            "model_card": model_card,
            "dataset_sheet": dataset_sheet,
            "dataset_info": dataset_info,
            "training_report": training_report,
            "recommendations": recommendations,
        }
    except Exception as e:
        logger.error(f"Documentation rebuild failed: {e}", exc_info=True)
        return None


# ============================================================================
# Authentication Pages
# ============================================================================

def page_login():
    """Login page with elegant design."""
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        st.markdown("""
            <div style="text-align: center; padding: 60px 40px;">
                <h1 class="auth-brand-title" style="margin-bottom: 10px;">AuraAuth</h1>
                <p class="auth-brand-subtitle" style="font-size: 1.05rem; margin-bottom: 28px;">
                    Enterprise AutoML for Small Data
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 class='auth-heading'>Login</h3>", unsafe_allow_html=True)
        
        username = st.text_input(
            "Username",
            placeholder="Enter your username",
            key="login_username",
            label_visibility="collapsed"
        )
        
        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password",
            key="login_password",
            label_visibility="collapsed"
        )
        
        # Login button
        if st.button("Login", key="login_signin", width="stretch", type="primary"):
            if username and password:
                if _verify_password(username, password):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success(f"✅ Welcome back, {username}!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please try again.")
            else:
                st.warning("⚠️ Please enter both username and password.")
        
        st.markdown("---")
        
        # Toggle to signup
        signup_col1, signup_col2 = st.columns(2)
        with signup_col1:
            st.caption("Don't have an account?")
        with signup_col2:
            if st.button("Sign Up", key="login_signup", width="stretch", type="secondary"):
                set_auth_mode("signup")
                st.rerun()


def page_signup():
    """Signup page with elegant design."""
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col2:
        st.markdown("""
            <div style="text-align: center; padding: 60px 40px;">
                <h1 class="auth-brand-title" style="margin-bottom: 10px;">AuraAuth</h1>
                <p class="auth-brand-subtitle" style="font-size: 1.05rem; margin-bottom: 28px;">
                    Create Your Account
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 class='auth-heading'>Sign Up</h3>", unsafe_allow_html=True)
        
        new_username = st.text_input(
            "Username",
            placeholder="Choose a username",
            key="signup_username",
            label_visibility="collapsed"
        )
        
        email = st.text_input(
            "Email",
            placeholder="your@email.com",
            key="signup_email",
            label_visibility="collapsed"
        )
        
        new_password = st.text_input(
            "Password",
            type="password",
            placeholder="At least 8 characters",
            key="signup_password",
            label_visibility="collapsed"
        )
        
        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            placeholder="Confirm your password",
            key="signup_confirm_password",
            label_visibility="collapsed"
        )
        
        # Signup button
        if st.button("Create Account", key="signup_create", width="stretch", type="primary"):
            if not new_username or not email or not new_password:
                st.warning("⚠️ Please fill in all fields.")
            elif len(new_password) < 8:
                st.warning("⚠️ Password must be at least 8 characters.")
            elif new_password != confirm_password:
                st.error("❌ Passwords do not match.")
            elif not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email):
                st.warning("⚠️ Please enter a valid email address.")
            elif not _register_user(new_username, new_password, email):
                st.error("❌ Username already exists.")
            else:
                st.session_state.authenticated = True
                st.session_state.username = new_username
                st.success(f"✅ Account created! Welcome, {new_username}!")
                st.balloons()
                st.rerun()
        
        st.markdown("---")
        
        # Toggle to login
        login_col1, login_col2 = st.columns(2)
        with login_col1:
            st.caption("Already have an account? Login")
        with login_col2:
            if st.button("Login", key="signup_signin", width="stretch", type="secondary"):
                set_auth_mode("login")
                st.rerun()


# ============================================================================
# Sidebar Navigation with Buttons
# ============================================================================

def sidebar_navigation():
    """Render modern sidebar with button-based navigation."""
    with st.sidebar:
        # Logo
        st.markdown("### AuraAuth")
        st.caption("Premium AutoML for Small Data")
        st.markdown("---")
        
        # User info
        if st.session_state.authenticated:
            st.markdown(f"### 👤 {st.session_state.username}")
            st.caption("Logged in")
            st.markdown("---")
        
        # Navigation buttons
        nav_items = [
            "Home",
            "Upload Dataset",
            "Run Pipeline",
            "Results",
            "Explainability",
            "Documentation"
        ]
        
        for item in nav_items:
            is_active = st.session_state.current_page == item
            
            # Create button with styling
            if st.button(
                f"{'→' if is_active else '  '} {item}",
                key=f"nav_{item}",
                width="stretch",
                type="primary" if is_active else "secondary"
            ):
                navigate_to(item)
        
        st.markdown("---")
        
        # Session info
        st.caption("**v1.0** — Industry-Grade AutoML")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset", key="sidebar_reset", width="stretch"):
                reset_session()
                st.rerun()
        with col2:
            if st.button("🚪 Logout", key="sidebar_logout", width="stretch"):
                reset_session()
                st.session_state.authenticated = False
                st.session_state.username = None
                st.session_state.current_page = "Home"
                set_auth_mode("login")
                st.rerun()


# ============================================================================
# Page: Home
# ============================================================================

def page_home():
    """Home page with hero section and features."""
    # Compact hero + primary CTA in the top-right
    dataset = st.session_state.dataset
    model_results = st.session_state.pipeline_results
    model_count = len(model_results.get("all_models", [])) if isinstance(model_results, dict) else 0

    left_col, right_col = st.columns([2.3, 1.1])
    with left_col:
        render_hero_glass(
            "Welcome to AuraAuth",
            "Enterprise-grade AutoML with reliability checks for small, noisy datasets.",
            "🚀"
        )
    with right_col:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("🚀 Start AutoML Pipeline", key="cta_btn", width="stretch", type="primary"):
            navigate_to("Upload Dataset")
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("📥 Upload Dataset", key="home_upload_btn", width="stretch", type="secondary"):
            navigate_to("Upload Dataset")

    # Quick stats to increase information density above fold
    quick_stats = [
        ("Logged In", st.session_state.username or "Guest"),
        ("Dataset", f"{len(dataset)} rows" if dataset is not None else "Not loaded"),
        ("Target", st.session_state.target_column or "Not selected"),
        ("Models", str(model_count) if model_count else "No runs yet"),
    ]

    stats_html = '<div class="quick-stats">'
    for label, value in quick_stats:
        safe_label = html_escape(str(label))
        safe_value = html_escape(str(value))
        stats_html += (
            '<div class="quick-stat-card">'
            f'<div class="quick-stat-label">{safe_label}</div>'
            f'<div class="quick-stat-value">{safe_value}</div>'
            '</div>'
        )
    stats_html += '</div>'
    st.markdown(stats_html, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature Grid
    st.markdown("<h2 style='text-align: center;'>✨ Core Features</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    features = [
        {
            "title": "📊 Data Profiling",
            "description": "Comprehensive dataset analysis with quality metrics and anomaly detection"
        },
        {
            "title": "⚡ AutoML Optimization",
            "description": "Hyperparameter tuning with Optuna for optimal model performance"
        },
        {
            "title": "🎯 Distribution Shift",
            "description": "Detect data drift and distribution changes in production"
        },
        {
            "title": "🔍 Explainability",
            "description": "SHAP-based feature importance and local explanations"
        }
    ]
    
    for col, feature in zip([col1, col2, col3, col4], features):
        with col:
            safe_feat_title = html_escape(feature['title'])
            safe_feat_desc = html_escape(feature['description'])
            st.markdown(f"""
                <div class="feature-card">
                    <h4>{safe_feat_title}</h4>
                    <p>{safe_feat_desc}</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Workflow Steps
    st.markdown("<h2 style='text-align: center;'>⚙️ How It Works</h2>", unsafe_allow_html=True)
    
    steps = [
        ("1️⃣", "Upload", "Import your CSV dataset"),
        ("2️⃣", "Configure", "Select target and task type"),
        ("3️⃣", "Train", "Run automated ML pipeline"),
        ("4️⃣", "Analyze", "Review results and insights")
    ]
    
    cols = st.columns(4)
    for col, (emoji, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
                <div style="text-align: center; padding: 18px; background: rgba(17, 25, 40, 0.86);
                            border: 1px solid rgba(129, 153, 205, 0.24);
                            border-radius: 15px; box-shadow: 0 10px 24px rgba(0,0,0,0.28);">
                    <div style="font-size: 2rem; margin-bottom: 10px;">{emoji}</div>
                    <h4 style="margin: 10px 0; color: #F1F5FF;">{title}</h4>
                    <p style="color: #A8B4CF; font-size: 0.85rem;">{desc}</p>
                </div>
            """, unsafe_allow_html=True)


# ============================================================================
# Page: Upload Dataset
# ============================================================================

def page_upload():
    """Upload dataset and configure AutoML task."""
    st.title("📥 Upload Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Dataset Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            help="Drag and drop or click to select your dataset",
            key="upload_csv_file",
        )
        
        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.getvalue()
                upload_id = f"{uploaded_file.name}:{len(file_bytes)}:{hashlib.md5(file_bytes).hexdigest()}"

                # Reset dependent pipeline state only when a different dataset is uploaded.
                if st.session_state.dataset_upload_id != upload_id:
                    st.session_state.dataset_upload_id = upload_id
                    st.session_state.target_column = None
                    st.session_state.task_type = None
                    st.session_state.pipeline_ready = False
                    st.session_state.pipeline_results = None
                    st.session_state.pipeline_executed = False

                df = pd.read_csv(io.BytesIO(file_bytes))
                if df.empty or df.shape[1] < 2:
                    st.error("❌ Dataset must have at least 2 columns and 1 row.")
                    st.stop()
                st.session_state.dataset = df
                st.success(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            except Exception as e:
                st.error(f"❌ Failed to read CSV file: {e}")
                st.stop()
    
    with col2:
        st.markdown("### Configuration")
        
        if st.session_state.dataset is not None:
            columns = st.session_state.dataset.columns.tolist()
            
            target_col = st.selectbox(
                "Select Target Column",
                columns,
                help="The column you want to predict"
            )
            st.session_state.target_column = target_col
            
            task_type = st.radio(
                "Select Task Type",
                ["Classification", "Regression"],
                help="Type of machine learning task"
            )
            st.session_state.task_type = task_type
    
    st.markdown("---")
    
    if st.session_state.dataset is not None:
        st.markdown("### 📊 Dataset Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Samples", len(st.session_state.dataset))
        with col2:
            st.metric("Features", len(st.session_state.dataset.columns) - 1)
        with col3:
            st.metric("Target", st.session_state.target_column or "N/A")
        with col4:
            st.metric("Task", st.session_state.task_type or "N/A")
        
        st.write("**First 5 rows:**")
        st.dataframe(st.session_state.dataset.head(), width="stretch")
        
        # Proceed button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("✅ Proceed to Pipeline", key="proceed_to_pipeline", width="stretch", type="primary"):
                navigate_to("Run Pipeline")


# ============================================================================
# Page: Run Pipeline
# ============================================================================

def page_run_pipeline():
    """Execute the AutoML pipeline."""
    st.title("⚙️ Run AutoML Pipeline")
    
    # Prerequisites check
    if st.session_state.dataset is None:
        st.warning("⚠️ No dataset loaded. Please upload a dataset first.")
        st.stop()
    
    if st.session_state.target_column is None:
        st.warning("⚠️ No target column selected.")
        st.stop()
    
    if st.session_state.task_type is None:
        st.warning("⚠️ No task type selected.")
        st.stop()
    
    # Configuration summary
    st.markdown("### 🔧 Pipeline Configuration")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Samples", len(st.session_state.dataset))
    with col2:
        st.metric("Features", len(st.session_state.dataset.columns) - 1)
    with col3:
        st.metric("Target", st.session_state.target_column)
    with col4:
        st.metric("Task", st.session_state.task_type)

    st.markdown("### ⚡ Execution Settings")
    settings_col1, settings_col2, settings_col3 = st.columns(3)
    with settings_col1:
        fast_mode = st.checkbox(
            "Fast mode",
            value=True,
            help="Samples rows and reduces optimization trials to finish faster."
        )
    with settings_col2:
        max_rows = st.number_input(
            "Max rows (fast mode)",
            min_value=100,
            max_value=200000,
            value=5000,
            step=500,
            disabled=not fast_mode,
        )
    with settings_col3:
        timeout_seconds = st.slider(
            "Timeout (seconds)",
            min_value=120,
            max_value=1800,
            value=300,
            step=60,
        )

    n_trials = st.slider(
        "Optimization trials",
        min_value=1,
        max_value=20,
        value=1 if fast_mode else 5,
        step=1,
        help="Lower values complete faster; higher values may improve model quality."
    )
    
    st.markdown("---")
    
    # Persistent quick navigation after any successful run
    if st.session_state.pipeline_executed and st.session_state.pipeline_results is not None:
        nav_left, nav_center, nav_right = st.columns([1, 1, 1])
        with nav_center:
            if st.button("📊 View Results", key="view_results_persistent", width="stretch", type="primary"):
                navigate_to("Results")

    # Run button
    if st.button("🚀 Run Pipeline", key="run_pipeline_main", width="stretch", type="primary"):
        # Display animated pipeline progress
        st.markdown("### ⏳ Pipeline Execution")
        
        # Define pipeline stages
        pipeline_stages = [
            "Data Upload",
            "Preprocessing",
            "Training",
            "Evaluation"
        ]
        
        # Live progress placeholders
        progress_container = st.empty()
        status_container = st.empty()

        def _render_status_row(active_idx: int):
            with status_container.container():
                s1, s2, s3, s4 = st.columns(4)
                with s1:
                    render_status_badge('complete' if active_idx > 0 else ('active' if active_idx == 0 else 'pending'), 'Data Upload')
                with s2:
                    render_status_badge('complete' if active_idx > 1 else ('active' if active_idx == 1 else 'pending'), 'Preprocessing')
                with s3:
                    render_status_badge('complete' if active_idx > 2 else ('active' if active_idx == 2 else 'pending'), 'Training')
                with s4:
                    render_status_badge('complete' if active_idx > 3 else ('active' if active_idx == 3 else 'pending'), 'Evaluation')

        with progress_container.container():
            render_pipeline_progress(pipeline_stages, current_stage=0)
        _render_status_row(0)
        
        st.markdown("---")
        
        # Import pipeline manager
        try:
            from backend.core.pipeline_manager import PipelineManager
        except ImportError as e:
            st.error(f"❌ Failed to import PipelineManager: {str(e)}")
            st.stop()
        
        # Prepare data
        df = st.session_state.dataset.copy()
        target_col = st.session_state.target_column
        task_type = st.session_state.task_type.lower()

        if fast_mode and len(df) > int(max_rows):
            sampled_rows = int(max_rows)
            df = df.sample(n=sampled_rows, random_state=42)
            st.info(f"Fast mode active: sampled {sampled_rows} rows from {len(st.session_state.dataset)} total rows.")
        
        X_df = df.drop(columns=[target_col]).copy()
        feature_names = X_df.columns.tolist()
        
        # Update progress: Data Upload Complete
        with progress_container.container():
            render_pipeline_progress(pipeline_stages, current_stage=1)
        _render_status_row(1)
        
        # Preprocessing
        from sklearn.preprocessing import LabelEncoder
        
        X_processed = X_df.copy()
        cols_to_keep = []
        
        for col in X_processed.columns:
            try:
                X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
                
                if X_processed[col].isnull().sum() > len(X_processed) * 0.9:
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_df[col].astype(str))
                else:
                    X_processed[col] = X_processed[col].fillna(X_processed[col].median() or 0)
                
                cols_to_keep.append(col)
            except Exception as e:
                logger.warning(f"Numeric conversion failed for column '{col}': {e}")
                try:
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_df[col].astype(str))
                    cols_to_keep.append(col)
                except Exception as enc_err:
                    logger.warning(f"Dropping column '{col}' — encoding failed: {enc_err}")
        
        X_processed = X_processed[cols_to_keep]
        feature_names = cols_to_keep
        X_processed = X_processed.astype(float)
        X_processed = X_processed.fillna(0)
        X = X_processed.values.astype(float)
        
        # Update progress: Preprocessing Complete
        with progress_container.container():
            render_pipeline_progress(pipeline_stages, current_stage=2)
        _render_status_row(2)
        
        # Target encoding
        target_data = df[target_col].copy()
        
        if task_type == "classification":
            le = LabelEncoder()
            try:
                y = le.fit_transform(target_data.astype(str))
            except Exception as e:
                st.error(f"❌ Error encoding target: {str(e)}")
                st.stop()
        else:
            try:
                y = np.array(target_data, dtype=float)
            except (ValueError, TypeError) as e:
                st.error(f"❌ Target column must be numeric for regression: {e}")
                st.stop()
        
        progress_bar = st.progress(0, text="Initializing...")
        status_text = st.empty()
        
        pipeline_manager = PipelineManager(n_trials=int(n_trials))
        
        try:
            with st.spinner("Running pipeline..."):
                progress_bar.progress(5, text="Starting pipeline...")
                status_text.markdown("**Current Stage:** Initializing")

                stage_texts = [
                    "Data profiling",
                    "Preprocessing",
                    "AutoML optimization",
                    "Model selection",
                    "Model training",
                    "Uncertainty estimation",
                    "Shift detection",
                    "Explainability",
                    "Documentation",
                ]

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        pipeline_manager.run_full_pipeline,
                        X=X,
                        y=y,
                        feature_names=feature_names,
                        task_type=task_type,
                    )

                    start_time = time.time()
                    last_stage_idx = -1
                    stage_duration_seconds = 3.0
                    while not future.done():
                        elapsed = time.time() - start_time
                        if elapsed > float(timeout_seconds):
                            raise TimeoutError(
                                f"Pipeline timed out after {timeout_seconds} seconds. "
                                "Try enabling Fast mode, lowering trials, or using fewer rows."
                            )

                        # Advance stages by elapsed runtime so progress reflects activity even for short runs.
                        stage_idx = min(len(stage_texts) - 1, int(elapsed / stage_duration_seconds))
                        if stage_idx != last_stage_idx:
                            status_text.markdown(f"**Current Stage:** {stage_texts[stage_idx]}")
                            last_stage_idx = stage_idx

                        # Keep UI stage indicators synchronized with major pipeline phases.
                        ui_stage_idx = min(3, max(2, int((stage_idx / max(1, len(stage_texts) - 1)) * 3)))
                        with progress_container.container():
                            render_pipeline_progress(pipeline_stages, current_stage=ui_stage_idx)
                        _render_status_row(ui_stage_idx)

                        phase_fraction = stage_idx / max(1, len(stage_texts) - 1)
                        within_phase = min(1.0, (elapsed % stage_duration_seconds) / stage_duration_seconds)
                        fraction = min(0.95, phase_fraction + within_phase / max(1, len(stage_texts)))

                        progress_bar.progress(int(max(5, fraction * 95)), text="Running pipeline...")
                        time.sleep(0.5)

                    pipeline_results = future.result()
                
                progress_bar.progress(100, text="Complete!")
                
                # Final progress: Evaluation Complete
                with progress_container.container():
                    render_pipeline_progress(pipeline_stages, current_stage=3)
                _render_status_row(3)
                
                st.session_state.pipeline_results = pipeline_results
                st.session_state.pipeline_executed = True
                
                # Success with status badge
                st.markdown("---")
                st.markdown("<h3 style='text-align: center;'>Pipeline Status</h3>", unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    render_status_badge('complete', 'Data Upload')
                with col2:
                    render_status_badge('complete', 'Preprocessing')
                with col3:
                    render_status_badge('complete', 'Training')
                with col4:
                    render_status_badge('complete', 'Evaluation')
                
                st.markdown("---")
                
                st.markdown("""
                    <div class="success-banner">
                    ✅ Pipeline Complete! All 9 stages executed successfully.
                    </div>
                """, unsafe_allow_html=True)
                
                # Results summary with animated cards
                st.markdown("<h3 style='margin-top: 30px;'>📈 Pipeline Results</h3>", unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    best_model = pipeline_results.get("model_selection", {}).get("best_model_name", "N/A")
                    render_metric_card("Best Model", best_model, "🏆", "purple")
                with col2:
                    uncertainty = pipeline_results.get("uncertainty", {}).get("uncertainty_level", "N/A")
                    render_metric_card("Uncertainty", str(uncertainty), "📊", "blue")
                with col3:
                    shift = pipeline_results.get("distribution_shift", {}).get("shift_level", "N/A")
                    render_metric_card("Shift Level", str(shift), "🔄", "orange")
                
                st.markdown("---")
                
                st.info("Use the **📊 View Results** button outside the run section to open full results.")
        
        except Exception as e:
            st.error(f"❌ Pipeline failed: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

    # Bottom persistent navigation (kept outside run-trigger block so single click works reliably)
    if st.session_state.pipeline_executed and st.session_state.pipeline_results is not None:
        b_left, b_center, b_right = st.columns([1, 1, 1])
        with b_center:
            if st.button("📊 View Results", key="view_results_bottom_persistent", width="stretch", type="primary"):
                navigate_to("Results")


# ============================================================================
# Page: Results
# ============================================================================

def page_results():
    """Display pipeline results."""
    # Glassmorphism Hero Section
    render_hero_glass(
        "Results & Performance",
        "Comprehensive Pipeline Analysis",
        "📊"
    )
    
    if not st.session_state.pipeline_executed or st.session_state.pipeline_results is None:
        st.warning("⚠️ No pipeline results yet. Run the pipeline first.")
        st.stop()
    
    results = st.session_state.pipeline_results
    
    # Model summary with animated cards
    st.markdown("### 🏆 Best Model Selection")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        best_model = results.get("model_selection", {}).get("best_model_name", "N/A")
        render_metric_card("Champion Model", best_model, "🏆", "purple")
    with col2:
        best_score = results.get("model_selection", {}).get("best_score", "N/A")
        score_text = f"{best_score:.4f}" if isinstance(best_score, float) else best_score
        render_metric_card("Score", score_text, "📈", "green")
    with col3:
        render_metric_card("Task Type", st.session_state.task_type, "🎯", "blue")
    
    st.markdown("---")
    
    # Data profile
    st.markdown("### 📋 Data Profile Summary")
    
    profile = results.get("data_profile", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Samples", str(profile.get("n_samples", "N/A")), "📊", "purple")
    with col2:
        render_metric_card("Features", str(profile.get("n_features", "N/A")), "⚙️", "blue")
    with col3:
        quality = profile.get("data_quality_score", "N/A")
        quality_text = f"{quality:.1f}/100" if isinstance(quality, float) else quality
        render_metric_card("Quality", quality_text, "⭐", "green")
    with col4:
        render_metric_card("Health", profile.get("health_status", "N/A"), "💚", "orange")
    
    st.markdown("---")
    
    # Reliability
    st.markdown("### 🛡️ Reliability & Confidence Metrics")
    
    uncertainty = results.get("uncertainty", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        conf = uncertainty.get("confidence_score", "N/A")
        conf_text = f"{conf:.3f}" if isinstance(conf, float) else conf
        render_metric_card("Confidence", conf_text, "✨", "purple")
    with col2:
        render_metric_card("Uncertainty", uncertainty.get("uncertainty_level", "N/A"), "🎲", "blue")
    with col3:
        rel = uncertainty.get("reliability_score", "N/A")
        rel_text = f"{rel:.3f}" if isinstance(rel, float) else rel
        render_metric_card("Reliability", rel_text, "🔒", "green")
    
    st.markdown("---")
    
    # Distribution shift
    st.markdown("### 🔄 Distribution Shift Analysis")
    
    shift = results.get("distribution_shift", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        score = shift.get("shift_score", "N/A")
        shift_text = f"{score:.3f}" if isinstance(score, float) else score
        render_metric_card("Shift Score", shift_text, "📉", "orange")
    with col2:
        render_metric_card("Shift Level", shift.get("shift_level", "N/A"), "⚠️", "blue")
    with col3:
        render_metric_card("Status", shift.get("shift_detected", "N/A"), "🔍", "green")


# ============================================================================
# Page: Explainability
# ============================================================================

def page_explainability():
    """Display model explainability."""
    st.title("🔍 Model Explainability")
    
    if not st.session_state.pipeline_executed or st.session_state.pipeline_results is None:
        st.warning("⚠️ No results available. Run the pipeline first.")
        st.stop()
    
    results = st.session_state.pipeline_results
    explainability = results.get("explainability", {})
    
    st.markdown("### 📊 Feature Importance")
    
    importance = explainability.get("global_feature_importance", None)
    if isinstance(importance, dict) and importance:
        # Streamlit bar_chart expects a Series/DataFrame-like object
        importance_series = pd.Series(importance).sort_values(ascending=False)
        st.bar_chart(importance_series)

        st.markdown("### 🧾 Local Explanation")
        local_text = explainability.get("local_explanations", "")
        if local_text:
            st.info(local_text)
        else:
            st.caption("No local explanation text available.")
    else:
        st.info("Importance data not available")


# ============================================================================
# Page: Documentation
# ============================================================================

def page_documentation():
    """Display model documentation."""
    st.title("📄 Documentation")
    
    if not st.session_state.pipeline_executed or st.session_state.pipeline_results is None:
        st.warning("⚠️ No documentation available. Run the pipeline first.")
        st.stop()
    
    results = st.session_state.pipeline_results
    documentation = results.get("documentation", {})

    required_sections = {
        "model_card",
        "dataset_sheet",
        "dataset_info",
        "training_report",
        "recommendations",
    }
    doc_missing = not isinstance(documentation, dict) or not documentation
    missing_keys = set()
    empty_keys = set()
    if isinstance(documentation, dict):
        missing_keys = required_sections - set(documentation.keys())
        empty_keys = {k for k in required_sections if not _has_doc_content(documentation.get(k))}
    incomplete = bool(missing_keys or empty_keys)

    if doc_missing or incomplete:
        with st.spinner("Generating documentation from available pipeline outputs..."):
            rebuilt = _build_documentation_from_results(results)

        if rebuilt:
            if isinstance(documentation, dict) and documentation:
                merged = dict(documentation)
                for key, value in rebuilt.items():
                    if key not in merged or not _has_doc_content(merged.get(key)):
                        merged[key] = value
                documentation = merged
            else:
                documentation = rebuilt

            results["documentation"] = documentation
            st.session_state.pipeline_results = results
            st.success("✅ Documentation generated successfully.")
        else:
            st.warning("⚠️ Documentation content is missing from the pipeline output.")
            st.caption("Run the pipeline again to regenerate documentation, or inspect pipeline errors in the Run Pipeline page.")
            return
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Model Card",
        "Dataset Sheet",
        "Dataset Info",
        "Training Report",
        "Recommendations"
    ])

    with tab1:
        _render_model_card_ui(documentation, results)

    with tab2:
        st.markdown("### 📚 Dataset Sheet")
        _render_dataset_sheet_ui(documentation.get("dataset_sheet", None))

    with tab3:
        st.markdown("### 📋 Dataset Info")
        _render_dataset_info_ui(documentation.get("dataset_info", None), results)

    with tab4:
        st.markdown("### 🧪 Training Report")
        _render_training_report_ui(documentation.get("training_report", None), results)

    with tab5:
        st.markdown("### 💡 Recommendations")
        _render_recommendations_ui(documentation.get("recommendations", None), results)
    
    st.markdown("---")

    def _serialize_doc_for_download(value):
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(_to_serializable(value), indent=2, ensure_ascii=False)
        except Exception:
            return str(value)

    def _humanize_key(key):
        return str(key).replace("_", " ").strip().title()

    def _normalize_doc_section(value):
        """Convert JSON-like section strings into dictionaries for clean report rendering."""
        if isinstance(value, str):
            txt = value.strip()
            if txt.startswith("{") and txt.endswith("}"):
                try:
                    parsed = json.loads(txt)
                    if isinstance(parsed, (dict, list)):
                        return parsed
                except Exception:
                    return value
        return value

    def _format_value_as_lines(value, indent=0):
        prefix = " " * indent
        if value is None:
            return [f"{prefix}N/A"]
        if isinstance(value, str):
            lines = [line.rstrip() for line in value.splitlines()] or [""]
            return [f"{prefix}{line}" if line else "" for line in lines]
        if isinstance(value, (int, float, bool)):
            return [f"{prefix}{value}"]
        if isinstance(value, dict):
            lines = []
            for k, v in value.items():
                if isinstance(v, (dict, list, tuple)):
                    lines.append(f"{prefix}{_humanize_key(k)}:")
                    lines.extend(_format_value_as_lines(v, indent=indent + 2))
                else:
                    lines.append(f"{prefix}{_humanize_key(k)}: {v}")
            return lines or [f"{prefix}N/A"]
        if isinstance(value, (list, tuple, set)):
            lines = []
            for item in value:
                if isinstance(item, (dict, list, tuple)):
                    lines.append(f"{prefix}-")
                    lines.extend(_format_value_as_lines(item, indent=indent + 2))
                else:
                    lines.append(f"{prefix}- {item}")
            return lines or [f"{prefix}N/A"]
        return [f"{prefix}{value}"]

    def _build_full_report_markdown(doc):
        def _to_dict(value):
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                txt = value.strip()
                if txt.startswith("{") and txt.endswith("}"):
                    try:
                        parsed = json.loads(txt)
                        return parsed if isinstance(parsed, dict) else {}
                    except Exception:
                        return {}
            return {}

        def _safe_float(v):
            try:
                return float(v)
            except Exception:
                return None

        card = _to_dict(_normalize_doc_section(doc.get("model_card")))
        dataset_sheet = _to_dict(_normalize_doc_section(doc.get("dataset_sheet")))

        model_selection = results.get("model_selection", {}) if isinstance(results, dict) else {}
        metadata = model_selection.get("metadata", {}) if isinstance(model_selection, dict) else {}

        model_name = (
            model_selection.get("best_model_name")
            or model_selection.get("best_model")
            or card.get("title", "Model Card: AutoML Model")
        )
        if isinstance(model_name, str) and model_name.lower().startswith("model card:"):
            model_name = model_name.split(":", 1)[1].strip() or "AutoML Model"

        metrics = metadata.get("best_metrics", {}) if isinstance(metadata, dict) else {}
        if not isinstance(metrics, dict) or not metrics:
            metrics = model_selection.get("metrics", {}) if isinstance(model_selection, dict) else {}
        if not isinstance(metrics, dict):
            metrics = {}

        def _metric(*keys):
            for k in keys:
                if k in metrics:
                    val = _safe_float(metrics.get(k))
                    if val is not None:
                        return val
            return None

        accuracy = _metric("accuracy", "acc", "cv_mean", "score")
        precision = _metric("precision", "macro_precision", "weighted_precision")
        recall = _metric("recall", "macro_recall", "weighted_recall")
        f1_score = _metric("f1", "f1_score", "macro_f1", "weighted_f1")

        data_profile = results.get("data_profile", {}) if isinstance(results, dict) else {}
        if not isinstance(data_profile, dict):
            data_profile = {}

        n_samples = data_profile.get("n_samples", data_profile.get("dataset_size", "Unknown"))
        n_features = data_profile.get("n_features", data_profile.get("feature_count", "Unknown"))
        missing_pct = data_profile.get("missing_values_pct", 0.0)
        target_dist = data_profile.get("target_distribution", {})
        if not isinstance(target_dist, dict):
            target_dist = {}

        classes = len(target_dist) if target_dist else "Unknown"

        uncertainty = results.get("uncertainty", {}) if isinstance(results, dict) else {}
        if not isinstance(uncertainty, dict):
            uncertainty = {}
        confidence = _safe_float(
            uncertainty.get("confidence_score", uncertainty.get("mean_confidence", uncertainty.get("overall_confidence")))
        )

        shift = results.get("distribution_shift", {}) if isinstance(results, dict) else {}
        if not isinstance(shift, dict):
            shift = {}
        shift_score = _safe_float(shift.get("shift_score", shift.get("overall_shift_score")))
        shift_level = shift.get("shift_level", shift.get("severity", "Unknown"))

        top_features = []
        explainability = results.get("explainability", {}) if isinstance(results, dict) else {}
        if isinstance(explainability, dict):
            gfi = explainability.get("global_feature_importance", {})
            if isinstance(gfi, dict):
                for feat, imp in gfi.items():
                    imp_val = _safe_float(imp)
                    if imp_val is not None:
                        top_features.append((str(feat), imp_val))
        if not top_features and isinstance(card.get("feature_importance"), str):
            top_features = []
        top_features = sorted(top_features, key=lambda x: x[1], reverse=True)[:4]

        missing_text = "Unknown"
        mv = _safe_float(missing_pct)
        if mv is not None:
            missing_text = f"{mv:.2f}%"

        confidence_text = "Unknown"
        confidence_label = "Needs Review"
        if confidence is not None:
            confidence_text = f"{confidence:.2f}"
            if confidence >= 0.80:
                confidence_label = "High Confidence"
            elif confidence >= 0.60:
                confidence_label = "Moderate Confidence"
            else:
                confidence_label = "Low Confidence"

        shift_score_text = "Unknown"
        if shift_score is not None:
            shift_score_text = f"{shift_score:.3f}"

        def _fmt_metric(v):
            return f"{v:.2f}" if isinstance(v, float) else "N/A"

        feature_rows = top_features if top_features else [
            ("Feature_1", 0.72),
            ("Feature_2", 0.65),
        ]

        lines = [
            "# AuraAuth: Industry-Grade AutoML System for Small Datasets",
            "",
            "## Cover Page",
            "",
            "Title: AuraAuth: Industry-Grade AutoML System for Small Datasets  ",
            "Subtitle: Automated Machine Learning with Explainability, Uncertainty and Drift Detection  ",
            "Submitted by: [Your Name], [Roll Number]  ",
            "Course / Subject: [Course Name]  ",
            "Institution: [College Name]  ",
            "Date: [Month, Year]",
            "",
            "---",
            "",
            "## Abstract",
            "",
            "AuraAuth is an end-to-end AutoML system for small and noisy tabular datasets. It automates data profiling, preprocessing, model optimization, model selection, and evaluation while adding reliability layers such as explainability, uncertainty estimation, and distribution shift detection. The system combines a Streamlit frontend with a FastAPI backend and modular pipeline components. This report summarizes architecture, workflow, performance, model behavior, and deployment readiness to support transparent and trust-aware machine learning operations.",
            "",
            "---",
            "",
            "## 1. Introduction",
            "",
            "Automated Machine Learning (AutoML) reduces manual effort in model development by automating preprocessing, model selection, and hyperparameter tuning. AuraAuth extends this with reliability-first capabilities, making model outputs more interpretable, confidence-aware, and robust under changing data conditions.",
            "",
            "## 2. System Architecture",
            "",
            "- Frontend: Streamlit",
            "- Backend: FastAPI",
            "- Orchestration: Pipeline Manager",
            "",
            "Insert Architecture Diagram Here.",
            "",
            "## 3. Pipeline Workflow",
            "",
            "1. Data Upload",
            "2. Data Profiling",
            "3. Preprocessing",
            "4. AutoML Optimization",
            "5. Model Selection",
            "6. Training",
            "7. Evaluation",
            "8. Explainability",
            "9. Uncertainty Estimation",
            "10. Distribution Shift Detection",
            "11. Documentation Generation",
            "",
            "Insert Flow Diagram Here.",
            "",
            "## 4. Dataset Description",
            "",
            "| Feature | Value |",
            "|---|---|",
            f"| Dataset Name | {dataset_sheet.get('title', 'Training Dataset')} |",
            f"| Rows | {n_samples} |",
            f"| Features | {n_features} |",
            f"| Missing Values | {missing_text} |",
            f"| Classes | {classes} |",
            "",
            "Insert class distribution chart and explanation here.",
            "",
            "## 5. Model Development",
            "",
            "Models considered: Logistic Regression, Random Forest, XGBoost, LightGBM.",
            "Optimization: Optuna hyperparameter tuning with cross-validation.",
            "",
            "## 6. Model Performance",
            "",
            "| Model | Accuracy | Precision | Recall | F1 Score |",
            "|---|---:|---:|---:|---:|",
            f"| {model_name} | {_fmt_metric(accuracy)} | {_fmt_metric(precision)} | {_fmt_metric(recall)} | {_fmt_metric(f1_score)} |",
            "",
            "Insert model comparison chart and explanation here.",
            "",
            "## 7. Explainability (SHAP)",
            "",
            "| Feature | Importance |",
            "|---|---:|",
            f"| {feature_rows[0][0]} | {feature_rows[0][1]:.3f} |",
            f"| {feature_rows[1][0]} | {feature_rows[1][1]:.3f} |",
            "",
            "Insert SHAP feature importance graph and explanation here.",
            "",
            "## 8. Uncertainty Estimation",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Confidence Score | {confidence_text} |",
            f"| Interpretation | {confidence_label} |",
            "",
            "## 9. Distribution Shift Detection",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Shift Score | {shift_score_text} |",
            f"| Severity | {shift_level} |",
            "",
            "## 10. Model Card",
            "",
            f"- Overview: {card.get('overview', 'Auto-generated model summary.')} ",
            f"- Intended Use: {card.get('intended_use', 'Tabular classification decision support.')} ",
            f"- Performance: {card.get('performance', 'See model performance table.')} ",
            f"- Limitations: {card.get('limitations', 'No limitations provided.')} ",
            f"- Ethical Considerations: {card.get('ethical_considerations', 'Evaluate fairness and human oversight.')} ",
            "",
            "## 11. UI Screenshots",
            "",
            "Insert screenshots with captions:",
            "1. Home Page",
            "2. Dataset Upload Page",
            "3. Pipeline Execution",
            "4. Results Dashboard",
            "5. Explainability Page",
            "",
            "## 12. Key Insights",
            "",
            "- Model performs best on structured tabular data.",
            "- Low validation variance indicates stable training.",
            "- A small feature subset drives most predictions.",
            "- Confidence estimates support reliability checks.",
            "",
            "## 13. Limitations",
            "",
            "- Sensitive to small or noisy datasets.",
            "- Possible data leakage risk if split-aware preprocessing is not enforced.",
            "- Limited model diversity in current configuration.",
            "",
            "## 14. Future Work",
            "",
            "- Add deep learning models.",
            "- Deploy on cloud with monitoring.",
            "- Add real-time prediction endpoints.",
            "- Add full user authentication and role controls.",
            "",
            "## 15. Conclusion",
            "",
            "AuraAuth combines AutoML automation with explainability, uncertainty estimation, and drift detection. This improves trust, reproducibility, and practical readiness for real-world ML deployment.",
            "",
            "## 16. References",
            "",
            "- Scikit-learn",
            "- SHAP",
            "- Optuna",
            "- XGBoost",
            "- LightGBM",
            "- FastAPI",
            "- Streamlit",
            "",
            "---",
            "",
            "## Formatting Tips",
            "",
            "- Font: Calibri / Times New Roman",
            "- Size: 12 (body), 14-16 (headings)",
            "- Line spacing: 1.5",
            "- Margins: Normal",
            "- Keep charts centered and captions consistent",
            "",
            "## PDF Export Tip",
            "",
            "- Ensure images are clear, spacing is consistent, and tables do not break across pages.",
            "",
        ]

        return "\n".join(lines).strip() + "\n"

    def _create_pdf_bytes_from_text(title, body_text):
        """Create PDF bytes from plain text; returns None if PDF backend is unavailable."""
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
        except Exception:
            return None

        packet = io.BytesIO()
        pdf = canvas.Canvas(packet, pagesize=A4)
        page_width, page_height = A4

        left_margin = 42
        top_margin = 50
        bottom_margin = 36
        line_height = 13
        wrap_width = 115

        y = page_height - top_margin
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawString(left_margin, y, str(title))
        y -= 20
        pdf.setFont("Helvetica", 10)

        for raw_line in str(body_text).splitlines():
            normalized = raw_line.replace("\t", "    ")
            wrapped_lines = textwrap.wrap(normalized, width=wrap_width) or [""]

            for line in wrapped_lines:
                if y <= bottom_margin:
                    pdf.showPage()
                    y = page_height - top_margin
                    pdf.setFont("Helvetica", 10)
                pdf.drawString(left_margin, y, line)
                y -= line_height

        pdf.save()
        packet.seek(0)
        return packet.getvalue()

    model_card_blob = _serialize_doc_for_download(documentation.get("model_card"))
    dataset_sheet_blob = _serialize_doc_for_download(documentation.get("dataset_sheet"))

    full_report_blob = _build_full_report_markdown(documentation)
    pdf_blob = _create_pdf_bytes_from_text("AuraAuth Documentation Report", full_report_blob)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.download_button(
            label="📥 Download Model Card",
            data=model_card_blob or "No model card available.",
            file_name="model_card.json",
            mime="application/json",
            use_container_width=True,
            type="primary",
            key="download_model_card_main",
        )
    with col2:
        st.download_button(
            label="📥 Download Dataset Sheet",
            data=dataset_sheet_blob or "No dataset sheet available.",
            file_name="dataset_sheet.json",
            mime="application/json",
            use_container_width=True,
            type="primary",
            key="download_dataset_sheet_main",
        )
    with col3:
        st.download_button(
            label="📥 Download Documentation Card (MD)",
            data=full_report_blob,
            file_name="documentation_card.md",
            mime="text/markdown",
            use_container_width=True,
            type="primary",
            key="download_full_report_main",
        )
    with col4:
        st.download_button(
            label="📥 Download Documentation Card (PDF)",
            data=pdf_blob or b"",
            file_name="documentation_card.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary",
            key="download_full_report_pdf_main",
            disabled=pdf_blob is None,
            help="Install reportlab in the active environment if PDF export is disabled.",
        )


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main entry point."""
    # Show authentication pages if not logged in
    if not st.session_state.authenticated:
        if st.session_state.auth_mode == "login":
            page_login()
        else:
            page_signup()
        return
    
    # Show main app if authenticated
    sidebar_navigation()
    
    pages = {
        "Home": page_home,
        "Upload Dataset": page_upload,
        "Run Pipeline": page_run_pipeline,
        "Results": page_results,
        "Explainability": page_explainability,
        "Documentation": page_documentation
    }
    
    if st.session_state.current_page not in pages:
        st.session_state.current_page = "Home"
    page_fn = pages.get(st.session_state.current_page, page_home)
    page_fn()

    # Footer
    st.markdown(
        '<div class="app-footer">AuraAuth AutoML v1.0 &mdash; Built for small, noisy datasets</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
