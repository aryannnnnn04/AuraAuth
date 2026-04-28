"""Styling helpers for the Streamlit frontend."""

import streamlit as st


def inject_css() -> None:
    """Inject app-wide CSS variables and component styles."""
    st.markdown(
        """
        <style>
            :root {
                --bg: #0e1117;
                --surface: #1c1f26;
                --surface-soft: #232833;
                --text: #ffffff;
                --muted: #aeb7c9;
                --accent: #00ff9f;
                --accent-dark: #00cc7f;
                --danger: #ff6b6b;
                --ok: #55e27b;
            }

            .stApp {
                background: var(--bg);
                color: var(--text);
            }

            .stApp .main .block-container {
                max-width: 1200px;
                padding-top: 1.6rem;
                padding-bottom: 2.4rem;
            }

            .hero-card {
                border: 1px solid rgba(0, 255, 159, 0.35);
                border-radius: 14px;
                padding: 1rem 1.2rem;
                background: var(--surface);
                box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
                margin-bottom: 0.8rem;
            }

            .stTextInput > div > div > input,
            .stSelectbox > div > div,
            .stNumberInput input,
            .stTextArea textarea {
                background: var(--surface-soft) !important;
                color: var(--text) !important;
                border: 1px solid rgba(0, 255, 159, 0.28) !important;
                border-radius: 10px !important;
            }

            .stTextInput > label,
            .stSelectbox > label,
            .stNumberInput > label,
            .stTextArea > label {
                color: var(--muted) !important;
                font-weight: 600;
            }

            .stButton > button {
                background: linear-gradient(135deg, var(--accent) 0%, #6fffc7 100%);
                color: #07251b;
                border: 0;
                border-radius: 10px;
                font-weight: 700;
                min-height: 2.6rem;
                transition: transform 0.12s ease, box-shadow 0.12s ease;
                box-shadow: 0 6px 18px rgba(0, 255, 159, 0.25);
            }

            .stButton > button:hover {
                transform: translateY(-1px);
                box-shadow: 0 9px 22px rgba(0, 255, 159, 0.3);
            }

            .app-footer {
                margin-top: 1.5rem;
                color: var(--muted);
                font-size: 0.9rem;
                text-align: center;
            }

            .metric-good {
                color: var(--ok);
                font-weight: 600;
            }

            .metric-bad {
                color: var(--danger);
                font-weight: 600;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
