"""Layout primitives for navigation and global controls."""

import streamlit as st


_PAGES = [
    "Home",
    "Upload Dataset",
    "Run Pipeline",
    "Results",
    "Explainability",
    "Documentation",
]


def sidebar_navigation() -> None:
    """Render sidebar navigation and auth controls."""
    with st.sidebar:
        st.title("AuraAuth")
        st.caption("AutoML for noisy, small datasets")

        if st.session_state.get("username"):
            st.success(f"Signed in as {st.session_state['username']}")

        current = st.session_state.get("current_page", "Home")
        selected = st.radio("Navigate", _PAGES, index=_PAGES.index(current) if current in _PAGES else 0)
        st.session_state.current_page = selected

        st.divider()
        if st.button("Sign out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.auth_mode = "login"
            st.session_state.current_page = "Home"
            st.rerun()
