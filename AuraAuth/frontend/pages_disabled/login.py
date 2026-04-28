"""Authentication pages for login and sign-up."""

import streamlit as st


def _auth_form(title: str, action_label: str) -> tuple[str, str, bool]:
    st.markdown('<div class="hero-card">', unsafe_allow_html=True)
    st.subheader(title)
    username = st.text_input("Username", key=f"{action_label}_username")
    password = st.text_input("Password", type="password", key=f"{action_label}_password")
    submitted = st.button(action_label, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    return username, password, submitted


def page_login() -> None:
    """Render login page and update session auth state."""
    st.title("Welcome Back")
    _, form_col, _ = st.columns([1, 2, 1])
    with form_col:
        username, password, submitted = _auth_form("Sign in to continue", "Login")

    _, action_col, _ = st.columns([1, 2, 1])
    with action_col:
        if st.button("Create account"):
            st.session_state.auth_mode = "signup"
            st.rerun()

    if submitted:
        if not username or not password:
            st.error("Please enter both username and password.")
            return
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.current_page = "Home"
        st.success("Signed in successfully.")
        st.rerun()


def page_signup() -> None:
    """Render signup page with local-only account simulation."""
    st.title("Create Account")
    _, form_col, _ = st.columns([1, 2, 1])
    with form_col:
        username, password, submitted = _auth_form("Create your AuraAuth account", "Sign up")

    _, action_col, _ = st.columns([1, 2, 1])
    with action_col:
        if st.button("Back to login"):
            st.session_state.auth_mode = "login"
            st.rerun()

    if submitted:
        if len(username.strip()) < 3:
            st.error("Username must be at least 3 characters.")
            return
        if len(password) < 6:
            st.error("Password must be at least 6 characters.")
            return
        st.session_state.authenticated = True
        st.session_state.username = username.strip()
        st.session_state.current_page = "Home"
        st.success("Account created. You are now signed in.")
        st.rerun()
