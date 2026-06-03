import streamlit as st
import hashlib
from streamlit.web.server.websocket_headers import _get_websocket_headers

PASSWORDS = {
    "anker_admin": hashlib.sha256("anker2026".encode()).hexdigest(),
    "store_manager": hashlib.sha256("store2026".encode()).hexdigest(),
}


def _is_local() -> bool:
    try:
        headers = _get_websocket_headers()
        if headers:
            host = headers.get("Host", "")
            if host.startswith("localhost") or host.startswith("127.0.0.1"):
                return True
    except Exception:
        pass
    import os

    return os.environ.get("STREAMLIT_LOCAL_BYPASS") == "1"


def check_password() -> bool:
    if _is_local():
        return True

    if st.session_state.get("password_correct", False):
        return True

    def password_entered():
        username = st.session_state.get("username", "")
        password = st.session_state.get("password", "")
        if username in PASSWORDS:
            hashed = hashlib.sha256(password.encode()).hexdigest()
            if hashed == PASSWORDS[username]:
                st.session_state["password_correct"] = True
                st.session_state["authenticated_user"] = username
                del st.session_state["password"]
                return
        st.session_state["password_correct"] = False

    st.markdown("## 🔐 Anker Store 分析ダッシュボード")
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.text_input("ユーザー名", key="username")
        st.text_input("パスワード", type="password", key="password")
        if st.button("ログイン", use_container_width=True):
            password_entered()
            if st.session_state.get("password_correct", False):
                st.success("✅ ログイン成功")
                st.rerun()
            else:
                st.error("❌ ユーザー名またはパスワードが違います")
    st.markdown("---")
    st.caption("🔒 このダッシュボードは社内限定です")
    return False
