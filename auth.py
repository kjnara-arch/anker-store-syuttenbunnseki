import streamlit as st
import hashlib

PASSWORDS = {
    "anker_admin": hashlib.sha256("anker2026".encode()).hexdigest(),
    "store_manager": hashlib.sha256("store2026".encode()).hexdigest(),
}


def check_password() -> bool:
    if st.session_state.get("password_correct", False):
        return True

    st.title("🔐 Anker Store 分析ダッシュボード")
    st.markdown("---")

    username = st.text_input("ユーザー名", key="username")
    password = st.text_input("パスワード", type="password", key="password")

    if st.button("ログイン", use_container_width=True, type="primary"):
        if username in PASSWORDS:
            hashed = hashlib.sha256(password.encode()).hexdigest()
            if hashed == PASSWORDS[username]:
                st.session_state["password_correct"] = True
                st.session_state["authenticated_user"] = username
                st.rerun()
        st.error(" ユーザー名またはパスワードが違います")

    st.markdown("---")
    st.caption(" このダッシュボードは社内限定です")
    return False
