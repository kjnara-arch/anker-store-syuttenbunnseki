# ☁️ Streamlit Cloud デプロイ手順

## 前提条件

- GitHubアカウント
- Streamlit Cloudアカウント（無料）

## 📋 手順

### 1. GitHubリポジトリ作成

```bash
cd store_analytics_dashboard

# Git初期化
git init
git add .
git commit -m "Initial commit: Anker Store分析ダッシュボード"

# GitHubで新規リポジトリ作成後
git remote add origin https://github.com/YOUR_USERNAME/anker-store-analytics.git
git branch -M main
git push -u origin main
```

### 2. Streamlit Cloud設定

1. https://streamlit.io/cloud にアクセス
2. 「Sign up」→ GitHubアカウントで認証
3. 「New app」をクリック

### 3. デプロイ設定

#### Repository設定
- **Repository**: `YOUR_USERNAME/anker-store-analytics`
- **Branch**: `main`
- **Main file path**: `app.py`

#### Advanced settings（オプション）
- **Python version**: 3.9以上

### 4. デプロイ実行

「Deploy!」ボタンをクリック

→ 2-3分でデプロイ完了

→ 公開URL取得: `https://YOUR_APP_NAME.streamlit.app`

## 🔒 セキュリティ注意

### データの機密性

現在のデータには以下が含まれます：
- 店舗名
- 売上実績
- 立地情報

**推奨事項：**

#### オプション1: Private リポジトリ（推奨）
GitHubをPrivateリポジトリにして、アクセス制限

#### オプション2: 認証機能追加
Streamlitの認証機能を追加:

```python
# app.py の先頭に追加
import streamlit as st

def check_password():
    def password_entered():
        if st.session_state["password"] == "YOUR_PASSWORD":
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("パスワード", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("パスワード", type="password", on_change=password_entered, key="password")
        st.error("パスワードが違います")
        return False
    else:
        return True

if not check_password():
    st.stop()
```

#### オプション3: データ匿名化
店舗名を匿名化:

```python
# 店舗名を「店舗A」「店舗B」に変換
df['店舗名'] = [f'店舗{chr(65+i)}' for i in range(len(df))]
```

## 🔄 更新方法

### ローカルで変更後

```bash
git add .
git commit -m "機能追加: XXX"
git push origin main
```

→ Streamlit Cloudが自動で再デプロイ

### 手動デプロイ

Streamlit Cloud管理画面から「Reboot app」

## 📊 使用制限（無料プラン）

- **アプリ数**: 1つ
- **リソース**: 1GB RAM
- **実行時間**: 無制限
- **アクセス**: パブリックアクセス可

※有料プラン（Starter: $20/月）で制限解除

## 🐛 トラブルシューティング

### デプロイエラー: requirements.txt

バージョン指定を緩和:

```txt
streamlit>=1.31.0
pandas>=2.1.0
statsmodels>=0.14.0
```

### メモリ不足エラー

`@st.cache_resource` でモデルをキャッシュ（既に実装済み）

### アクセスが遅い

無料プランは起動に時間がかかる（初回アクセス時）

## 📧 サポート

Streamlit Community: https://discuss.streamlit.io/

## 🎉 デプロイ完了後

URLを社内Slack/Notionで共有:

```
🎉 Anker Store 出店分析ダッシュボード公開！

URL: https://YOUR_APP_NAME.streamlit.app

機能:
- 📊 重回帰分析結果
- 🔮 新規出店シミュレーター
- 🏬 既存店ポテンシャル評価
- 📈 感度分析
```
