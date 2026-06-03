"""Streamlit店舗分析ダッシュボード"""

import streamlit as st
import pandas as pd
import plotly.express as px
from regression_model import StoreRegressionModel, DecisionTreeModel
import base64
from auth import check_password

st.set_page_config(page_title="Anker Store 出店分析", layout="wide", page_icon="🏪")

# 認証チェック
if not check_password():
    st.stop()


@st.cache_resource
def load_model_and_data(model_type="ols"):
    model = StoreRegressionModel(model_type=model_type)
    df = model.load_data()
    X, y_log, store_names = model.prepare_features(df)
    model.fit(X, y_log)
    eval_df = model.evaluate(X, y_log, store_names, df["平均売上"].values)
    return model, df, eval_df


@st.cache_resource
def load_decision_tree_model():
    tree_model = DecisionTreeModel(max_depth=5, min_samples_split=5, min_samples_leaf=3)
    df = tree_model.load_data()
    X, y_log, store_names = tree_model.prepare_features(df)
    tree_model.fit(X, y_log)
    eval_df = tree_model.evaluate(X, y_log, store_names, df["平均売上"].values)
    return tree_model, df, eval_df


# サイドバー
st.sidebar.title("🏪 Anker Store 分析")

# モデル選択
st.sidebar.markdown("### 📐 モデル選択")
model_type = st.sidebar.selectbox(
    "分析手法",
    ["ols", "ridge", "lasso"],
    format_func=lambda x: {
        "ols": "通常OLS（基本16変数）",
        "ridge": "Ridge回帰（正則化・25変数）",
        "lasso": "Lasso回帰（変数選択・25変数）",
    }[x],
    help="Ridge/Lassoは過学習を防ぎながら多数の変数を扱えます",
)

model, df, eval_df = load_model_and_data(model_type)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "メニュー",
    [
        "📊 分析結果",
        "🔮 新規出店シミュレーター",
        "🏬 既存店ポテンシャル",
        "📈 感度分析",
        "🌳 決定木分析",
    ],
)

if page == "📊 分析結果":
    st.title("📊 重回帰分析結果")

    # モデル情報
    model_name = {
        "ols": "通常OLS回帰",
        "ridge": "Ridge回帰（正則化）",
        "lasso": "Lasso回帰（変数選択）",
    }[model_type]
    st.info(f"🔧 使用モデル: **{model_name}**")

    stats = model.get_summary_stats()

    if model_type == "ols":
        col1, col2, col3 = st.columns(3)
        col1.metric("決定係数 R²", f"{stats['R二乗']:.3f}")
        col2.metric("調整済みR²", f"{stats['調整済みR二乗']:.3f}")
        col3.metric("F統計量", f"{stats['F統計量']:.2f}")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("決定係数 R²", f"{stats['R二乗']:.3f}")
        col2.metric("調整済みR²", f"{stats['調整済みR二乗']:.3f}")
        col3.metric("変数数", f"{stats['変数数']}個")
        col4.metric("正則化α", f"{stats['正則化パラメータ(alpha)']:.4f}")

    st.subheader("🎯 変数の影響度（標準化係数）")
    importance_df = model.get_feature_importance()

    if model_type == "ols":
        st.info("**有意水準**: p値 < 0.05 で統計的に有意（✓マーク）")
    else:
        st.info("**Ridge/Lasso回帰**: 正則化により全変数を考慮（p値は計算されません）")

    fig = px.bar(
        importance_df,
        x="標準化係数",
        y="変数",
        orientation="h",
        color="標準化係数",
        color_continuous_scale="RdBu_r",
        title="各変数が売上に与える影響",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(importance_df, use_container_width=True)

    st.subheader("📈 実測値 vs 予測値")
    fig2 = px.scatter(
        eval_df,
        x="予測値",
        y="実測値",
        hover_data=["店舗名"],
        trendline="ols",
        title="予測精度",
    )
    fig2.add_shape(
        type="line",
        x0=eval_df["予測値"].min(),
        y0=eval_df["予測値"].min(),
        x1=eval_df["予測値"].max(),
        y1=eval_df["予測値"].max(),
        line=dict(color="red", dash="dash"),
    )
    st.plotly_chart(fig2, use_container_width=True)

elif page == "🔮 新規出店シミュレーター":
    st.title("🔮 新規出店シミュレーター")
    st.write("条件を入力して売上予測を確認")

    col1, col2, col3 = st.columns(3)

    with col1:
        館売上規模 = st.selectbox("館売上規模", [1, 2, 3, 4, 5], help="1=小, 5=大")
        自店坪数 = st.number_input("自店坪数", 5.0, 100.0, 25.0)
        間口スコア = st.selectbox("間口スコア", [1, 2, 3, 4, 5])
        KDDI人流数値 = st.number_input("KDDI人流数値", 0, 3000000, 500000)
        フロア評価 = st.selectbox("フロア評価", [0.5, 0.8, 1.0])

    with col2:
        視認性 = st.selectbox("視認性", [1, 2, 3, 4, 5])
        動線スコア = st.selectbox("動線スコア", [1, 2, 3, 4, 5])
        商圏年収 = st.selectbox("商圏年収", [1, 2, 3, 4, 5])
        競合店数 = st.selectbox("競合店数", [0, 1, 2, 3, 4, 5])
        インバウンド = st.selectbox("インバウンド", [1, 2, 3, 4, 5])

    with col3:
        st.write("**立地タイプ（1つ選択）**")
        立地 = st.radio(
            "", ["郊外商業", "アウトレット", "駅ビル駅近", "量販店内", "路面店"]
        )
        郊外商業 = 1 if 立地 == "郊外商業" else 0
        アウトレット = 1 if 立地 == "アウトレット" else 0
        駅ビル駅近 = 1 if 立地 == "駅ビル駅近" else 0
        量販店内 = 1 if 立地 == "量販店内" else 0
        路面店 = 1 if 立地 == "路面店" else 0

        量販店近接 = st.selectbox("量販店近接", [0, 1])

    # 基本16変数のみのDataFrame作成
    input_data = pd.DataFrame(
        [
            {
                "館売上規模": 館売上規模,
                "自店坪数": 自店坪数,
                "間口スコア": 間口スコア,
                "KDDI人流数値": KDDI人流数値,
                "フロア評価": フロア評価,
                "視認性": 視認性,
                "動線スコア": 動線スコア,
                "郊外商業": 郊外商業,
                "アウトレット": アウトレット,
                "駅ビル駅近": 駅ビル駅近,
                "量販店内": 量販店内,
                "路面店": 路面店,
                "量販店近接": 量販店近接,
                "商圏年収": 商圏年収,
                "競合店数": 競合店数,
                "インバウンド": インバウンド,
            }
        ]
    )

    # predict内で特徴量エンジニアリングが実行される
    predicted_sales = model.predict(input_data)[0]

    st.success(f"### 📊 予測月間売上: ¥{predicted_sales:,.0f}")

    avg_sales = df["平均売上"].mean()
    diff_pct = ((predicted_sales - avg_sales) / avg_sales) * 100
    st.metric("既存店平均との差", f"{diff_pct:+.1f}%")

    similar = eval_df.iloc[(eval_df["予測値"] - predicted_sales).abs().argsort()[:5]]
    st.subheader("類似条件の既存店舗")
    st.dataframe(similar[["店舗名", "実測値", "予測値"]])

elif page == "🏬 既存店ポテンシャル":
    st.title("🏬 既存店舗ポテンシャル評価")

    # ポテンシャル売上列を追加
    eval_df_enhanced = eval_df.copy()
    eval_df_enhanced["ポテンシャル売上"] = eval_df_enhanced["予測値"]
    eval_df_enhanced["現在売上"] = eval_df_enhanced["実測値"]
    eval_df_enhanced["未達成額"] = (
        eval_df_enhanced["ポテンシャル売上"] - eval_df_enhanced["現在売上"]
    )
    eval_df_enhanced["ポテンシャル達成率"] = (
        eval_df_enhanced["現在売上"] / eval_df_enhanced["ポテンシャル売上"] * 100
    )
    eval_df_enhanced["改善余地"] = eval_df_enhanced.apply(
        lambda x: "大"
        if x["未達成額"] > 5000000
        else ("中" if x["未達成額"] > 2000000 else "小"),
        axis=1,
    )

    # サマリー指標
    col1, col2, col3, col4 = st.columns(4)
    総ポテンシャル = eval_df_enhanced["ポテンシャル売上"].sum()
    総実績 = eval_df_enhanced["現在売上"].sum()
    総未達成額 = 総ポテンシャル - 総実績
    全体達成率 = 総実績 / 総ポテンシャル * 100

    col1.metric("全店舗ポテンシャル売上", f"¥{総ポテンシャル / 100000000:.1f}億円")
    col2.metric("全店舗実績売上", f"¥{総実績 / 100000000:.1f}億円")
    col3.metric(
        "改善余地（未達成額）",
        f"¥{総未達成額 / 100000000:.1f}億円",
        delta=f"{総未達成額 / 10000:.0f}万円/月",
    )
    col4.metric("ポテンシャル達成率", f"{全体達成率:.1f}%")

    # 評価カテゴリ別
    st.subheader("📊 評価カテゴリ別")
    col1, col2, col3 = st.columns(3)
    過小評価 = len(eval_df_enhanced[eval_df_enhanced["評価"] == "過小評価"])
    適正 = len(eval_df_enhanced[eval_df_enhanced["評価"] == "適正"])
    過大評価 = len(eval_df_enhanced[eval_df_enhanced["評価"] == "過大評価"])

    col1.metric("🔥 好調（実績>ポテンシャル）", f"{過小評価}店舗")
    col2.metric("✅ 適正", f"{適正}店舗")
    col3.metric("⚠️ 要改善（実績<ポテンシャル）", f"{過大評価}店舗")

    # 散布図：ポテンシャル売上 vs 現在売上
    fig = px.scatter(
        eval_df_enhanced,
        x="ポテンシャル売上",
        y="現在売上",
        color="改善余地",
        size=abs(eval_df_enhanced["未達成額"]),
        hover_data=["店舗名", "ポテンシャル達成率", "未達成額"],
        color_discrete_map={"大": "red", "中": "orange", "小": "green"},
        title="ポテンシャル売上 vs 現在売上",
    )
    fig.add_shape(
        type="line",
        x0=eval_df_enhanced["ポテンシャル売上"].min(),
        y0=eval_df_enhanced["ポテンシャル売上"].min(),
        x1=eval_df_enhanced["ポテンシャル売上"].max(),
        y1=eval_df_enhanced["ポテンシャル売上"].max(),
        line=dict(color="blue", dash="dash"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 改善インパクトランキング
    st.subheader("💰 改善インパクトランキング（未達成額上位）")
    改善余地大 = (
        eval_df_enhanced[eval_df_enhanced["未達成額"] > 0]
        .sort_values("未達成額", ascending=False)
        .head(10)
    )
    改善余地大_表示 = 改善余地大[
        [
            "店舗名",
            "現在売上",
            "ポテンシャル売上",
            "未達成額",
            "ポテンシャル達成率",
            "改善余地",
        ]
    ].copy()
    改善余地大_表示["現在売上"] = 改善余地大_表示["現在売上"].apply(
        lambda x: f"¥{x:,.0f}"
    )
    改善余地大_表示["ポテンシャル売上"] = 改善余地大_表示["ポテンシャル売上"].apply(
        lambda x: f"¥{x:,.0f}"
    )
    改善余地大_表示["未達成額"] = 改善余地大_表示["未達成額"].apply(
        lambda x: f"¥{x:,.0f}"
    )
    改善余地大_表示["ポテンシャル達成率"] = 改善余地大_表示["ポテンシャル達成率"].apply(
        lambda x: f"{x:.1f}%"
    )
    st.dataframe(改善余地大_表示, use_container_width=True)

    # タブ表示
    tab1, tab2, tab3, tab4 = st.tabs(
        ["⚠️ 要改善店舗", "🔥 好調店舗", "📋 全店舗詳細", "📊 改善余地分布"]
    )

    with tab1:
        st.write("**ポテンシャル未達成店舗（改善余地あり）**")
        要改善 = eval_df_enhanced[eval_df_enhanced["未達成額"] > 0].sort_values(
            "未達成額", ascending=False
        )
        要改善_表示 = 要改善[
            [
                "店舗名",
                "現在売上",
                "ポテンシャル売上",
                "未達成額",
                "ポテンシャル達成率",
                "改善余地",
            ]
        ].copy()
        要改善_表示["現在売上"] = 要改善_表示["現在売上"].apply(lambda x: f"¥{x:,.0f}")
        要改善_表示["ポテンシャル売上"] = 要改善_表示["ポテンシャル売上"].apply(
            lambda x: f"¥{x:,.0f}"
        )
        要改善_表示["未達成額"] = 要改善_表示["未達成額"].apply(lambda x: f"¥{x:,.0f}")
        要改善_表示["ポテンシャル達成率"] = 要改善_表示["ポテンシャル達成率"].apply(
            lambda x: f"{x:.1f}%"
        )
        st.dataframe(要改善_表示, use_container_width=True)

    with tab2:
        st.write("**ポテンシャル超過達成店舗（好調店舗）**")
        好調 = eval_df_enhanced[eval_df_enhanced["未達成額"] < 0].sort_values(
            "未達成額"
        )
        好調_表示 = 好調[
            ["店舗名", "現在売上", "ポテンシャル売上", "未達成額", "ポテンシャル達成率"]
        ].copy()
        好調_表示["現在売上"] = 好調_表示["現在売上"].apply(lambda x: f"¥{x:,.0f}")
        好調_表示["ポテンシャル売上"] = 好調_表示["ポテンシャル売上"].apply(
            lambda x: f"¥{x:,.0f}"
        )
        好調_表示["未達成額"] = 好調_表示["未達成額"].apply(lambda x: f"¥{x:,.0f}")
        好調_表示["ポテンシャル達成率"] = 好調_表示["ポテンシャル達成率"].apply(
            lambda x: f"{x:.1f}%"
        )
        st.dataframe(好調_表示, use_container_width=True)

    with tab3:
        全店舗_表示 = eval_df_enhanced[
            [
                "店舗名",
                "現在売上",
                "ポテンシャル売上",
                "未達成額",
                "ポテンシャル達成率",
                "改善余地",
                "評価",
            ]
        ].copy()
        全店舗_表示["現在売上"] = 全店舗_表示["現在売上"].apply(lambda x: f"¥{x:,.0f}")
        全店舗_表示["ポテンシャル売上"] = 全店舗_表示["ポテンシャル売上"].apply(
            lambda x: f"¥{x:,.0f}"
        )
        全店舗_表示["未達成額"] = 全店舗_表示["未達成額"].apply(lambda x: f"¥{x:,.0f}")
        全店舗_表示["ポテンシャル達成率"] = 全店舗_表示["ポテンシャル達成率"].apply(
            lambda x: f"{x:.1f}%"
        )
        st.dataframe(全店舗_表示, use_container_width=True)

    with tab4:
        fig_dist = px.histogram(
            eval_df_enhanced,
            x="未達成額",
            nbins=20,
            title="未達成額の分布",
            labels={"未達成額": "未達成額（円）", "count": "店舗数"},
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        fig_achieve = px.histogram(
            eval_df_enhanced,
            x="ポテンシャル達成率",
            nbins=20,
            title="ポテンシャル達成率の分布",
            labels={"ポテンシャル達成率": "達成率（%）", "count": "店舗数"},
        )
        st.plotly_chart(fig_achieve, use_container_width=True)

elif page == "📈 感度分析":
    st.title("📈 感度分析")
    st.write("各変数を変化させたときの売上変化を確認")

    # 基本16変数のみ選択可能（Ridge/Lassoでも基本変数を入力として使用）
    基本変数 = [
        "館売上規模",
        "自店坪数",
        "間口スコア",
        "KDDI人流数値",
        "フロア評価",
        "視認性",
        "動線スコア",
        "郊外商業",
        "アウトレット",
        "駅ビル駅近",
        "量販店内",
        "路面店",
        "量販店近接",
        "商圏年収",
        "競合店数",
        "インバウンド",
    ]

    if model_type in ["ridge", "lasso"]:
        st.info(
            "📌 Ridge/Lassoモデルでは内部で特徴量エンジニアリングが自動適用されます"
        )

    変数選択 = st.selectbox("分析する変数", 基本変数)

    基準店舗 = st.selectbox("基準店舗", df["店舗名"].tolist())
    基準データ = df[df["店舗名"] == 基準店舗].iloc[0]

    変数範囲 = []
    if 変数選択 in [
        "館売上規模",
        "間口スコア",
        "視認性",
        "動線スコア",
        "商圏年収",
        "競合店数",
        "インバウンド",
    ]:
        変数範囲 = list(range(1, 6))
    elif 変数選択 == "フロア評価":
        変数範囲 = [0.5, 0.8, 1.0]
    elif 変数選択 in ["自店坪数"]:
        変数範囲 = list(range(5, 101, 5))
    elif 変数選択 == "KDDI人流数値":
        変数範囲 = list(range(0, 3000001, 300000))
    else:
        変数範囲 = [0, 1]

    結果 = []
    for val in 変数範囲:
        # 基本16変数のみでDataFrame作成
        テストデータ = 基準データ[基本変数].copy()
        テストデータ[変数選択] = val
        予測売上 = model.predict(pd.DataFrame([テストデータ]))[0]
        結果.append({"値": val, "予測売上": 予測売上})

    結果df = pd.DataFrame(結果)

    fig = px.line(
        結果df,
        x="値",
        y="予測売上",
        markers=True,
        title=f"{変数選択}の変化による売上への影響（基準: {基準店舗}）",
    )
    fig.add_hline(
        y=基準データ["平均売上"],
        line_dash="dash",
        line_color="red",
        annotation_text="実績売上",
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "🌳 決定木分析":
    st.title("🌳 決定木分析")
    st.write("if-thenルールで売上を予測")

    tree_model, tree_df, tree_eval_df = load_decision_tree_model()

    # タブ分割
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 決定木構造",
            "🔮 新規出店シミュレーター",
            "🏬 既存店ポテンシャル",
            "📈 変数重要度",
        ]
    )

    with tab1:
        st.subheader("🌳 決定木の構造")
        stats = tree_model.get_summary_stats()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("決定係数 R²", f"{stats['R二乗']:.3f}")
        col2.metric("調整済みR²", f"{stats['調整済みR二乗']:.3f}")
        col3.metric("木の深さ", f"{stats['木の深さ']}層")
        col4.metric("分岐数", f"{stats['葉ノード数']}個")

        st.info("💡 決定木は「もし〇〇なら△△」というルールで売上を予測します")

        # 決定木の可視化
        display_depth = st.slider("表示する深さ", 1, stats["木の深さ"], 3)

        with st.spinner("決定木を描画中..."):
            tree_img_base64 = tree_model.plot_tree(max_depth_display=display_depth)
            if tree_img_base64:
                st.image(
                    f"data:image/png;base64,{tree_img_base64}", use_column_width=True
                )

        st.markdown("""
        **読み方:**
        - 上から順にif-thenルールが適用されます
        - 色が濃いほど高売上を予測
        - `samples`は該当する店舗数
        - `value`は予測値（対数変換後）
        """)

    with tab2:
        st.subheader("🔮 新規出店シミュレーター（決定木版）")
        st.write("条件を入力して売上予測を確認")

        col1, col2, col3 = st.columns(3)

        with col1:
            館売上規模 = st.selectbox(
                "館売上規模", [1, 2, 3, 4, 5], help="1=小, 5=大", key="tree_館売上規模"
            )
            自店坪数 = st.number_input(
                "自店坪数", 5.0, 100.0, 25.0, key="tree_自店坪数"
            )
            間口スコア = st.selectbox(
                "間口スコア", [1, 2, 3, 4, 5], key="tree_間口スコア"
            )
            KDDI人流数値 = st.number_input(
                "KDDI人流数値", 0, 3000000, 500000, key="tree_KDDI人流数値"
            )
            フロア評価 = st.selectbox(
                "フロア評価", [0.5, 0.8, 1.0], key="tree_フロア評価"
            )

        with col2:
            視認性 = st.selectbox("視認性", [1, 2, 3, 4, 5], key="tree_視認性")
            動線スコア = st.selectbox(
                "動線スコア", [1, 2, 3, 4, 5], key="tree_動線スコア"
            )
            商圏年収 = st.selectbox("商圏年収", [1, 2, 3, 4, 5], key="tree_商圏年収")
            競合店数 = st.selectbox("競合店数", [0, 1, 2, 3, 4, 5], key="tree_競合店数")
            インバウンド = st.selectbox(
                "インバウンド", [1, 2, 3, 4, 5], key="tree_インバウンド"
            )

        with col3:
            st.write("**立地タイプ（1つ選択）**")
            立地 = st.radio(
                "",
                ["郊外商業", "アウトレット", "駅ビル駅近", "量販店内", "路面店"],
                key="tree_立地",
            )
            郊外商業 = 1 if 立地 == "郊外商業" else 0
            アウトレット = 1 if 立地 == "アウトレット" else 0
            駅ビル駅近 = 1 if 立地 == "駅ビル駅近" else 0
            量販店内 = 1 if 立地 == "量販店内" else 0
            路面店 = 1 if 立地 == "路面店" else 0

            量販店近接 = st.selectbox("量販店近接", [0, 1], key="tree_量販店近接")

        # 予測
        input_data = pd.DataFrame(
            [
                {
                    "館売上規模": 館売上規模,
                    "自店坪数": 自店坪数,
                    "間口スコア": 間口スコア,
                    "KDDI人流数値": KDDI人流数値,
                    "フロア評価": フロア評価,
                    "視認性": 視認性,
                    "動線スコア": 動線スコア,
                    "郊外商業": 郊外商業,
                    "アウトレット": アウトレット,
                    "駅ビル駅近": 駅ビル駅近,
                    "量販店内": 量販店内,
                    "路面店": 路面店,
                    "量販店近接": 量販店近接,
                    "商圏年収": 商圏年収,
                    "競合店数": 競合店数,
                    "インバウンド": インバウンド,
                }
            ]
        )

        predicted_sales = tree_model.predict(input_data)[0]

        st.success(f"### 📊 予測月間売上: ¥{predicted_sales:,.0f}")

        avg_sales = tree_df["平均売上"].mean()
        diff_pct = ((predicted_sales - avg_sales) / avg_sales) * 100
        st.metric("既存店平均との差", f"{diff_pct:+.1f}%")

        similar = tree_eval_df.iloc[
            (tree_eval_df["予測値"] - predicted_sales).abs().argsort()[:5]
        ]
        st.subheader("類似条件の既存店舗")
        st.dataframe(similar[["店舗名", "実測値", "予測値"]])

    with tab3:
        st.subheader("🏬 既存店舗ポテンシャル評価（決定木版）")

        # ポテンシャル売上列を追加
        tree_eval_enhanced = tree_eval_df.copy()
        tree_eval_enhanced["ポテンシャル売上"] = tree_eval_enhanced["予測値"]
        tree_eval_enhanced["現在売上"] = tree_eval_enhanced["実測値"]
        tree_eval_enhanced["未達成額"] = (
            tree_eval_enhanced["ポテンシャル売上"] - tree_eval_enhanced["現在売上"]
        )
        tree_eval_enhanced["ポテンシャル達成率"] = (
            tree_eval_enhanced["現在売上"]
            / tree_eval_enhanced["ポテンシャル売上"]
            * 100
        )
        tree_eval_enhanced["改善余地"] = tree_eval_enhanced.apply(
            lambda x: "大"
            if x["未達成額"] > 5000000
            else ("中" if x["未達成額"] > 2000000 else "小"),
            axis=1,
        )

        # サマリー指標
        col1, col2, col3, col4 = st.columns(4)
        総ポテンシャル = tree_eval_enhanced["ポテンシャル売上"].sum()
        総実績 = tree_eval_enhanced["現在売上"].sum()
        総未達成額 = 総ポテンシャル - 総実績
        全体達成率 = 総実績 / 総ポテンシャル * 100

        col1.metric("全店舗ポテンシャル売上", f"¥{総ポテンシャル / 100000000:.1f}億円")
        col2.metric("全店舗実績売上", f"¥{総実績 / 100000000:.1f}億円")
        col3.metric(
            "改善余地（未達成額）",
            f"¥{総未達成額 / 100000000:.1f}億円",
            delta=f"{総未達成額 / 10000:.0f}万円/月",
        )
        col4.metric("ポテンシャル達成率", f"{全体達成率:.1f}%")

        # 散布図
        fig = px.scatter(
            tree_eval_enhanced,
            x="ポテンシャル売上",
            y="現在売上",
            color="改善余地",
            size=abs(tree_eval_enhanced["未達成額"]),
            hover_data=["店舗名", "ポテンシャル達成率", "未達成額"],
            color_discrete_map={"大": "red", "中": "orange", "小": "green"},
            title="ポテンシャル売上 vs 現在売上（決定木）",
        )
        fig.add_shape(
            type="line",
            x0=tree_eval_enhanced["ポテンシャル売上"].min(),
            y0=tree_eval_enhanced["ポテンシャル売上"].min(),
            x1=tree_eval_enhanced["ポテンシャル売上"].max(),
            y1=tree_eval_enhanced["ポテンシャル売上"].max(),
            line=dict(color="blue", dash="dash"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # 改善インパクトランキング
        st.subheader("💰 改善インパクトランキング（未達成額上位）")
        改善余地大 = (
            tree_eval_enhanced[tree_eval_enhanced["未達成額"] > 0]
            .sort_values("未達成額", ascending=False)
            .head(10)
        )
        改善余地大_表示 = 改善余地大[
            [
                "店舗名",
                "現在売上",
                "ポテンシャル売上",
                "未達成額",
                "ポテンシャル達成率",
                "改善余地",
            ]
        ].copy()
        改善余地大_表示["現在売上"] = 改善余地大_表示["現在売上"].apply(
            lambda x: f"¥{x:,.0f}"
        )
        改善余地大_表示["ポテンシャル売上"] = 改善余地大_表示["ポテンシャル売上"].apply(
            lambda x: f"¥{x:,.0f}"
        )
        改善余地大_表示["未達成額"] = 改善余地大_表示["未達成額"].apply(
            lambda x: f"¥{x:,.0f}"
        )
        改善余地大_表示["ポテンシャル達成率"] = 改善余地大_表示[
            "ポテンシャル達成率"
        ].apply(lambda x: f"{x:.1f}%")
        st.dataframe(改善余地大_表示, use_container_width=True)

        # 詳細タブ
        tab_sub1, tab_sub2, tab_sub3 = st.tabs(
            ["⚠️ 要改善店舗", "🔥 好調店舗", "📋 全店舗詳細"]
        )

        with tab_sub1:
            st.write("**ポテンシャル未達成店舗（改善余地あり）**")
            要改善 = tree_eval_enhanced[tree_eval_enhanced["未達成額"] > 0].sort_values(
                "未達成額", ascending=False
            )
            要改善_表示 = 要改善[
                [
                    "店舗名",
                    "現在売上",
                    "ポテンシャル売上",
                    "未達成額",
                    "ポテンシャル達成率",
                    "改善余地",
                ]
            ].copy()
            要改善_表示["現在売上"] = 要改善_表示["現在売上"].apply(
                lambda x: f"¥{x:,.0f}"
            )
            要改善_表示["ポテンシャル売上"] = 要改善_表示["ポテンシャル売上"].apply(
                lambda x: f"¥{x:,.0f}"
            )
            要改善_表示["未達成額"] = 要改善_表示["未達成額"].apply(
                lambda x: f"¥{x:,.0f}"
            )
            要改善_表示["ポテンシャル達成率"] = 要改善_表示["ポテンシャル達成率"].apply(
                lambda x: f"{x:.1f}%"
            )
            st.dataframe(要改善_表示, use_container_width=True)

        with tab_sub2:
            st.write("**ポテンシャル超過達成店舗（好調店舗）**")
            好調 = tree_eval_enhanced[tree_eval_enhanced["未達成額"] < 0].sort_values(
                "未達成額"
            )
            好調_表示 = 好調[
                [
                    "店舗名",
                    "現在売上",
                    "ポテンシャル売上",
                    "未達成額",
                    "ポテンシャル達成率",
                ]
            ].copy()
            好調_表示["現在売上"] = 好調_表示["現在売上"].apply(lambda x: f"¥{x:,.0f}")
            好調_表示["ポテンシャル売上"] = 好調_表示["ポテンシャル売上"].apply(
                lambda x: f"¥{x:,.0f}"
            )
            好調_表示["未達成額"] = 好調_表示["未達成額"].apply(lambda x: f"¥{x:,.0f}")
            好調_表示["ポテンシャル達成率"] = 好調_表示["ポテンシャル達成率"].apply(
                lambda x: f"{x:.1f}%"
            )
            st.dataframe(好調_表示, use_container_width=True)

        with tab_sub3:
            全店舗_表示 = tree_eval_enhanced[
                [
                    "店舗名",
                    "現在売上",
                    "ポテンシャル売上",
                    "未達成額",
                    "ポテンシャル達成率",
                    "改善余地",
                    "評価",
                ]
            ].copy()
            全店舗_表示["現在売上"] = 全店舗_表示["現在売上"].apply(
                lambda x: f"¥{x:,.0f}"
            )
            全店舗_表示["ポテンシャル売上"] = 全店舗_表示["ポテンシャル売上"].apply(
                lambda x: f"¥{x:,.0f}"
            )
            全店舗_表示["未達成額"] = 全店舗_表示["未達成額"].apply(
                lambda x: f"¥{x:,.0f}"
            )
            全店舗_表示["ポテンシャル達成率"] = 全店舗_表示["ポテンシャル達成率"].apply(
                lambda x: f"{x:.1f}%"
            )
            st.dataframe(全店舗_表示, use_container_width=True)

    with tab4:
        st.subheader("📈 変数重要度ランキング")
        st.write("決定木での分岐に最も使われている変数")

        importance_df = tree_model.get_feature_importance()

        fig = px.bar(
            importance_df,
            x="重要度",
            y="変数",
            orientation="h",
            color="重要度",
            color_continuous_scale="Viridis",
            title="各変数の重要度（決定木）",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(importance_df, use_container_width=True)

        st.info("""
        **変数重要度とは:**
        - 決定木の分岐でどの変数がよく使われているか
        - 重要度が高いほど売上予測への影響が大きい
        - 合計が1.0になるように正規化されています
        """)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **使い方**\n- 📊 分析結果: 重回帰分析の結果\n- 🔮 シミュレーター: 新規出店の売上予測\n- 🏬 ポテンシャル: 既存店評価\n- 📈 感度分析: 変数変化の影響\n- 🌳 決定木分析: if-thenルールで予測"
)
