"""Streamlit店舗分析ダッシュボード"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from regression_model import (
    StoreRegressionModel,
    DecisionTreeModel,
    LocationSpecificModel,
    LOCATION_TYPES,
)
import base64
from auth import check_password

st.set_page_config(page_title="Anker Store 出店分析", layout="wide", page_icon="🏪")

# 認証チェック
if not check_password():
    st.stop()


@st.cache_resource
def load_location_specific_model(exclude_shiodome=False):
    loc_model = LocationSpecificModel(exclude_shiodome=exclude_shiodome)
    df = loc_model.load_data()
    loc_model.fit_all(df)
    return loc_model, df


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

# 汐留除外オプション
st.sidebar.markdown("### ⚙️ オプション")
exclude_shiodome = st.sidebar.checkbox(
    "汐留を学習データから除外（路面店R²改善）",
    value=False,
    help="路面店の決定係数が80-90%圏内に収まります"
)

loc_model, loc_df = load_location_specific_model(exclude_shiodome=exclude_shiodome)

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "メニュー",
    [
        "📍 立地別モデル",
        "🏬 既存店ポテンシャル",
        "🌳 決定木分析",
    ],
)


# ========== 既存店ポテンシャル（立地別統合モデルベース） ==========
if page == "🏬 既存店ポテンシャル":
    st.title("🏬 既存店舗ポテンシャル評価")
    st.markdown("立地別統合Ridgeモデル（α=0.28）による全店舗のポテンシャル診断")
    st.caption(f"特徴量数={len(loc_model.feature_names)} / 汐留除外={'オン' if exclude_shiodome else 'オフ'}")

    # 全店舗を立地別モデルで1店舗ずつ予測
    predictions = []
    for i, row in loc_df.iterrows():
        # 立地タイプを判定
        loc_type = None
        for loc in LOCATION_TYPES:
            if row[loc] == 1:
                loc_type = loc
                break
        if loc_type is None:
            continue
        input_data = pd.DataFrame([row[['館売上規模','自店坪数','間口スコア','KDDI人流数値','フロア評価','視認性','動線スコア','郊外商業','アウトレット','駅ビル駅近','量販店内','路面店','量販店近接','商圏年収','競合店数','インバウンド']]])
        for c in ['リピーター率','Cafe','間口mm','KDDI人流数値_リピート非計上']:
            if c in row.index:
                input_data[c] = row[c]
        pred = loc_model.predict(loc_type, input_data)
        predictions.append({
            '店舗名': row['店舗名'],
            '立地': loc_type,
            'ポテンシャル売上': pred or 0,
            '現在売上': row['平均売上'],
        })

    pot_df = pd.DataFrame(predictions)
    pot_df['未達成額'] = pot_df['ポテンシャル売上'] - pot_df['現在売上']
    pot_df['ポテンシャル達成率'] = pot_df['現在売上'] / pot_df['ポテンシャル売上'] * 100
    pot_df['改善余地'] = pot_df['未達成額'].apply(
        lambda x: '大' if x > 5000000 else ('中' if x > 2000000 else ('小' if x > 0 else '超過達成'))
    )
    pot_df['評価'] = pot_df['改善余地'].apply(
        lambda x: '🔥好調' if x == '超過達成' else ('⚠️要改善' if x in ['大','中'] else '✅適正')
    )

    # サマリー
    col1, col2, col3, col4 = st.columns(4)
    total_pot = pot_df['ポテンシャル売上'].sum()
    total_act = pot_df['現在売上'].sum()
    total_gap = total_pot - total_act
    col1.metric("全店舗ポテンシャル売上", f"¥{total_pot/1e8:.1f}億円")
    col2.metric("全店舗実績売上", f"¥{total_act/1e8:.1f}億円")
    col3.metric("改善余地（未達成額）", f"¥{total_gap/1e8:.1f}億円", delta=f"{total_gap/1e4:.0f}万円/月")
    col4.metric("ポテンシャル達成率", f"{total_act/total_pot*100:.1f}%")

    # 散布図
    fig = px.scatter(
        pot_df, x='ポテンシャル売上', y='現在売上', color='立地',
        size=abs(pot_df['未達成額']), hover_data=['店舗名','ポテンシャル達成率','未達成額'],
        title='ポテンシャル売上 vs 現在売上（立地別モデル）',
    )
    fig.add_shape(type='line', x0=pot_df['ポテンシャル売上'].min(), y0=pot_df['ポテンシャル売上'].min(),
                  x1=pot_df['ポテンシャル売上'].max(), y1=pot_df['ポテンシャル売上'].max(),
                  line=dict(color='blue', dash='dash'))
    st.plotly_chart(fig, use_container_width=True)

    # 改善ランキング
    st.subheader("💰 改善インパクトランキング（未達成額上位）")
    rank_df = pot_df[pot_df['未達成額'] > 0].sort_values('未達成額', ascending=False).head(10)
    disp = rank_df[['店舗名','立地','現在売上','ポテンシャル売上','未達成額','ポテンシャル達成率']].copy()
    for c in ['現在売上','ポテンシャル売上','未達成額']:
        disp[c] = disp[c].apply(lambda x: f"¥{x:,.0f}")
    disp['ポテンシャル達成率'] = disp['ポテンシャル達成率'].apply(lambda x: f"{x:.1f}%")
    st.dataframe(disp, use_container_width=True)

    # タブ
    tab1, tab2 = st.tabs(["⚠️ 要改善店舗", "📋 全店舗詳細"])
    with tab1:
        bad = pot_df[pot_df['未達成額'] > 0].sort_values('未達成額', ascending=False)
        bad_disp = bad[['店舗名','立地','現在売上','ポテンシャル売上','未達成額','ポテンシャル達成率']].copy()
        for c in ['現在売上','ポテンシャル売上','未達成額']:
            bad_disp[c] = bad_disp[c].apply(lambda x: f"¥{x:,.0f}")
        bad_disp['ポテンシャル達成率'] = bad_disp['ポテンシャル達成率'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(bad_disp, use_container_width=True)
    with tab2:
        all_disp = pot_df[['店舗名','立地','現在売上','ポテンシャル売上','未達成額','ポテンシャル達成率','評価']].copy()
        for c in ['現在売上','ポテンシャル売上','未達成額']:
            all_disp[c] = all_disp[c].apply(lambda x: f"¥{x:,.0f}")
        all_disp['ポテンシャル達成率'] = all_disp['ポテンシャル達成率'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(all_disp, use_container_width=True)

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

elif page == "📍 立地別モデル":
    st.title("📍 立地タイプ別専用モデル")
    st.markdown("全49店舗を使った統合Ridge回帰＋立地別交互作用項で予測します。")
    st.caption(f"α={loc_model.alpha}, 特徴量数={len(loc_model.feature_names)}")

    selected_loc = st.selectbox("立地タイプを選択", LOCATION_TYPES)

    if loc_model.model is not None:
        count = loc_model.data_counts.get(selected_loc, 0)
        r2_fit = loc_model.get_r2(selected_loc)

        col1, col2 = st.columns(2)
        col1.metric("学習店舗数", f"{count}店舗")
        col2.metric("決定係数 R²", f"{r2_fit:.3f}" if r2_fit else "N/A")
        if r2_fit and r2_fit >= 0.8:
            st.success(f"✅ {selected_loc} のR²={r2_fit:.1%}（目標80%達成）")
        elif r2_fit:
            st.warning(f"⚠️ {selected_loc} のR²={r2_fit:.1%}")

        st.subheader(" 変数の影響度（標準化係数）")
        importance_df = loc_model.get_feature_importance(selected_loc)
        if importance_df is not None:
            fig = px.bar(
                importance_df.head(10),
                x="標準化係数",
                y="変数",
                orientation="h",
                color="標準化係数",
                color_continuous_scale="RdBu_r",
                title=f"{selected_loc} の重要変数 Top10",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(importance_df, use_container_width=True)

        st.subheader(" シミュレーター")
        col1, col2, col3 = st.columns(3)
        with col1:
            館売上規模 = st.selectbox(
                "館売上規模", [1, 2, 3, 4, 5], key=f"loc_{selected_loc}_館"
            )
            自店坪数 = st.number_input(
                "自店坪数", 5.0, 100.0, 25.0, key=f"loc_{selected_loc}_坪"
            )
            間口スコア = st.selectbox(
                "間口スコア", [1, 2, 3, 4, 5], key=f"loc_{selected_loc}_間"
            )
            KDDI人流数値 = st.number_input(
                "KDDI人流数値", 0, 3000000, 500000, key=f"loc_{selected_loc}_人"
            )
            フロア評価 = st.selectbox(
                "フロア評価", [0.5, 0.8, 1.0], key=f"loc_{selected_loc}_フ"
            )
        with col2:
            視認性 = st.selectbox(
                "視認性", [1, 2, 3, 4, 5], key=f"loc_{selected_loc}_視"
            )
            動線スコア = st.selectbox(
                "動線スコア", [1, 2, 3, 4, 5], key=f"loc_{selected_loc}_動"
            )
            商圏年収 = st.selectbox(
                "商圏年収", [1, 2, 3, 4, 5], key=f"loc_{selected_loc}_商"
            )
            競合店数 = st.selectbox(
                "競合店数", [0, 1, 2, 3, 4, 5], key=f"loc_{selected_loc}_競"
            )
            インバウンド = st.selectbox(
                "インバウンド", [1, 2, 3, 4, 5], key=f"loc_{selected_loc}_イ"
            )
        with col3:
            量販店近接 = st.selectbox(
                "量販店近接", [0, 1], key=f"loc_{selected_loc}_量"
            )

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
                    "郊外商業": 0,
                    "アウトレット": 0,
                    "駅ビル駅近": 0,
                    "量販店内": 0,
                    "路面店": 0,
                    "量販店近接": 量販店近接,
                    "商圏年収": 商圏年収,
                    "競合店数": 競合店数,
                    "インバウンド": インバウンド,
                }
            ]
        )

        predicted = loc_model.predict(selected_loc, input_data)
        if predicted is not None:
            st.success(f"### 📊 予測月間売上: ¥{predicted:,.0f}")
            avg_sales = loc_df["平均売上"].mean()
            diff_pct = ((predicted - avg_sales) / avg_sales) * 100
            st.metric("既存店平均との差", f"{diff_pct:+.1f}%")
        else:
            st.warning("⚠️ モデルが未学習です。管理者にご連絡ください。")
    else:
        st.warning("⚠️ モデルが未学習です。管理者にご連絡ください。")

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **新メニュー**\n"
    "- 📍 立地別モデル: 係数表＋シミュレーター\n"
    "- 🏬 既存店ポテンシャル: 立地別モデルで全店舗評価\n"
    "- 🌳 決定木分析: if-thenルールで予測"
)
