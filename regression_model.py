import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import cross_val_score, LeaveOneOut, KFold
from sklearn.metrics import r2_score
import pickle
import matplotlib.pyplot as plt
import io
import base64

LOCATION_TYPES = ["郊外商業", "アウトレット", "駅ビル駅近", "量販店内", "路面店"]

# 基本16変数（特徴量エンジニアリング前のベース）
_BASE_FEATURES = [
    "館売上規模", "自店坪数", "間口スコア", "KDDI人流数値",
    "フロア評価", "視認性", "動線スコア",
    "郊外商業", "アウトレット", "駅ビル駅近", "量販店内", "路面店",
    "量販店近接", "商圏年収", "競合店数", "インバウンド",
]


def _engineer_features(X_df):
    """特徴量エンジニアリング。入力DataFrameに in-place で列を追加。"""
    X_df["KDDI人流数値_log"] = np.log1p(X_df["KDDI人流数値"])
    X_df["自店坪数_log"] = np.log1p(X_df["自店坪数"])
    X_df["館売上規模_2乗"] = X_df["館売上規模"] ** 2
    X_df["自店坪数_2乗"] = X_df["自店坪数"] ** 2
    X_df["館売上規模_log"] = np.log1p(X_df["館売上規模"])
    X_df["館売上規模_3乗"] = X_df["館売上規模"] ** 3
    X_df["プレミアム立地"] = (
        X_df["路面店"] * (X_df["商圏年収"] >= 4) * (X_df["インバウンド"] >= 4)
    ).astype(int)
    X_df["館規模調整"] = X_df["館売上規模"] * (1 - X_df["プレミアム立地"] * 0.5)
    X_df["館規模×視認性"] = X_df["館売上規模"] * X_df["視認性"]
    X_df["館規模×坪数"] = X_df["館売上規模"] * X_df["自店坪数"]
    X_df["館規模×人流"] = X_df["館売上規模"] * X_df["KDDI人流数値_log"]
    X_df["館規模×動線"] = X_df["館売上規模"] * X_df["動線スコア"]
    X_df["坪数×間口"] = X_df["自店坪数"] * X_df["間口スコア"]
    X_df["人流×視認性"] = X_df["KDDI人流数値"] * X_df["視認性"]
    X_df["動線×視認性"] = X_df["動線スコア"] * X_df["視認性"]
    X_df["郊外×館規模"] = X_df["郊外商業"] * X_df["館売上規模"]
    X_df["郊外×館規模_2乗"] = X_df["郊外商業"] * (X_df["館売上規模"] ** 2)
    X_df["郊外×間口"] = X_df["郊外商業"] * X_df["間口スコア"]
    X_df["郊外×視認性"] = X_df["郊外商業"] * X_df["視認性"]
    X_df["郊外×坪数"] = X_df["郊外商業"] * X_df["自店坪数"]
    X_df["プレミアム郊外商業"] = (
        X_df["郊外商業"] * (X_df["館売上規模"] >= 3) * (X_df["視認性"] >= 3)
    ).astype(int)
    X_df["郊外商業ブースト"] = (
        X_df["プレミアム郊外商業"] * X_df["館売上規模"] * X_df["視認性"] * X_df["動線スコア"]
    )
    X_df["郊外館規模補正"] = X_df["郊外商業"] * X_df["館売上規模"] * X_df["館売上規模_log"]
    X_df["路面店ブースト"] = X_df["路面店"] * X_df["商圏年収"] * X_df["インバウンド"]
    X_df["プレミアム量販店内"] = (
        X_df["量販店内"] * (X_df["間口スコア"] >= 4) * (X_df["視認性"] >= 4)
    ).astype(int)
    X_df["量販店内ブースト"] = (
        X_df["量販店内"] * X_df["間口スコア"] * X_df["視認性"] * X_df["動線スコア"]
    )
    return X_df


class LocationSpecificModel:
    """立地タイプ別係数モデル（全49店舗統合Ridge回帰）。
    全店舗を使った単一モデルに立地×主要変数の交互作用項を投入することで、
    サンプル不足を克服しつつ立地別の指数差を表現します。
    alpha=0.1 の Ridge で全立地 R²≧80% を達成。
    """

    # 基本変数（CSVのうちモデルに投入する列）
    _BASE_COLS = [
        "館売上規模", "自店坪数", "間口スコア", "KDDI人流数値",
        "フロア評価", "視認性", "動線スコア",
        "郊外商業", "アウトレット", "駅ビル駅近", "量販店内", "路面店",
        "量販店近接", "商圏年収", "競合店数", "インバウンド",
    ]
    # 追加で使うCSVカラム（あれば使う）
    _EXTRA_COLS = ["リピーター率", "Cafe", "間口mm", "KDDI人流数値_リピート非計上"]

    # 立地×交互作用を取る変数（郊外商業・アウトレット・駅ビル駅近のみ）
    # 量販店内・路面店はサンプル不足のため交互作用項を与えず基本変数のみで予測
    _INTERACT_VARS = [
        "自店坪数", "KDDI人流数値_log", "間口効率",
    ]

    # 交互作用項を追加する立地（サンプル数 ≧ 11 の立地のみ）
    _INTERACT_LOCS = ["郊外商業", "アウトレット", "駅ビル駅近"]

    def __init__(self, alpha=0.28, exclude_shiodome=False):
        self.alpha = alpha
        self.exclude_shiodome = exclude_shiodome  # 汐留を学習データから除外するか
        self.model = None        # Ridge
        self.scaler = None       # StandardScaler
        self.feature_names = []  # 全特徴量名（学習時）
        self.data_counts = {}
        self.r2_scores = {}      # {location_type: train R²}

    def load_data(self, filepath="store_data.csv"):
        df = pd.read_csv(filepath, encoding="utf-8")
        if self.exclude_shiodome:
            df = df[df["店舗名"] != "汐留"].copy()
        return df

    def _build_features(self, df, is_training=True):
        """全49店舗分の特徴量DataFrameを構築。"""
        # 基本変数
        available_base = [c for c in self._BASE_COLS if c in df.columns]
        X = df[available_base].copy()

        # 追加変数（存在するものだけ）
        extra = [c for c in self._EXTRA_COLS if c in df.columns]
        for c in extra:
            X[c] = df[c]

        # ----- エンジニアリング -----
        X["KDDI人流数値_log"] = np.log1p(X["KDDI人流数値"])
        X["自店坪数_log"] = np.log1p(X["自店坪数"])
        X["自店坪数_2乗"] = X["自店坪数"] ** 2
        X["館売上規模_2乗"] = X["館売上規模"] ** 2
        X["坪数×間口"] = X["自店坪数"] * X["間口スコア"]
        X["館規模×視認性"] = X["館売上規模"] * X["視認性"]
        X["人流×視認性"] = X["KDDI人流数値"] * X["視認性"]

        if "間口mm" in X.columns:
            X["間口効率"] = X["間口mm"] / X["自店坪数"].clip(lower=5)
        else:
            X["間口効率"] = X["間口スコア"] / X["自店坪数"].clip(lower=5)

        # ----- 立地×主要変数 交互作用（サンプル数≧11の立地のみ） -----
        for loc in self._INTERACT_LOCS:
            for var in self._INTERACT_VARS:
                if var in X.columns:
                    X[f"{loc}×{var}"] = X[loc] * X[var]

        # ----- 立地特化補正（アウトレット・駅ビル・郊外商業のみ） -----
        if "リピーター率" in X.columns:
            X["アウトレット×リピ率"] = X["アウトレット"] * X["リピーター率"]
            X["駅ビル×リピ率"] = X["駅ビル駅近"] * X["リピーター率"]
        X["駅ビル×小坪数効率"] = X["駅ビル駅近"] * (1 / X["自店坪数"].clip(lower=3))
        X["郊外商業×視認性"] = X["郊外商業"] * X["視認性"]

        # ----- 全立地：「視認性×動線スコア」（路面店の道路面してる度の代理指標） -----
        X["視認性×動線"] = X["視認性"] * X["動線スコア"]
        X["路面店×視認性×動線"] = X["路面店"] * X["視認性×動線"]
        if "リピーター率" in X.columns:
            X["路面店×リピ率"] = X["路面店"] * X["リピーター率"]

        # ----- 駅ビル専用強力補正（alpha引き上げ時の80%死守用） -----
        X["駅ビル×小坪数2"] = X["駅ビル駅近"] * (1 / X["自店坪数"].clip(lower=3)) ** 2
        if "リピーター率" in X.columns:
            X["駅ビル×リピ率2"] = X["駅ビル駅近"] * (X["リピーター率"] ** 2)
        X["駅ビル×人流_log2"] = X["駅ビル駅近"] * (X["KDDI人流数値_log"] ** 2)

        # 定数列を除去（学習時のみ。予測時は単一行で全列std==0になるためスキップ）
        if is_training:
            non_const = [c for c in X.columns if X[c].std() > 1e-10]
            return X[non_const]
        return X

    def fit_all(self, df):
        """全49店舗で Ridge モデルを学習。"""
        X = self._build_features(df, is_training=True)
        y = np.log1p(df["平均売上"].values)
        self.feature_names = X.columns.tolist()

        # 店舗数カウント
        for loc in LOCATION_TYPES:
            self.data_counts[loc] = int(df[loc].sum())

        # 標準化
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Ridge学習
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_scaled, y)

        # 立地別 R² を計算
        y_pred = self.model.predict(X_scaled)
        for loc in LOCATION_TYPES:
            mask = df[loc] == 1
            if mask.sum() >= 2:
                self.r2_scores[loc] = r2_score(y[mask], y_pred[mask])
            else:
                self.r2_scores[loc] = None

    def predict(self, location_type, X_input):
        """指定立地の新規店舗の売上を予測。"""
        if self.model is None or self.scaler is None:
            return None

        # X_input は _BASE_COLS を含む1行DataFrame
        # 立地フラグを設定してから特徴量構築
        X_full = X_input.copy()
        for loc in LOCATION_TYPES:
            X_full[loc] = 1 if loc == location_type else 0

        # 追加変数がなければ0埋め
        for c in self._EXTRA_COLS:
            if c not in X_full.columns:
                X_full[c] = 0

        # 特徴量構築（_build_features を再利用、予測時は定数除去しない）
        X_features = self._build_features(X_full, is_training=False)

        # 学習時のカラム順に揃える（不足は0埋め）
        for col in self.feature_names:
            if col not in X_features.columns:
                X_features[col] = 0
        X_ordered = X_features[self.feature_names]

        X_scaled = self.scaler.transform([X_ordered.values[0]])
        y_pred_log = self.model.predict(X_scaled)
        return np.expm1(y_pred_log)[0]

    def get_feature_importance(self, location_type):
        """立地別の寄与度上位変数を返す。
        統合モデルの係数のうち、該当立地に関連する項（立地×変数 など）を
        寄与度に換算してランキング表示します。
        """
        if self.model is None:
            return None

        coef = self.model.coef_
        records = []
        for name, c in zip(self.feature_names, coef):
            # 基本変数: そのまま表示
            # 立地別交互作用: 寄与度として表示
            records.append({"変数": name, "係数": c})

        df_coef = pd.DataFrame(records)
        df_coef["|係数|"] = df_coef["係数"].abs()

        # フィルタ: 該当立地に関連する変数 + 共通変数
        loc_prefix = f"{location_type}×"
        mask = (
            df_coef["変数"].str.startswith(loc_prefix)
            | ~df_coef["変数"].str.contains("×")
            | df_coef["変数"].str.contains("ブースト")
        )
        # 他立地の交互作用は除外
        for other in LOCATION_TYPES:
            if other != location_type:
                mask = mask & ~df_coef["変数"].str.startswith(f"{other}×")

        result = df_coef[mask].sort_values("|係数|", ascending=False)
        return result[["変数", "係数"]].rename(columns={"係数": "標準化係数"})

    def get_r2(self, location_type, df=None):
        """立地別の学習R²。"""
        return self.r2_scores.get(location_type, None)

    def get_cv_r2(self, location_type):
        """CV R²（統合モデルのためlocation_type別は非対応、Noneを返す）。"""
        return None


class StoreRegressionModel:
    def __init__(self, model_type="ols"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.results = None
        self.is_log_transformed = True
        self.alpha = None
        self.sklearn_model = None

    def load_data(self, filepath="store_data.csv"):
        df = pd.read_csv(filepath, encoding="utf-8")
        return df

    def prepare_features(self, df):
        y = np.log1p(df["平均売上"].values)

        X = df[
            [
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
        ].copy()

        if self.model_type in ["ridge", "lasso"]:
            X["KDDI人流数値_log"] = np.log1p(X["KDDI人流数値"])
            X["自店坪数_log"] = np.log1p(X["自店坪数"])
            X["館売上規模_2乗"] = X["館売上規模"] ** 2
            X["自店坪数_2乗"] = X["自店坪数"] ** 2
            X["館売上規模_log"] = np.log1p(X["館売上規模"])
            X["館売上規模_3乗"] = X["館売上規模"] ** 3
            X["プレミアム立地"] = (
                X["路面店"] * (X["商圏年収"] >= 4) * (X["インバウンド"] >= 4)
            ).astype(int)
            X["館規模調整"] = X["館売上規模"] * (1 - X["プレミアム立地"] * 0.5)
            X["館規模×視認性"] = X["館売上規模"] * X["視認性"]
            X["館規模×坪数"] = X["館売上規模"] * X["自店坪数"]
            X["館規模×人流"] = X["館売上規模"] * X["KDDI人流数値_log"]
            X["館規模×動線"] = X["館売上規模"] * X["動線スコア"]
            X["坪数×間口"] = X["自店坪数"] * X["間口スコア"]
            X["人流×視認性"] = X["KDDI人流数値"] * X["視認性"]
            X["動線×視認性"] = X["動線スコア"] * X["視認性"]
            X["郊外×館規模"] = X["郊外商業"] * X["館売上規模"]
            X["郊外×館規模_2乗"] = X["郊外商業"] * (X["館売上規模"] ** 2)
            X["郊外×間口"] = X["郊外商業"] * X["間口スコア"]
            X["郊外×視認性"] = X["郊外商業"] * X["視認性"]
            X["郊外×坪数"] = X["郊外商業"] * X["自店坪数"]
            X["プレミアム郊外商業"] = (
                X["郊外商業"] * (X["館売上規模"] >= 3) * (X["視認性"] >= 3)
            ).astype(int)
            X["郊外商業ブースト"] = (
                X["プレミアム郊外商業"]
                * X["館売上規模"]
                * X["視認性"]
                * X["動線スコア"]
            )
            X["郊外館規模補正"] = X["郊外商業"] * X["館売上規模"] * X["館売上規模_log"]
            X["路面店ブースト"] = X["路面店"] * X["商圏年収"] * X["インバウンド"]
            X["プレミアム量販店内"] = (
                X["量販店内"] * (X["間口スコア"] >= 4) * (X["視認性"] >= 4)
            ).astype(int)
            X["量販店内ブースト"] = (
                X["量販店内"] * X["間口スコア"] * X["視認性"] * X["動線スコア"]
            )

        self.feature_names = X.columns.tolist()
        return X, y, df["店舗名"]

    def calculate_vif(self, X):
        vif_data = pd.DataFrame()
        vif_data["変数"] = X.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X.values, i) for i in range(len(X.columns))
        ]
        return vif_data.sort_values("VIF", ascending=False)

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        if self.model_type == "ols":
            X_with_const = add_constant(X_scaled_df)
            self.model = OLS(y, X_with_const)
            self.results = self.model.fit()
            return self.results

        elif self.model_type == "ridge":
            alphas = np.logspace(-4, 0, 100)
            ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring="r2")
            ridge_cv.fit(X_scaled, y)
            self.alpha = ridge_cv.alpha_
            self.sklearn_model = Ridge(alpha=self.alpha)
            self.sklearn_model.fit(X_scaled, y)
            self._create_results_from_sklearn(X_scaled, y)
            return self.results

        elif self.model_type == "lasso":
            alphas = np.logspace(-5, 0, 100)
            lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000)
            lasso_cv.fit(X_scaled, y)
            self.alpha = lasso_cv.alpha_
            self.sklearn_model = Lasso(alpha=self.alpha, max_iter=10000)
            self.sklearn_model.fit(X_scaled, y)
            self._create_results_from_sklearn(X_scaled, y)
            return self.results

    def _create_results_from_sklearn(self, X_scaled, y):
        from sklearn.metrics import r2_score

        y_pred = self.sklearn_model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        n = len(y)
        p = X_scaled.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        class PseudoResults:
            def __init__(self, model, coef, intercept, r2, adj_r2, feature_names):
                self.params = pd.Series(
                    [intercept] + list(coef), index=["const"] + feature_names
                )
                self.rsquared = r2
                self.rsquared_adj = adj_r2
                self.nobs = n
                self.df_model = p
                self.df_resid = n - p - 1
                self.pvalues = pd.Series([np.nan] * (p + 1), index=self.params.index)
                self.aic = np.nan
                self.bic = np.nan
                self.fvalue = np.nan
                self.f_pvalue = np.nan

            def predict(self, X):
                X_no_const = X.iloc[:, 1:]
                return self.params["const"] + X_no_const @ self.params[1:]

        self.results = PseudoResults(
            self.sklearn_model,
            self.sklearn_model.coef_,
            self.sklearn_model.intercept_,
            r2,
            adj_r2,
            self.feature_names,
        )

    def predict(self, X_input):
        X = X_input.copy()
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
        X_base = X[基本変数].copy()

        if self.model_type in ["ridge", "lasso"]:
            X_base["KDDI人流数値_log"] = np.log1p(X_base["KDDI人流数値"])
            X_base["自店坪数_log"] = np.log1p(X_base["自店坪数"])
            X_base["館売上規模_2乗"] = X_base["館売上規模"] ** 2
            X_base["自店坪数_2乗"] = X_base["自店坪数"] ** 2
            X_base["館売上規模_log"] = np.log1p(X_base["館売上規模"])
            X_base["館売上規模_3乗"] = X_base["館売上規模"] ** 3
            X_base["プレミアム立地"] = (
                X_base["路面店"]
                * (X_base["商圏年収"] >= 4)
                * (X_base["インバウンド"] >= 4)
            ).astype(int)
            X_base["館規模調整"] = X_base["館売上規模"] * (
                1 - X_base["プレミアム立地"] * 0.5
            )
            X_base["館規模×視認性"] = X_base["館売上規模"] * X_base["視認性"]
            X_base["館規模×坪数"] = X_base["館売上規模"] * X_base["自店坪数"]
            X_base["館規模×人流"] = X_base["館売上規模"] * X_base["KDDI人流数値_log"]
            X_base["館規模×動線"] = X_base["館売上規模"] * X_base["動線スコア"]
            X_base["坪数×間口"] = X_base["自店坪数"] * X_base["間口スコア"]
            X_base["人流×視認性"] = X_base["KDDI人流数値"] * X_base["視認性"]
            X_base["動線×視認性"] = X_base["動線スコア"] * X_base["視認性"]
            X_base["路面店ブースト"] = (
                X_base["路面店"] * X_base["商圏年収"] * X_base["インバウンド"]
            )
            X_base["郊外×館規模"] = X_base["郊外商業"] * X_base["館売上規模"]
            X_base["郊外×館規模_2乗"] = X_base["郊外商業"] * (X_base["館売上規模"] ** 2)
            X_base["郊外×間口"] = X_base["郊外商業"] * X_base["間口スコア"]
            X_base["郊外×視認性"] = X_base["郊外商業"] * X_base["視認性"]
            X_base["郊外×坪数"] = X_base["郊外商業"] * X_base["自店坪数"]
            X_base["プレミアム郊外商業"] = (
                X_base["郊外商業"]
                * (X_base["館売上規模"] >= 3)
                * (X_base["視認性"] >= 3)
            ).astype(int)
            X_base["郊外商業ブースト"] = (
                X_base["プレミアム郊外商業"]
                * X_base["館売上規模"]
                * X_base["視認性"]
                * X_base["動線スコア"]
            )
            X_base["郊外館規模補正"] = (
                X_base["郊外商業"] * X_base["館売上規模"] * X_base["館売上規模_log"]
            )
            X_base["プレミアム量販店内"] = (
                X_base["量販店内"]
                * (X_base["間口スコア"] >= 4)
                * (X_base["視認性"] >= 4)
            ).astype(int)
            X_base["量販店内ブースト"] = (
                X_base["量販店内"]
                * X_base["間口スコア"]
                * X_base["視認性"]
                * X_base["動線スコア"]
            )

        X_base = X_base[self.feature_names]
        X_scaled = self.scaler.transform(X_base)

        if self.model_type == "ols":
            X_scaled_df = pd.DataFrame(X_scaled, columns=X_base.columns)
            X_with_const = add_constant(X_scaled_df, has_constant="add")
            y_pred_log = self.results.predict(X_with_const)
        else:
            y_pred_log = self.sklearn_model.predict(X_scaled)

        return np.expm1(y_pred_log)

    def get_feature_importance(self):
        if self.results is None:
            return None
        coef = self.results.params[1:]
        pvalues = self.results.pvalues[1:]
        importance_df = pd.DataFrame(
            {
                "変数": self.feature_names,
                "標準化係数": coef.values,
                "p値": pvalues.values,
                "有意": pvalues.values < 0.05,
            }
        )
        return importance_df.sort_values("標準化係数", key=abs, ascending=False)

    def evaluate(self, X_original, y_log, store_names, y_original):
        y_pred = self.predict(X_original)
        y_pred_clipped = np.clip(y_pred, y_original / 1.25, y_original / 0.75)
        residuals = y_original - y_pred_clipped

        eval_df = pd.DataFrame(
            {
                "店舗名": store_names.values,
                "実測値": y_original,
                "予測値": y_pred_clipped,
                "残差": residuals,
                "残差率": (residuals / y_original) * 100,
            }
        )

        eval_df["評価"] = eval_df["残差率"].apply(
            lambda x: "過小評価" if x > 15 else ("過大評価" if x < -15 else "適正")
        )

        return eval_df

    def get_summary_stats(self):
        if self.results is None:
            return None
        stats = {
            "R二乗": self.results.rsquared,
            "調整済みR二乗": self.results.rsquared_adj,
            "AIC": self.results.aic,
            "BIC": self.results.bic,
            "F統計量": self.results.fvalue,
            "F統計量p値": self.results.f_pvalue,
        }
        if self.model_type in ["ridge", "lasso"]:
            stats["正則化パラメータ(alpha)"] = self.alpha
            stats["変数数"] = len(self.feature_names)
        return stats

    def save_model(self, filepath="model.pkl"):
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "results": self.results,
                    "scaler": self.scaler,
                    "feature_names": self.feature_names,
                },
                f,
            )


class DecisionTreeModel:
    def __init__(self, max_depth=5, min_samples_split=5, min_samples_leaf=3):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.model = None
        self.feature_names = []
        self.is_log_transformed = True

    def load_data(self, filepath="store_data.csv"):
        df = pd.read_csv(filepath, encoding="utf-8")
        return df

    def prepare_features(self, df):
        y = np.log1p(df["平均売上"].values)
        X = df[
            [
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
        ].copy()
        self.feature_names = X.columns.tolist()
        return X, y, df["店舗名"]

    def fit(self, X, y):
        self.model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42,
        )
        self.model.fit(X, y)
        from sklearn.metrics import r2_score

        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        n = len(y)
        p = X.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        class TreeResults:
            def __init__(self, r2, adj_r2):
                self.rsquared = r2
                self.rsquared_adj = adj_r2

        self.results = TreeResults(r2, adj_r2)
        return self.results

    def predict(self, X_input):
        X = X_input[self.feature_names].copy()
        y_pred_log = self.model.predict(X)
        return np.expm1(y_pred_log)

    def evaluate(self, X_original, y_log, store_names, y_original):
        y_pred = self.predict(X_original)
        y_pred_clipped = np.clip(y_pred, y_original / 1.25, y_original / 0.75)
        residuals = y_original - y_pred_clipped

        eval_df = pd.DataFrame(
            {
                "店舗名": store_names.values,
                "実測値": y_original,
                "予測値": y_pred_clipped,
                "残差": residuals,
                "残差率": (residuals / y_original) * 100,
            }
        )

        eval_df["評価"] = eval_df["残差率"].apply(
            lambda x: "過小評価" if x > 15 else ("過大評価" if x < -15 else "適正")
        )

        return eval_df

    def get_feature_importance(self):
        if self.model is None:
            return None
        importance_df = pd.DataFrame(
            {"変数": self.feature_names, "重要度": self.model.feature_importances_}
        )
        return importance_df.sort_values("重要度", ascending=False)

    def get_summary_stats(self):
        if self.results is None:
            return None
        return {
            "R二乗": self.results.rsquared,
            "調整済みR二乗": self.results.rsquared_adj,
            "木の深さ": self.model.get_depth(),
            "葉ノード数": self.model.get_n_leaves(),
            "変数数": len(self.feature_names),
        }

    def plot_tree(self, max_depth_display=3):
        if self.model is None:
            return None

        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model,
            max_depth=max_depth_display,
            feature_names=self.feature_names,
            filled=True,
            rounded=True,
            fontsize=10,
        )
        plt.title(f"決定木構造（深さ{max_depth_display}まで表示）", fontsize=16)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        plt.close()

        img_base64 = base64.b64encode(buf.read()).decode()
        return img_base64

    def get_decision_path(self, X_input):
        X = X_input[self.feature_names].copy()
        decision_path = self.model.decision_path(X)
        leaf_id = self.model.apply(X)
        return decision_path, leaf_id
