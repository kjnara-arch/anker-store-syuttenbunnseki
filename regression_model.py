import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import pickle

LOCATION_TYPES = ["郊外商業", "アウトレット", "駅ビル駅近", "量販店内", "路面店"]


class LocationSpecificModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.data_counts = {}

    def load_data(self, filepath="store_data.csv"):
        return pd.read_csv(filepath, encoding="utf-8")

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
            X["プレミアム郊外商業"] * X["館売上規模"] * X["視認性"] * X["動線スコア"]
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
        return X, y

    def fit_all(self, df):
        X_all, y_all = self.prepare_features(df)

        for loc in LOCATION_TYPES:
            mask = df[loc] == 1
            X_loc = X_all[mask]
            y_loc = y_all[mask]

            self.data_counts[loc] = len(X_loc)

            if len(X_loc) < 3:
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_loc)

            alphas = np.logspace(-5, 1, 50)
            model = RidgeCV(alphas=alphas, cv=min(3, len(X_loc)), scoring="r2")
            model.fit(X_scaled, y_loc)

            self.models[loc] = model
            self.scalers[loc] = scaler

    def predict(self, location_type, X_input):
        if location_type not in self.models:
            return None

        model = self.models[location_type]
        scaler = self.scalers[location_type]

        # 入力データの整形
        X = X_input.copy()
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
            X["プレミアム郊外商業"] * X["館売上規模"] * X["視認性"] * X["動線スコア"]
        )
        X["郊外館規模補正"] = X["郊外商業"] * X["館売上規模"] * X["館売上規模_log"]
        X["路面店ブースト"] = X["路面店"] * X["商圏年収"] * X["インバウンド"]
        X["プレミアム量販店内"] = (
            X["量販店内"] * (X["間口スコア"] >= 4) * (X["視認性"] >= 4)
        ).astype(int)
        X["量販店内ブースト"] = (
            X["量販店内"] * X["間口スコア"] * X["視認性"] * X["動線スコア"]
        )

        # 立地タイプのフラグを1に設定
        for loc in LOCATION_TYPES:
            X[loc] = 1 if loc == location_type else 0

        X_ordered = X[self.feature_names]
        X_scaled = scaler.transform([X_ordered.values[0]])

        y_pred_log = model.predict(X_scaled)
        return np.expm1(y_pred_log)[0]

    def get_feature_importance(self, location_type):
        if location_type not in self.models:
            return None
        model = self.models[location_type]
        coef = model.coef_
        importance_df = pd.DataFrame({"変数": self.feature_names, "標準化係数": coef})
        return importance_df.sort_values("標準化係数", key=abs, ascending=False)

    def get_r2(self, location_type, df):
        if location_type not in self.models:
            return None
        mask = df[location_type] == 1
        X_loc, y_loc = self.prepare_features(df)
        X_loc = X_loc[mask]
        y_loc = y_loc[mask]

        scaler = self.scalers[location_type]
        X_scaled = scaler.transform(X_loc)
        y_pred_log = self.models[location_type].predict(X_scaled)

        from sklearn.metrics import r2_score

        return r2_score(y_loc, y_pred_log)
