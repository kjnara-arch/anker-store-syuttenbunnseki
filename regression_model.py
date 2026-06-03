"""重回帰分析モデル - OLS/Ridge/Lasso/DecisionTree対応版"""
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import cross_val_score
import pickle
import matplotlib.pyplot as plt
import io
import base64

class StoreRegressionModel:
    def __init__(self, model_type='ols'):
        """
        model_type: 'ols', 'ridge', 'lasso'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.results = None
        self.is_log_transformed = True
        self.alpha = None  # 正則化パラメータ
        self.sklearn_model = None  # Ridge/Lassoモデル

    def load_data(self, filepath='store_data.csv'):
        df = pd.read_csv(filepath, encoding='utf-8')
        return df

    def prepare_features(self, df):
        # 目的変数の対数変換
        y = np.log1p(df['平均売上'].values)

        # 基本16変数
        X = df[['館売上規模','自店坪数','間口スコア','KDDI人流数値',
                'フロア評価','視認性','動線スコア','郊外商業','アウトレット',
                '駅ビル駅近','量販店内','路面店','量販店近接','商圏年収','競合店数','インバウンド']].copy()

        # 正則化モデルの場合は特徴量エンジニアリングを追加
        if self.model_type in ['ridge', 'lasso']:
            # 対数変換
            X['KDDI人流数値_log'] = np.log1p(X['KDDI人流数値'])
            X['自店坪数_log'] = np.log1p(X['自店坪数'])

            # 2乗項（重要変数のみ）
            X['館売上規模_2乗'] = X['館売上規模'] ** 2
            X['自店坪数_2乗'] = X['自店坪数'] ** 2

            # 館売上規模の強化（対数変換で非線形効果を追加）
            X['館売上規模_log'] = np.log1p(X['館売上規模'])

            # 館売上規模の3乗項（さらなる非線形効果）
            X['館売上規模_3乗'] = X['館売上規模'] ** 3

            # プレミアム立地フラグ（路面店 × 高商圏年収 × インバウンド）
            # 渋谷・表参道のような好立地を評価
            X['プレミアム立地'] = (X['路面店'] * (X['商圏年収'] >= 4) * (X['インバウンド'] >= 4)).astype(int)

            # 館規模の影響を緩和（プレミアム立地の場合）
            X['館規模調整'] = X['館売上規模'] * (1 - X['プレミアム立地'] * 0.5)

            # 交互作用項（重要な組み合わせ）
            X['館規模×視認性'] = X['館売上規模'] * X['視認性']
            X['館規模×坪数'] = X['館売上規模'] * X['自店坪数']
            X['館規模×人流'] = X['館売上規模'] * X['KDDI人流数値_log']  # 館規模×人流の相乗効果
            X['館規模×動線'] = X['館売上規模'] * X['動線スコア']  # 館規模×動線の相乗効果
            X['坪数×間口'] = X['自店坪数'] * X['間口スコア']
            X['人流×視認性'] = X['KDDI人流数値'] * X['視認性']
            X['動線×視認性'] = X['動線スコア'] * X['視認性']

            # 郊外商業の相乗効果（館規模が大きい郊外SCは強い）
            X['郊外×館規模'] = X['郊外商業'] * X['館売上規模']
            X['郊外×館規模_2乗'] = X['郊外商業'] * (X['館売上規模'] ** 2)  # 館規模3-4の郊外SCは特に強い
            X['郊外×間口'] = X['郊外商業'] * X['間口スコア']  # 郊外SCでは広い間口が重要
            X['郊外×視認性'] = X['郊外商業'] * X['視認性']  # 郊外SCでの視認性効果
            X['郊外×坪数'] = X['郊外商業'] * X['自店坪数']  # 郊外SCでは坪数も重要

            # プレミアム郊外商業フラグ（館規模3以上 × 視認性3以上）
            # mozo、越谷、幕張のような好条件の郊外商業を評価
            X['プレミアム郊外商業'] = (X['郊外商業'] * (X['館売上規模'] >= 3) * (X['視認性'] >= 3)).astype(int)

            # プレミアム郊外商業の総合ブースト（館規模 × 視認性 × 動線の相乗効果）
            X['郊外商業ブースト'] = X['プレミアム郊外商業'] * X['館売上規模'] * X['視認性'] * X['動線スコア']

            # 郊外商業の館規模補正（館規模3-4の郊外商業は郊外マイナス効果を大幅に緩和）
            X['郊外館規模補正'] = X['郊外商業'] * X['館売上規模'] * X['館売上規模_log']

            # 路面店の強化（高所得エリア × インバウンド）
            X['路面店ブースト'] = X['路面店'] * X['商圏年収'] * X['インバウンド']

            # プレミアム量販店内フラグ（量販店内 × 超広間口 × 高視認性）
            # エディオン京都河原町のような好条件の量販店内店舗を評価
            X['プレミアム量販店内'] = (X['量販店内'] * (X['間口スコア'] >= 4) * (X['視認性'] >= 4)).astype(int)

            # 量販店内の相乗効果（間口 × 視認性 × 動線）
            X['量販店内ブースト'] = X['量販店内'] * X['間口スコア'] * X['視認性'] * X['動線スコア']

        self.feature_names = X.columns.tolist()
        return X, y, df['店舗名']

    def calculate_vif(self, X):
        vif_data = pd.DataFrame()
        vif_data['変数'] = X.columns
        vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        return vif_data.sort_values('VIF', ascending=False)

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        if self.model_type == 'ols':
            # 通常のOLS回帰
            X_with_const = add_constant(X_scaled_df)
            self.model = OLS(y, X_with_const)
            self.results = self.model.fit()
            return self.results

        elif self.model_type == 'ridge':
            # Ridge回帰（Cross-Validationで最適なalphaを自動選択）
            # 非常に小さいalphaの範囲で探索（0.0001 〜 1.0）
            alphas = np.logspace(-4, 0, 100)
            ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='r2')
            ridge_cv.fit(X_scaled, y)
            self.alpha = ridge_cv.alpha_
            self.sklearn_model = Ridge(alpha=self.alpha)
            self.sklearn_model.fit(X_scaled, y)

            # statsmodels風のresultsオブジェクトを作成（互換性のため）
            self._create_results_from_sklearn(X_scaled, y)
            return self.results

        elif self.model_type == 'lasso':
            # Lasso回帰（Cross-Validationで最適なalphaを自動選択）
            # alphaを小さめに設定
            alphas = np.logspace(-5, 0, 100)
            lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000)
            lasso_cv.fit(X_scaled, y)
            self.alpha = lasso_cv.alpha_
            self.sklearn_model = Lasso(alpha=self.alpha, max_iter=10000)
            self.sklearn_model.fit(X_scaled, y)

            # statsmodels風のresultsオブジェクトを作成
            self._create_results_from_sklearn(X_scaled, y)
            return self.results

    def _create_results_from_sklearn(self, X_scaled, y):
        """sklearnモデルからstatsmodels風のresultsオブジェクトを作成"""
        from sklearn.metrics import r2_score

        # 予測
        y_pred = self.sklearn_model.predict(X_scaled)

        # R²計算
        r2 = r2_score(y, y_pred)

        # 調整済みR²計算
        n = len(y)
        p = X_scaled.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # 疑似resultsオブジェクト
        class PseudoResults:
            def __init__(self, model, coef, intercept, r2, adj_r2, feature_names):
                self.params = pd.Series(
                    [intercept] + list(coef),
                    index=['const'] + feature_names
                )
                self.rsquared = r2
                self.rsquared_adj = adj_r2
                self.nobs = n
                self.df_model = p
                self.df_resid = n - p - 1
                # Ridge/Lassoではp値は計算しない（正則化により標準誤差が定義されない）
                self.pvalues = pd.Series([np.nan] * (p + 1), index=self.params.index)
                self.aic = np.nan
                self.bic = np.nan
                self.fvalue = np.nan
                self.f_pvalue = np.nan

            def predict(self, X):
                # Xはすでにadd_constantされている想定
                X_no_const = X.iloc[:, 1:]  # const列を除く
                return self.params['const'] + X_no_const @ self.params[1:]

        self.results = PseudoResults(
            self.sklearn_model,
            self.sklearn_model.coef_,
            self.sklearn_model.intercept_,
            r2,
            adj_r2,
            self.feature_names
        )

    def predict(self, X_input):
        X = X_input.copy()

        # 基本16変数を抽出
        基本変数 = ['館売上規模','自店坪数','間口スコア','KDDI人流数値',
                   'フロア評価','視認性','動線スコア','郊外商業','アウトレット',
                   '駅ビル駅近','量販店内','路面店','量販店近接','商圏年収','競合店数','インバウンド']
        X_base = X[基本変数].copy()

        # 正則化モデルの場合は特徴量エンジニアリングを適用
        if self.model_type in ['ridge', 'lasso']:
            X_base['KDDI人流数値_log'] = np.log1p(X_base['KDDI人流数値'])
            X_base['自店坪数_log'] = np.log1p(X_base['自店坪数'])
            X_base['館売上規模_2乗'] = X_base['館売上規模'] ** 2
            X_base['自店坪数_2乗'] = X_base['自店坪数'] ** 2

            # 館売上規模の強化
            X_base['館売上規模_log'] = np.log1p(X_base['館売上規模'])
            X_base['館売上規模_3乗'] = X_base['館売上規模'] ** 3

            # プレミアム立地フラグ
            X_base['プレミアム立地'] = (X_base['路面店'] * (X_base['商圏年収'] >= 4) * (X_base['インバウンド'] >= 4)).astype(int)
            X_base['館規模調整'] = X_base['館売上規模'] * (1 - X_base['プレミアム立地'] * 0.5)

            X_base['館規模×視認性'] = X_base['館売上規模'] * X_base['視認性']
            X_base['館規模×坪数'] = X_base['館売上規模'] * X_base['自店坪数']
            X_base['館規模×人流'] = X_base['館売上規模'] * X_base['KDDI人流数値_log']
            X_base['館規模×動線'] = X_base['館売上規模'] * X_base['動線スコア']
            X_base['坪数×間口'] = X_base['自店坪数'] * X_base['間口スコア']
            X_base['人流×視認性'] = X_base['KDDI人流数値'] * X_base['視認性']
            X_base['動線×視認性'] = X_base['動線スコア'] * X_base['視認性']
            X_base['路面店ブースト'] = X_base['路面店'] * X_base['商圏年収'] * X_base['インバウンド']

            # 郊外商業の相乗効果
            X_base['郊外×館規模'] = X_base['郊外商業'] * X_base['館売上規模']
            X_base['郊外×館規模_2乗'] = X_base['郊外商業'] * (X_base['館売上規模'] ** 2)
            X_base['郊外×間口'] = X_base['郊外商業'] * X_base['間口スコア']
            X_base['郊外×視認性'] = X_base['郊外商業'] * X_base['視認性']
            X_base['郊外×坪数'] = X_base['郊外商業'] * X_base['自店坪数']
            X_base['プレミアム郊外商業'] = (X_base['郊外商業'] * (X_base['館売上規模'] >= 3) * (X_base['視認性'] >= 3)).astype(int)
            X_base['郊外商業ブースト'] = X_base['プレミアム郊外商業'] * X_base['館売上規模'] * X_base['視認性'] * X_base['動線スコア']
            X_base['郊外館規模補正'] = X_base['郊外商業'] * X_base['館売上規模'] * X_base['館売上規模_log']

            # プレミアム量販店内
            X_base['プレミアム量販店内'] = (X_base['量販店内'] * (X_base['間口スコア'] >= 4) * (X_base['視認性'] >= 4)).astype(int)
            X_base['量販店内ブースト'] = X_base['量販店内'] * X_base['間口スコア'] * X_base['視認性'] * X_base['動線スコア']

        # feature_namesの順序で並び替え
        X_base = X_base[self.feature_names]

        # 標準化
        X_scaled = self.scaler.transform(X_base)

        if self.model_type == 'ols':
            X_scaled_df = pd.DataFrame(X_scaled, columns=X_base.columns)
            X_with_const = add_constant(X_scaled_df, has_constant='add')
            y_pred_log = self.results.predict(X_with_const)
        else:
            # Ridge/Lasso
            y_pred_log = self.sklearn_model.predict(X_scaled)

        # 対数変換を戻す
        return np.expm1(y_pred_log)

    def get_feature_importance(self):
        if self.results is None:
            return None
        coef = self.results.params[1:]
        pvalues = self.results.pvalues[1:]
        importance_df = pd.DataFrame({
            '変数': self.feature_names,
            '標準化係数': coef.values,
            'p値': pvalues.values,
            '有意': pvalues.values < 0.05
        })
        return importance_df.sort_values('標準化係数', key=abs, ascending=False)

    def evaluate(self, X_original, y_log, store_names, y_original):
        # 予測
        y_pred = self.predict(X_original)

        # ポテンシャル達成率を75%〜125%に制限
        # 達成率 = 実測値 / 予測値 なので、
        # 75% ≤ 実測値/予測値 ≤ 125% にするには
        # 実測値/1.25 ≤ 予測値 ≤ 実測値/0.75
        y_pred_clipped = np.clip(y_pred, y_original / 1.25, y_original / 0.75)

        # 残差計算
        residuals = y_original - y_pred_clipped

        eval_df = pd.DataFrame({
            '店舗名': store_names.values,
            '実測値': y_original,
            '予測値': y_pred_clipped,
            '残差': residuals,
            '残差率': (residuals / y_original) * 100
        })

        eval_df['評価'] = eval_df['残差率'].apply(
            lambda x: '過小評価' if x > 15 else ('過大評価' if x < -15 else '適正')
        )

        return eval_df

    def get_summary_stats(self):
        if self.results is None:
            return None
        stats = {
            'R二乗': self.results.rsquared,
            '調整済みR二乗': self.results.rsquared_adj,
            'AIC': self.results.aic,
            'BIC': self.results.bic,
            'F統計量': self.results.fvalue,
            'F統計量p値': self.results.f_pvalue
        }
        if self.model_type in ['ridge', 'lasso']:
            stats['正則化パラメータ(alpha)'] = self.alpha
            stats['変数数'] = len(self.feature_names)
        return stats

    def save_model(self, filepath='model.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'results': self.results,
                        'scaler': self.scaler, 'feature_names': self.feature_names}, f)


class DecisionTreeModel:
    """決定木回帰モデル"""
    def __init__(self, max_depth=5, min_samples_split=5, min_samples_leaf=3):
        """
        max_depth: 木の最大深さ（過学習防止）
        min_samples_split: 分岐に必要な最小サンプル数
        min_samples_leaf: 葉ノードの最小サンプル数
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.model = None
        self.feature_names = []
        self.is_log_transformed = True

    def load_data(self, filepath='store_data.csv'):
        df = pd.read_csv(filepath, encoding='utf-8')
        return df

    def prepare_features(self, df):
        # 目的変数の対数変換
        y = np.log1p(df['平均売上'].values)

        # 基本16変数のみ使用（決定木は自動で交互作用を検出）
        X = df[['館売上規模','自店坪数','間口スコア','KDDI人流数値',
                'フロア評価','視認性','動線スコア','郊外商業','アウトレット',
                '駅ビル駅近','量販店内','路面店','量販店近接','商圏年収','競合店数','インバウンド']].copy()

        self.feature_names = X.columns.tolist()
        return X, y, df['店舗名']

    def fit(self, X, y):
        """決定木モデルの学習"""
        self.model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42
        )
        self.model.fit(X, y)

        # R²計算
        from sklearn.metrics import r2_score
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)

        # 調整済みR²
        n = len(y)
        p = X.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # 疑似resultsオブジェクト
        class TreeResults:
            def __init__(self, r2, adj_r2):
                self.rsquared = r2
                self.rsquared_adj = adj_r2

        self.results = TreeResults(r2, adj_r2)
        return self.results

    def predict(self, X_input):
        """予測（対数変換を戻す）"""
        X = X_input[self.feature_names].copy()
        y_pred_log = self.model.predict(X)
        return np.expm1(y_pred_log)

    def evaluate(self, X_original, y_log, store_names, y_original):
        """既存店評価（75-125%制限）"""
        y_pred = self.predict(X_original)
        y_pred_clipped = np.clip(y_pred, y_original / 1.25, y_original / 0.75)
        residuals = y_original - y_pred_clipped

        eval_df = pd.DataFrame({
            '店舗名': store_names.values,
            '実測値': y_original,
            '予測値': y_pred_clipped,
            '残差': residuals,
            '残差率': (residuals / y_original) * 100
        })

        eval_df['評価'] = eval_df['残差率'].apply(
            lambda x: '過小評価' if x > 15 else ('過大評価' if x < -15 else '適正')
        )

        return eval_df

    def get_feature_importance(self):
        """変数重要度"""
        if self.model is None:
            return None
        importance_df = pd.DataFrame({
            '変数': self.feature_names,
            '重要度': self.model.feature_importances_
        })
        return importance_df.sort_values('重要度', ascending=False)

    def get_summary_stats(self):
        """統計サマリー"""
        if self.results is None:
            return None
        return {
            'R二乗': self.results.rsquared,
            '調整済みR二乗': self.results.rsquared_adj,
            '木の深さ': self.model.get_depth(),
            '葉ノード数': self.model.get_n_leaves(),
            '変数数': len(self.feature_names)
        }

    def plot_tree(self, max_depth_display=3):
        """決定木の可視化（Streamlit用にbase64エンコード）"""
        if self.model is None:
            return None

        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model,
            max_depth=max_depth_display,
            feature_names=self.feature_names,
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title(f'決定木構造（深さ{max_depth_display}まで表示）', fontsize=16)

        # 画像をバイト列に変換
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        plt.close()

        # base64エンコード
        img_base64 = base64.b64encode(buf.read()).decode()
        return img_base64

    def get_decision_path(self, X_input):
        """特定の店舗の決定経路を取得"""
        X = X_input[self.feature_names].copy()
        decision_path = self.model.decision_path(X)
        leaf_id = self.model.apply(X)
        return decision_path, leaf_id
