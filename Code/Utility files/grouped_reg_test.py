import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
import seaborn as sns
import matplotlib.patches as mpatches
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.stats import shapiro    
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy
from scipy.stats import spearmanr
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm



class Grouped_WLS:
    def __init__(self, dfs, predictors, outcome, n_splits=10):
        self.dfs = dfs
        self.predictors = predictors
        self.outcome = outcome
        self.n_splits = n_splits
        self.gkf = GroupKFold(n_splits=n_splits)
        self.results_values = {}
        self.summary_outputs = []
        self.residuals_dict = {}
        self.predictions_dict = {}

    def fit(self):
        self.results_wls_dict = {}
        results_values = {}

        for i, df in enumerate(self.dfs, start=1):
            df_grouped = df.groupby('PCS_ESE')[self.predictors + [self.outcome]].median().reset_index()
            r2_full, max_coeff_predictor, max_coeff_value, aic, bic, results_wls = self.fit_wls(df_grouped, i)
            CV_rmse_mean, CV_r2_mean = self.cross_validation(df, i)

            results_values[f'DataFrame {i}'] = (CV_rmse_mean, CV_r2_mean, r2_full, max_coeff_predictor, max_coeff_value, aic, bic)
            self.results_wls_dict[i] = results_wls

        results_table = pd.DataFrame(list(results_values.items()), columns=['DataFrame', 'Metrics'])
        results_table[['Mean RMSE (CV)', 'R2 (CV)', 'R2 (Full)', 'Max Coeff Predictor', 'Max Coeff Value', 'AIC', 'BIC']] = pd.DataFrame(results_table.Metrics.tolist(), index= results_table.index)
        results_table = results_table.drop(columns=['Metrics'])
        print(results_table)
        print(f'Number of folds used: {self.n_splits}')

    def fit_wls(self, df_grouped, i):
        X = sm.add_constant(df_grouped[self.predictors])
        y = df_grouped[self.outcome]
        model_OLS = sm.OLS(y, X)
        OLS_results = model_OLS.fit()
        OLS_residuals = y - OLS_results.predict(X)
        weights = 1.0 / (OLS_residuals ** 2)

        model_wls = sm.WLS(y, X, weights=weights)
        results_wls = model_wls.fit()
        p_values = results_wls.pvalues

        coefficients = results_wls.params
        significant_coefficients = coefficients[p_values < 0.05]
        if 'const' in significant_coefficients:
            significant_coefficients = significant_coefficients.drop('const')

        max_coeff_predictor = significant_coefficients.abs().idxmax()
        max_coeff_value = significant_coefficients[max_coeff_predictor]
        r2_full = results_wls.rsquared
        aic = results_wls.aic
        bic = results_wls.bic

        return r2_full, max_coeff_predictor, max_coeff_value, aic, bic, results_wls

    def cross_validation(self, df, i):
        CV_rmse_scores = []
        CV_r2_scores = []
        self.predictions_dict = {}

        for fold, (train_index, test_index) in enumerate(self.gkf.split(df[self.predictors], df[self.outcome], df['PCS_ESE']), start=1):
            train, test = df.iloc[train_index], df.iloc[test_index]
            train_grouped = train.groupby('PCS_ESE')[self.predictors + [self.outcome]].median().reset_index()
            test_grouped = test.groupby('PCS_ESE')[self.predictors + [self.outcome]].median().reset_index()

            X_train = sm.add_constant(train_grouped[self.predictors])
            y_train = train_grouped[self.outcome]
            OLS_model = sm.OLS(y_train, X_train)
            OLS_results = OLS_model.fit()
            residuals_train = y_train - OLS_results.predict(X_train)
            weights_train = 1.0 / (residuals_train ** 2)

            model_wls = sm.WLS(y_train, X_train, weights=weights_train)
            results_wls = model_wls.fit()

            X_test = sm.add_constant(test_grouped[self.predictors])
            y_test = test_grouped[self.outcome]
            predictions = results_wls.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            CV_rmse_scores.append(rmse)
            CV_r2_scores.append(results_wls.rsquared)

            residuals_fold = y_test - predictions
            self.residuals_dict[f'DataFrame {i}_Fold {fold}'] = residuals_fold
            if f'DataFrame {i}' not in self.predictions_dict:
                self.predictions_dict[f'DataFrame {i}'] = []
            self.predictions_dict[f'DataFrame {i}'].append((y_test, predictions, test_grouped['PCS_ESE']))

        CV_rmse_mean = np.mean(CV_rmse_scores)
        CV_r2_mean = np.mean(CV_r2_scores)

        return CV_rmse_mean, CV_r2_mean
