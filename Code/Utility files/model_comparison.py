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



class CrossValidation:
    """
    Class to perform cross-validation on a list of DataFrames using Weighted Least Squares (WLS) regression.
    
    The class allows fitting WLS models, performing cross-validation, and calculating various metrics 
    such as RMSE, R2, AIC, BIC, and correlation coefficients. It also supports plotting residuals 
    and summarizing the results.

    Attributes:
    ----------
    dfs : list
        A list of pandas DataFrames to be used for cross-validation.
    predictors : list
        A list of column names to be used as predictors in the regression models.
    outcome : str
        The column name of the outcome variable.
    n_splits : int, optional
        The number of splits for cross-validation, default is 10.
    gkf : GroupKFold
        GroupKFold object for performing grouped cross-validation.
    results_values : dict
        Dictionary to store results metrics for each DataFrame.
    summary_outputs : list
        List to store summary outputs of the WLS models.
    residuals_dict : dict
        Dictionary to store residuals for each fold in cross-validation.
    predictions_dict : dict
        Dictionary to store predictions for each fold in cross-validation.

    Methods:
    -------
    fit():
        Fits WLS models on the DataFrames and performs cross-validation to calculate metrics.
    fit_wls(df, i):
        Fits a WLS model to the specified DataFrame and returns various metrics and the model results.
    print_summaries():
        Prints the summary output for the fitted WLS models.
    cross_validation(df, i):
        Performs cross-validation on the specified DataFrame and returns mean RMSE and R2 scores.
    plot_residuals():
        Plots the distribution of residuals for each fold in cross-validation.
    calculate_correlations_median(grouping_column):
        Calculates the correlation between predictors and the outcome, grouped by the specified column.
    """
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
        self.results_wls_dict = {}  # Initialize a dictionary to store the results_wls objects
        results_values = {}

        for i, df in enumerate(self.dfs, start=1):  # start=1 to make the index 1-based
            r2_full, max_coeff_predictor, max_coeff_value, aic, bic, results_wls = self.fit_wls(df, i)
            CV_rmse_mean, CV_r2_mean = self.cross_validation(df, i)

            # Store the metrics for the current DataFrame
            results_values[f'DataFrame {i}'] = (CV_rmse_mean, CV_r2_mean, r2_full, max_coeff_predictor, max_coeff_value, aic, bic)

            # Store the results_wls object for the current DataFrame
            self.results_wls_dict[i] = results_wls

        # Convert the dictionary to a DataFrame and display it
        results_table = pd.DataFrame(list(results_values.items()), columns=['DataFrame', 'Metrics'])
        results_table[['Mean RMSE (CV)', 'R2 (CV)', 'R2 (Full)', 'Max Coeff Predictor', 'Max Coeff Value', 'AIC', 'BIC']] = pd.DataFrame(results_table.Metrics.tolist(), index= results_table.index)
        results_table = results_table.drop(columns=['Metrics'])
        print(results_table)

        # Print the number of folds
        print(f'Number of folds used: {self.n_splits}')


    def fit_wls(self, df, i):
        
        #Fit the OLS model to obtain weights
        X = sm.add_constant(df[self.predictors])  # Adding the intercept term
        y = df[self.outcome]
        model_OLS = sm.OLS(y, X)
        OLS_results = model_OLS.fit()
        OLS_residuals = y - OLS_results.predict(X) #redisudals on entire df
        weights = 1.0 / (OLS_residuals ** 2) #weights inverse of residuals of entire df
        

        # Fit a WLS model on the entire DataFrame using the estimated weights
        model_wls = sm.WLS(y, X, weights=weights)
        results_wls = model_wls.fit()  
        p_values = results_wls.pvalues

        # Obtain largest absoulte significant coefficients
        coefficients = results_wls.params
        significant_coefficients = coefficients[p_values < 0.05]

        if 'const' in significant_coefficients:
            significant_coefficients = significant_coefficients.drop('const') #ignore intercept when finding the largest absolute coef

        max_coeff_predictor = significant_coefficients.abs().idxmax()
        max_coeff_value = significant_coefficients[max_coeff_predictor]

        # Calculate R2 for the entire DataFrame
        r2_full = results_wls.rsquared

        #Get the summary output
        summary = results_wls.summary()

        # Get AIC and BIC of the model
        aic = results_wls.aic
        bic = results_wls.bic

        return r2_full, max_coeff_predictor, max_coeff_value, aic, bic, results_wls
    
    def print_summaries(self):
        for df_number, results_wls in self.results_wls_dict.items():
            print(f"Summary for DataFrame {df_number}:")
            print(results_wls.summary())

    def cross_validation(self, df, i):
        CV_rmse_scores = [] 
        CV_r2_scores = []  
        X = sm.add_constant(df[self.predictors])  # Adding the intercept term
        y = df[self.outcome]
        self.predictions_dict = {}

        for fold, (train_index, test_index) in enumerate(self.gkf.split(df[self.predictors], df[self.outcome], df['PCS_ESE']), start=1):
            # Split the data into training and test sets
            train, test = df.iloc[train_index], df.iloc[test_index]

            # Fit an OLS model on the training set to estimate weights
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            OLS_model = sm.OLS(y_train, X_train)
            OLS_results = OLS_model.fit()

            # Calculate residuals for the training set
            residuals_train = y_train - OLS_results.predict(X_train)

            # Estimate weights as the inverse of the squared residuals for the training set
            weights_train = 1.0 / (residuals_train ** 2)

            # Fit a WLS model on the training set using the estimated weights
            model_wls = sm.WLS(y_train, X_train, weights=weights_train)
            results_wls = model_wls.fit()

            # Evaluate the model on the test set
            X_test = X.iloc[test_index]
            y_test = y.iloc[test_index]
            predictions = results_wls.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)  # Calculate RMSE
            CV_rmse_scores.append(rmse)
            CV_r2_scores.append(results_wls.rsquared)

            # Store residuals and predictions for this fold
            residuals_fold = y_test - predictions
            self.residuals_dict[f'DataFrame {i}_Fold {fold}'] = residuals_fold
            if f'DataFrame {i}' not in self.predictions_dict:
                self.predictions_dict[f'DataFrame {i}'] = []
            self.predictions_dict[f'DataFrame {i}'].append((y_test, predictions, df.loc[test_index, 'PCS_ESE']))

        # Calculate the mean RMSE and R2 scores for the current DataFrame
        CV_rmse_mean = np.mean(CV_rmse_scores)
        CV_r2_mean = np.mean(CV_r2_scores)

        return CV_rmse_mean, CV_r2_mean



    def plot_residuals(self):
        if not self.residuals_dict:
            raise ValueError("residuals_dict is empty or not properly populated.")

        # Aggregate residuals by model
        model_residuals = {}
        for key, residuals in self.residuals_dict.items():
            model_name = key.split('_')[0]  # Assuming the model name is the first part of the key
            if model_name not in model_residuals:
                model_residuals[model_name] = residuals
            else:
                model_residuals[model_name] = pd.concat([model_residuals[model_name], residuals])

        num_models = len(model_residuals)
        num_cols = 3
        num_rows = math.ceil(num_models / num_cols)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 6 * num_rows))
        axes = axes.flatten()
        for ax in axes[num_models:]:
            fig.delaxes(ax)

        for ax, (model_name, residuals) in zip(axes, model_residuals.items()):
            sns.histplot(residuals, kde=True, ax=ax)
            ax.set_title(f'Residuals Distribution for {model_name}')
            ax.set_xlabel('Residuals')
            ax.set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()
    
    def calculate_correlations_median(self, grouping_column):
        correlations = {}
        for i, df in enumerate(self.dfs, start=1):
            # Select only required columns
            df = df[self.predictors + [self.outcome, grouping_column]]
            # Group by grouping_column and calculate median coordinates
            df_grouped = df.groupby(grouping_column).median()

            if len(self.predictors) == 1:
                # If there's only one predictor, calculate the correlation and p-value directly
                corr_value, p_value = spearmanr(df_grouped[self.predictors[0]], df_grouped[self.outcome])
                corr_values = pd.Series([corr_value], index=self.predictors)
                p_values = pd.Series([p_value], index=self.predictors)
            else:
                # Calculate correlation with outcome for each predictor
                corr_values, p_values = spearmanr(df_grouped)
                corr_values = pd.Series(corr_values[-1, :-1], index=self.predictors)
                p_values = pd.Series(p_values[-1, :-1], index=self.predictors)

            # Get predictor with highest absolute correlation
            max_corr_predictor = corr_values.abs().idxmax()
            max_corr_value = corr_values[max_corr_predictor]
            max_p_value = p_values[max_corr_predictor]
            correlations[f'DataFrame {i}'] = (max_corr_predictor, max_corr_value, max_p_value)

        correlations_df = pd.DataFrame(list(correlations.items()), columns=['DataFrame', 'Max Correlation'])
        correlations_df[['Predictor', 'Correlation', 'P-value']] = pd.DataFrame(correlations_df['Max Correlation'].tolist(), index=correlations_df.index)
        correlations_df = correlations_df.drop(columns=['Max Correlation'])
        print(correlations_df)
