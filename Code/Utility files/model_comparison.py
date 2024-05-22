from sklearn.model_selection import KFold
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import GroupKFold
import math
import matplotlib.pyplot as plt

class CrossValidation:
    def __init__(self, dfs, predictors, outcome, n_splits=10):
        self.dfs = dfs
        self.predictors = predictors
        self.outcome = outcome
        self.n_splits = n_splits
        self.gkf = GroupKFold(n_splits=n_splits)
        self.results_values = {}
        self.summary_outputs = []
        self.residuals_dict = {}

    def fit(self):
        best_rmse = np.inf
        best_r2 = -np.inf
        best_model = None
        best_df = None  
        best_fold = None
        summary_outputs = []
        results_values = {}

        for i, df in enumerate(self.dfs, start=1):  # start=1 to make the index 1-based
            rmse_scores = []  # Initialize the RMSE scores for each DataFrame
            r2_scores = []  # Initialize the R2 scores for each DataFrame
            for fold, (train_index, test_index) in enumerate(self.gkf.split(df[self.predictors], df[self.outcome], df['PCS_ESE']), start=1): #inner loop performs loops over all kfolds, fits the model, and calculates RMSE
                # Split the data into training and test sets
                train, test = df.iloc[train_index], df.iloc[test_index]

                # Fit an initial OLS model on the training set
                X_train = sm.add_constant(train[self.predictors])  # Adding the intercept term
                y_train = train[self.outcome]
                model = sm.OLS(y_train, X_train)
                results = model.fit()

                # Calculate residuals
                residuals = y_train - results.predict(X_train)

                # Estimate weights as the inverse of the squared residuals
                weights = 1.0 / (residuals ** 2)

                # Fit a WLS model using the estimated weights
                model_wls = sm.WLS(y_train, X_train, weights=weights)
                results_wls = model_wls.fit()

                # Evaluate the model on the test set
                X_test = sm.add_constant(test[self.predictors])
                y_test = test[self.outcome]
                predictions = results_wls.predict(X_test)
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)  # Calculate RMSE
                rmse_scores.append(rmse)
                r2_scores.append(results_wls.rsquared)

                # Check if this model is better than the previous best
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_r2 = results_wls.rsquared
                    best_model = results_wls
                    best_df = i  # Update the index of the DataFrame that gives the best model
                    best_fold = fold  # Update the fold number

            # Fit a WLS model on the entire DataFrame after the inner loop
            X = sm.add_constant(df[self.predictors])  # Adding the intercept term
            y = df[self.outcome]
            model = sm.OLS(y, X)
            results = model.fit()

            # Calculate residuals
            residuals = y - results.predict(X)

            # Estimate weights as the inverse of the squared residuals
            weights = 1.0 / (residuals ** 2)

            # Fit a WLS model using the estimated weights
            model_wls = sm.WLS(y, X, weights=weights)
            results_wls = model_wls.fit()

            # Append the summary of the model of the current DataFrame to the list
            summary_outputs.append(results_wls.summary())

            # Calculate the mean RMSE and R2 scores for the current DataFrame
            rmse_mean = np.mean(rmse_scores)
            r2_mean = np.mean(r2_scores)
            r2_full = results_wls.rsquared  # Calculate R2 for the entire DataFrame
            results_values[f'DataFrame {i}'] = (rmse_mean, r2_mean, r2_full)  # Store the metrics for the current DataFrame

        # Print the summary statistics of the best WLS regression model
        print(f'Best model from DataFrame {best_df}, Fold {best_fold} has RMSE: {best_rmse} and R2: {best_r2}')
        print(best_model.summary())

        # Convert the dictionary to a DataFrame and display it
        results_table = pd.DataFrame(list(results_values.items()), columns=['DataFrame', 'Metrics'])
        results_table[['RMSE', 'R2 (CV)', 'R2 (Full)']] = pd.DataFrame(results_table.Metrics.tolist(), index= results_table.index)
        results_table = results_table.drop(columns=['Metrics'])
        print(results_table)

    def plot_residuals(self):
        num_dfs = len(self.residuals_dict)
        num_cols = 3
        num_rows = math.ceil(num_dfs / num_cols)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 6 * num_rows))
        axes = axes.flatten()
        for ax in axes[num_dfs:]:
            fig.delaxes(ax)

        for ax, (df_fold, residuals) in zip(axes, self.residuals_dict.items()):
            sns.histplot(residuals, kde=True, ax=ax)
            ax.set_title(f'Residuals Distribution for {df_fold}')
            ax.set_xlabel('Residuals')
            ax.set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()