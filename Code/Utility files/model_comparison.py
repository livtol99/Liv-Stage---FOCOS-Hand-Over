import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
import seaborn as sns
import matplotlib.patches as mpatches

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
        self.summary_outputs = []
        results_values = {}
        self.predictions_dict = {}  # Initialize a dictionary to store the predictions

        for i, df in enumerate(self.dfs, start=1):  # start=1 to make the index 1-based
            rmse_scores = []  # Initialize the RMSE scores for each DataFrame
            r2_scores = []  # Initialize the R2 scores for each DataFrame
            self.predictions_dict[f'DataFrame {i}'] = []  # Initialize a list to store the predictions for the current DataFrame
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

                # Store the true and predicted values for the current fold
                self.predictions_dict[f'DataFrame {i}'].append((y_test, predictions, test['PCS_ESE']))

        # Fit a WLS model on the entire DataFrame after the inner loop
            X = sm.add_constant(df[self.predictors])  # Adding the intercept term
            y = df[self.outcome]
            model = sm.OLS(y, X)
            results = model.fit()

            # Calculate residuals
            residuals = y - results.predict(X)

            # Store residuals for the entire DataFrame
            self.residuals_dict[f'DataFrame {i}'] = residuals

            # Estimate weights as the inverse of the squared residuals
            weights = 1.0 / (residuals ** 2)

            # Fit a WLS model using the estimated weights
            model_wls = sm.WLS(y, X, weights=weights)
            results_wls = model_wls.fit()

            # Append the summary of the model of the current DataFrame to the list
            self.summary_outputs.append((i, results_wls.summary()))

            # Calculate the mean RMSE and R2 scores for the current DataFrame
            rmse_mean = np.mean(rmse_scores)
            r2_mean = np.mean(r2_scores)
            r2_full = results_wls.rsquared  # Calculate R2 for the entire DataFrame
            results_values[f'DataFrame {i}'] = (rmse_mean, r2_mean, r2_full)  # Store the metrics for the current DataFrame

        # Store the predicted values for the entire DataFrame
        self.predictions_dict[f'DataFrame {i}'].append((y, results_wls.predict(X), df['PCS_ESE']))
        # Add the predicted values to the DataFrame
        df['Predicted'] = results_wls.predict(X)

        # Convert the dictionary to a DataFrame and display it
        results_table = pd.DataFrame(list(results_values.items()), columns=['DataFrame', 'Metrics'])
        results_table[['Mean RMSE (CV)', 'R2 (CV)', 'R2 (Full)']] = pd.DataFrame(results_table.Metrics.tolist(), index= results_table.index)
        results_table = results_table.drop(columns=['Metrics'])
        print(results_table)

        # Print the number of folds used in the cross-validation
        print(f"\nNumber of folds used in the group fold cross-validation: {self.gkf.n_splits}")
    
    def print_summaries(self):
        for df_number, summary in self.summary_outputs:
            print(f"Summary for DataFrame {df_number}:")
            print(summary)

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
    

    def calculate_correlations(self, grouping_column):
        correlations = {}
        for i, df in enumerate(self.dfs, start=1):
            # Select only required columns
            df = df[self.predictors + [self.outcome, grouping_column]]
            # Group by grouping_column and calculate mean coordinates
            df_grouped = df.groupby(grouping_column).mean()
            # Calculate correlation with outcome for each predictor
            corr_values = df_grouped.corr()[self.outcome].drop(self.outcome)
            # Get predictor with highest correlation
            max_corr_predictor = corr_values.idxmax()
            max_corr_value = corr_values.max()
            correlations[f'DataFrame {i}'] = (max_corr_predictor, max_corr_value)
        correlations_df = pd.DataFrame(list(correlations.items()), columns=['DataFrame', 'Max Correlation'])
        correlations_df[['Predictor', 'Correlation']] = pd.DataFrame(correlations_df['Max Correlation'].tolist(), index=correlations_df.index)
        correlations_df = correlations_df.drop(columns=['Max Correlation'])
        print(correlations_df)


    def plot_mean_true_vs_predicted(self):
        # Determine the layout of the subplots
        num_dfs = len(self.predictions_dict)
        num_cols = 3  # Change this to 3 for a 3x3 grid
        num_rows = math.ceil(num_dfs / num_cols)

        # Initialize a dictionary to store the mean predictions and true values for each PCS_ESE group
        mean_values_dict = {}

        # Calculate the mean predicted and true values for each PCS_ESE group
        for df_name, values in self.predictions_dict.items():
            mean_values_dict[df_name] = {}
            for y_test, predictions, pcs_ese in values:  # Expect three values here
                for pcs_ese_value in pcs_ese.unique():
                    mask = pcs_ese == pcs_ese_value
                    mean_values_dict[df_name][pcs_ese_value] = (predictions[mask].mean(), y_test[mask].mean())

        # Get all unique PCS_ESE values from all dataframes
        all_pcs_ese_values = set()
        for _, values in self.predictions_dict.items():
            for y_test, predictions, pcs_ese in values:
                all_pcs_ese_values.update(pcs_ese.unique())

        # Sort the PCS_ESE values
        all_pcs_ese_values = sorted(all_pcs_ese_values)

        # Create a color palette
        color_palette = sns.color_palette('hsv', len(all_pcs_ese_values))

        # Map each PCS_ESE group to a color
        color_map = {pcs_ese_value: color for pcs_ese_value, color in zip(all_pcs_ese_values, color_palette)}

        # Create a figure and a grid of subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 6 * num_rows))

        # Flatten the axes array and remove extra subplots
        axes = axes.flatten()
        for ax in axes[num_dfs:]:
            fig.delaxes(ax)

        # Plot the mean true vs mean predicted values for each PCS_ESE group
        for ax, (df_name, pcs_ese_values) in zip(axes, mean_values_dict.items()):
            for pcs_ese_value, (mean_prediction, mean_true_value) in pcs_ese_values.items():
                color = color_map.get(pcs_ese_value, 'black')  # Use black as the default color
                sns.scatterplot(x=[mean_true_value], y=[mean_prediction], ax=ax, color=color)
            ax.set_title(f'Mean True vs Mean Predicted Values for {df_name}')
            ax.set_xlabel('Mean True Values')
            ax.set_ylabel('Mean Predicted Values')

        # Display the figure
        plt.tight_layout()
        plt.show()

        # Create a new figure for the legend
        fig_legend = plt.figure(figsize=(10, 2))  # Adjust the figure size as needed

        # Add the legend to the new figure
        legend_patches = [mpatches.Patch(color=color, label=pcs_ese_value) for pcs_ese_value, color in color_map.items()]
        plt.legend(handles=legend_patches, title='PCS_ESE', loc='center', ncol=3)  # Adjust the number of columns as needed
        plt.axis('off')

        # Display the legend
        plt.show()
    
    def plot_residuals_vs_fitted(self):
        num_dfs = len(self.predictions_dict)
        num_cols = 3
        num_rows = math.ceil(num_dfs / num_cols)

        residuals_dict = {}
        fitted_dict = {}

        for df_name, values in self.predictions_dict.items():
            residuals_dict[df_name] = {}
            fitted_dict[df_name] = {}
            for y_test, predictions, pcs_ese in values:
                for pcs_ese_value in pcs_ese.unique():
                    mask = pcs_ese == pcs_ese_value
                    residuals = y_test[mask] - predictions[mask]
                    residuals_dict[df_name][pcs_ese_value] = residuals
                    fitted_dict[df_name][pcs_ese_value] = predictions[mask]

        all_pcs_ese_values = set()
        for _, values in self.predictions_dict.items():
            for _, _, pcs_ese in values:
                all_pcs_ese_values.update(pcs_ese.unique())

        all_pcs_ese_values = sorted(all_pcs_ese_values)

        color_palette = sns.color_palette('hsv', len(all_pcs_ese_values))

        color_map = {pcs_ese_value: color for pcs_ese_value, color in zip(all_pcs_ese_values, color_palette)}

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 6 * num_rows))

        axes = axes.flatten()
        for ax in axes[num_dfs:]:
            fig.delaxes(ax)

        for ax, (df_name, pcs_ese_values) in zip(axes, residuals_dict.items()):
            for pcs_ese_value, residuals in pcs_ese_values.items():
                color = color_map.get(pcs_ese_value, 'black')
                fitted_values = fitted_dict[df_name][pcs_ese_value]
                sns.scatterplot(x=fitted_values, y=residuals, ax=ax, color=color)
            ax.axhline(0, color='red', linestyle='--')  # Add a horizontal line at y=0
            ax.set_title(f'Residuals vs Fitted for {df_name}')
            ax.set_xlabel('Fitted values')
            ax.set_ylabel('Residuals')

        plt.tight_layout()
        plt.show()

        fig_legend = plt.figure(figsize=(10, 2))

        legend_patches = [mpatches.Patch(color=color, label=pcs_ese_value) for pcs_ese_value, color in color_map.items()]
        plt.legend(handles=legend_patches, title='PCS_ESE', loc='center', ncol=3)
        plt.axis('off')

        plt.show()
    def plot_grouped_residuals_vs_fitted(self):
        num_dfs = len(self.predictions_dict)
        num_cols = 3
        num_rows = math.ceil(num_dfs / num_cols)

        residuals_dict = {}
        fitted_dict = {}

        for df_name, values in self.predictions_dict.items():
            residuals_dict[df_name] = {}
            fitted_dict[df_name] = {}
            for y_test, predictions, pcs_ese in values:
                for pcs_ese_value in pcs_ese.unique():
                    mask = pcs_ese == pcs_ese_value
                    residuals = y_test[mask] - predictions[mask]
                    residuals_dict[df_name][pcs_ese_value] = residuals.mean()
                    fitted_dict[df_name][pcs_ese_value] = predictions[mask].mean()

        all_pcs_ese_values = set()
        for _, values in self.predictions_dict.items():
            for _, _, pcs_ese in values:
                all_pcs_ese_values.update(pcs_ese.unique())

        all_pcs_ese_values = sorted(all_pcs_ese_values)

        color_palette = sns.color_palette('hsv', len(all_pcs_ese_values))

        color_map = {pcs_ese_value: color for pcs_ese_value, color in zip(all_pcs_ese_values, color_palette)}

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 6 * num_rows))

        axes = axes.flatten()
        for ax in axes[num_dfs:]:
            fig.delaxes(ax)

        for ax, (df_name, pcs_ese_values) in zip(axes, residuals_dict.items()):
            for pcs_ese_value, residuals in pcs_ese_values.items():
                color = color_map.get(pcs_ese_value, 'black')
                fitted_values = fitted_dict[df_name][pcs_ese_value]
                sns.scatterplot(x=[fitted_values], y=[residuals], ax=ax, color=color)
            ax.axhline(0, color='red', linestyle='--')  # Add a horizontal line at y=0
            ax.set_title(f'Mean Residuals vs Mean Fitted for {df_name}')
            ax.set_xlabel('Mean Fitted values')
            ax.set_ylabel('Mean Residuals')

        plt.tight_layout()
        plt.show()

        fig_legend = plt.figure(figsize=(10, 2))

        legend_patches = [mpatches.Patch(color=color, label=pcs_ese_value) for pcs_ese_value, color in color_map.items()]
        plt.legend(handles=legend_patches, title='PCS_ESE', loc='center', ncol=3)
        plt.axis('off')

        plt.show()
    
   
   