# Import Pandas, Scipy:
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
#import numpy as np
import io
import scipy.stats as sps
import seaborn as sns

# Read DOW JONES CSV data file:
df_DJIA = pd.read_csv(r'C:\DJIA.csv')
# Read SP500 CSV data file:
df_SP500 = pd.read_csv(r'C:\SP500.csv')
# Read NASDAQ CSV data file:
df_NASDAQ = pd.read_csv(r'C:\NASDAQCOM.csv')

# Dropping the rows that has missing values in each dataset:
def clean_missing_values(df):
    return df.dropna()

# Dictionary to store original dataframes
dataframes = {
    "DJIA": df_DJIA,
    "SP500": df_SP500,
    "NASDAQ": df_NASDAQ
}

# Cleaning missing values for each dataframe
cleaned_dataframes = {name: clean_missing_values(df) for name, df in dataframes.items()}
# Access cleaned dataframes using keys
df_DJIA_clean = cleaned_dataframes["DJIA"]; df_SP500_clean = cleaned_dataframes["SP500"]
df_NASDAQ_clean = cleaned_dataframes["NASDAQ"]


def calculate_skewness(dataframes, columns):
    skewness_values = {}
    for df in dataframes: # Directly iterate over the dataframes
        for column in columns:
            if column in df.columns:  # Check if the column exists in the dataframe
                skewness_values[column] = sps.skew(df[column])  # Calculate skewness
    return skewness_values

# Datasets
dataframes = [df_SP500_clean, df_DJIA_clean, df_NASDAQ_clean]
columns = ['SP500', 'DJIA', 'NASDAQCOM']

# Call the function
skewness_results = calculate_skewness(dataframes, columns)
print(f"Skewness Results: {skewness_results}")

# General function to export dataframes info to a file
def export_df_info_to_file(df, filename):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    with open(filename, "w", encoding="utf-8") as f:
        f.write(s)

export_df_info_to_file(df_SP500_clean, "df_SP500_clean"); (export_df_info_to_file(df_DJIA_clean,
                                                                                  "df_DJIA_clean"))
export_df_info_to_file(df_NASDAQ_clean, "df_NASDAQ_clean")


# Extracting the head of the Dataset of SP500:
# Could be done on DJ IA and NASDAQ databases

summary = df_SP500_clean.head()
summary.to_clipboard()

# Describe of the Dataset of SP500

summary = df_SP500_clean.describe()
summary.to_clipboard()

# Finding the column and row values of the datasets

print(df_SP500_clean.shape); print(df_DJIA_clean.shape); print(df_NASDAQ_clean.shape)

# Creating histogram and define functions for display the box plots of the Dataframes
def plot_boxplot_and_hist(data):
    """Plots a boxplot and histogram for the given specified columns to identify outliers."""
    columns = ['SP500', 'DJIA', 'NASDAQCOM']

    for df in data:
        for column in columns:
            if column in df.columns:
                sns.boxplot(y=df[column])
                sns.set_theme(color_codes = True)
                plt.show()

        # Plot histogram and show the result
        df.hist(column=df.columns)
        plt.show()

data = [df_SP500_clean, df_DJIA_clean, df_NASDAQ_clean]
# Result
plot_boxplot_and_hist(data)


# Defining function for converting format type to datetime format for databases
# Define function to convert datetime column to number of days
def datetime_to_days(dataframe, column_name):
    # List of Dataframes with updated datetime column
    updated_df = []

    for df in dataframe:

        # Create a copy of dataframe
        df = df.copy()

        # Convert specified column in each Dataframe to datetime format
        df[column_name] = pd.to_datetime(df[column_name])

        # Ensure the column is of the correct datetime64 type before calculation
        df[column_name] = pd.to_datetime(df[column_name], errors='coerce')

        # Calculate the observation date as the number of days since 2019-12-19
        df[column_name] = (df[column_name] - pd.Timestamp("2019-12-19")).dt.days

        updated_df.append(df)

    # Return dataframes
    return updated_df

# List of dataframes
dataframes = [df_SP500_clean, df_DJIA_clean, df_NASDAQ_clean]

# Update Dataframes with applying the function
updated_dfs = datetime_to_days(dataframes, 'observation_date')
# Access each Dataframe # from the list
df_SP500_clean = updated_dfs[0]; df_DJIA_clean = updated_dfs[1]; df_NASDAQ_clean = updated_dfs[2]

# Remove the first row from the dataframe since it includes a zero value
# Reset the index of the Dataframe after the row is dropped
df_NASDAQ_clean = df_NASDAQ_clean.drop(index = 0).reset_index(drop=True)

# Define function to process and calculate the correlation for Dataframes
def process_and_calculate_correlation(updated_df):
    # List of correlation matrices
    correlation_matrices = []

    for df in updated_df:

        # Compute the correlation matrix
        correlation_matrix_r = df.corr()

        # Compute the new correlation matrix as the squared values
        correlation_matrix_new = correlation_matrix_r ** 2

        # Add new correlation matrices to the list
        correlation_matrices.append(correlation_matrix_new)

    return correlation_matrices

# List of Dataframes
dataframes = [df_SP500_clean, df_DJIA_clean, df_NASDAQ_clean]

# Calculating correlation matrices for each Dataframes
correlation_matrix = process_and_calculate_correlation(dataframes)
# Access the correlation matrices
correlation_matrix_SP500_new = correlation_matrix[0]; correlation_matrix_DJIA_new = correlation_matrix[1]
correlation_matrix_NASDAQCOM_new = correlation_matrix[2]

# Define function to compute and plot heatmap for the list of correlation matrix of the Dataframes
def plot_correlation_heatmaps(correlation_matrices, cmap='YlGnBu'):

    for i, corr_matrix in enumerate(correlation_matrices):

        plt.figure(figsize=(10, 8))

        # Plot the heatmap using Seaborn
        sns.heatmap(corr_matrix, annot=True, vmin=0, vmax=1, fmt='.4f', cmap=cmap,
                cbar=True, linecolor=(0.1, 0.2, 0.3))
        plt.title(f"Correlation Heatmap")
        plt.show()

# List of correlation matrices
correlation_matrices1 = [correlation_matrix_SP500_new, correlation_matrix_DJIA_new, correlation_matrix_NASDAQCOM_new]
# Plot the heatmaps
plot_correlation_heatmaps(correlation_matrices1)

# General function to remove outliers
def remove_outliers(data):
    columns = ['SP500', 'DJIA', 'NASDAQCOM']
    for df in data:
        for column_name in columns:
            q1 = df[column_name].quantile(0.25)
            q3 = df[column_name].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            df_clean = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
            return df_clean
data = [df_SP500_clean, df_DJIA_clean, df_NASDAQ_clean]
remove_outliers(data)

# Dictionary to store original dataframes
dataframes_removed = {
    "DJIA": df_DJIA_clean,
    "SP500": df_SP500_clean,
    "NASDAQ": df_NASDAQ_clean
}

# Cleaning missing values for each dataframe
cleaned_dataframes = {name: clean_missing_values(df) for name, df in dataframes_removed.items()}
# Access cleaned dataframes using keys
df_DJIA_clean_new = cleaned_dataframes["DJIA"]; df_SP500_clean_new = cleaned_dataframes["SP500"]
df_NASDAQ_clean_new = cleaned_dataframes["NASDAQ"]


# Plot regression plots for specified columns in Dataframes with Seaborn
def plot_regressions(dataframes_columns, date_col, plot_residuals=True):

    for df_name, (df, columns) in dataframes_columns.items():

        for column in columns:

            # Set theme and create the regression plot
            sns.set_theme(color_codes=True)
            plt.figure(figsize=(10, 6))
            # Plot regression of the Dataframes
            sns.regplot(x=date_col, y=column, data=df, color=(0.1, 0.2, 0.3))
            plt.title(f"Regression of {column}")
            plt.legend(labels=['regression'])
            plt.show()

            # Plot residuals
            if plot_residuals:
                plt.figure(figsize=(10, 6))
                sns.residplot(x=date_col, y=column, data=df, color=(0.1, 0.2, 0.3))
                plt.title(f"Residuals of {column}")
                plt.legend(labels=['residuals'])
                plt.grid()
                plt.show()

# Use the function to plot regressions
plot_regressions(
    dataframes_columns={
    'SP500': (df_SP500_clean_new, ['SP500']),
    'DJIA': (df_DJIA_clean_new, ['DJIA']),
    'NASDAQCOM': (df_NASDAQ_clean_new, ['NASDAQCOM'])
    },
    date_col='observation_date',
    plot_residuals=True)


# Plot regression plots for specified columns in Dataframes with Stats model
def calculate_regression_stats(dataframes_columns, date_col):

    for df_name, (df, columns) in dataframes_columns.items():

        for column in columns:
            slope, intercept, r_value, p_value, std_err = sps.linregress(df[date_col], df[column])
            print(f"Results for {column}:")
            print("slope:", slope)
            print("intercept:", intercept)
            print("r-value:", r_value)
            print("p-value:", p_value)
            print("std err:", std_err)
            print("\n")

calculate_regression_stats(dataframes_columns={
        'SP500': (df_SP500_clean_new, ['SP500']),
        'DJIA': (df_DJIA_clean_new, ['DJIA']),
        'NASDAQCOM': (df_NASDAQ_clean_new, ['NASDAQCOM'])
    }, date_col='observation_date')


def fit_predict_plot(dataframes_columns, date_column, test_data):
    """"Handling model predictions for multiple dataframes. """
    prediction_list = []

    for i, (df, dependent_vars) in dataframes_columns.items():

        for dependent_var in dependent_vars:

            # Fit the model
            X = df[date_column]
            Y = df[dependent_var]
            Xc = sm.add_constant(X)
            model = sm.OLS(Y, Xc).fit()

            # Prepare test data with constant
            Bc = sm.add_constant(test_data)
            predictions = model.predict(Bc)

            prediction_list.append({
                "dataframe_index": i,
                "dependent_variable": dependent_var,
                "model_params": model.params ,
                "predictions": list(zip(test_data, predictions))
            })
            print("\n")

            # Plot regression diagnostics
            fig, ax = plt.subplots(figsize=(10, 6))
            sm.graphics.plot_fit(model, 1, ax=ax)
            ax.set_X = date_column
            ax.set_Y = dependent_var
            plt.title(f"Linear Regression Plot")
            plt.legend()
            plt.grid(True)
            plt.show()

    print(prediction_list)
    # Return list
    return prediction_list

# Main Program
fit_predict_plot(dataframes_columns={
    'SP500': (df_SP500_clean_new, ['SP500']),
    'DJIA': (df_DJIA_clean_new, ['DJIA']),
    'NASDAQCOM': (df_NASDAQ_clean_new, ['NASDAQCOM'])
    }, date_column = 'observation_date', test_data = [2025, 2026, 2027, 2028, 2029, 2030])

