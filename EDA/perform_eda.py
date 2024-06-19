# library doc string
"""
Helper functions to run EDA.ipynb in Jupyter
Author: Nayeem Ahsan
Date: 6/14/2024
"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def plot_boxplot(data, features, figsize=(15, 10), nrows=3, ncols=3):
  """
  Plots a seaborn boxplot with customizations.

  Args:
      data: pandas DataFrame containing the data to be plotted.
      x: String, name of the column in the DataFrame for the x-axis.

  Returns:
      A matplotlib Axes object containing the plot (optional).
  """

    # Create a figure for subplots
  fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

  axes = [axes[i][j] for i in range(nrows) for j in range(ncols)]

  for i, col in zip(range(nrows * ncols), features):
        sns.boxplot(data, x=col, ax=axes[i])
        # Customize the plot
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
        plt.ylabel(f'Frequency')
        plt.xticks(rotation=45)  # Rotate x-axis labels for readability (optional)


def escape_special_chars(text):
    """Escape special characters for Matplotlib."""
    return text.replace('$', r'\$').replace('%', r'\%').replace('&', r'\&')


def plot_barplot(dataframe, column):
    '''
    Plot barplot for a column and return the plot object.

    Parameters:
        dataframe (DataFrame): Pandas DataFrame containing the data.
        column (str): Name of the column to plot.

    Returns:
        plt.figure: The plot object.
    '''
    plt.figure(figsize=(20, 10))
    dataframe[column].value_counts().plot(kind='bar')
    
    # Escape special characters in the column name
    escaped_column = escape_special_chars(column)
    
    # Use raw strings to avoid LaTeX interpretation
    plt.title(rf'Barplot of {escaped_column}', usetex=False)
    plt.xlabel(rf'{escaped_column}', usetex=False)
    plt.ylabel('Frequency', usetex=False)
    plt.legend()
    
    return plt


def plot_correlation(dataframe):
    '''
    Plots a correlation heatmap for numerical columns of a dataframe and saves as JPEG.

    Parameters:
        dataframe (DataFrame): Pandas DataFrame containing the data.
        folder_name (str): Directory to save the plot.

        Returns:
        plt.figure: The plot object.
    '''
    # plot the correlation heatmap
    numeric_dataframe = dataframe.select_dtypes(include=['number'])
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        numeric_dataframe.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.title('Correlation Heatmap')
    plt.xlabel('Numeric Columns')
    plt.ylabel('Numeric Columns')
    return plt

# function for display the percentage
def with_per(total, ax):
    '''
    function for display the percentage
    '''
    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height()/total)
        x = p.get_x() + p.get_width()
        y = p.get_height()
        ax.annotate(percentage, (x, y),ha='center')

def plot_snsbarplot(dataframe, column):
    '''
    Plot sns barplot for a column and return the plot object.

    Parameters:
        dataframe (DataFrame): Pandas DataFrame containing the data.
        column (str): Name of the column to plot.

    Returns:
        plt.figure: The plot object.
    '''
    sns.set(style = 'whitegrid')
    plt.figure(figsize=(20,5))
    total = len(dataframe)
    ax = sns.countplot(x = dataframe[column], data = dataframe)
    #plt.title(feature)
    with_per(total, ax)
    plt.show()
    
    return plt

def density_plot(dataframe, column):
    '''
    Plot density plot for a column and return the plot object.

    Parameters:
        dataframe (DataFrame): Pandas DataFrame containing the data.
        column (str): Name of the column to plot.

    Returns:
        plt.figure: The plot object.
    '''
    sns.histplot(dataframe[column])
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.show()

def bivariare_hist_plot(dataframe, bivariate_col, column):
    '''
    Plot density plot for a column and return the plot object.

    Parameters:
        dataframe (DataFrame): Pandas DataFrame containing the data.
        column (str): Name of the column to plot.

    Returns:
        plt.figure: The plot object.
    '''
    g = sns.FacetGrid(dataframe, hue=bivariate_col, height=7)
    g.map(sns.histplot, column, kde=True).add_legend()
    plt.title('Salary vs {}'.format(column))
    plt.show()

def bivariare_density_plot(dataframe, bivariate_col, column):
    '''
    Plot density plot for a column and return the plot object.

    Parameters:
        dataframe (DataFrame): Pandas DataFrame containing the data.
        column (str): Name of the column to plot.

    Returns:
        plt.figure: The plot object.
    '''
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=dataframe, x=column, hue=bivariate_col, fill=True)
    plt.title('Salary vs {}'.format(column))
    plt.show()

def outlier_detection(dtaframe, col1, column):
    '''
    function for detecting outliers
    '''
    if column != col1:
        sns.boxplot(x =col1, y = column, data = dtaframe)
        plt.title(column)
        plt.show()

#functions for removing outliers
def remove_outliers(df,labels):
    '''
    functions for removing outliers
    '''
    for label in labels:
        q1 = df[label].quantile(0.25)
        q3 = df[label].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        df[label] = df[label].mask(df[label]< lower_bound, df[label].median(),axis=0)
        df[label] = df[label].mask(df[label]> upper_bound, df[label].median(),axis=0)

    return df