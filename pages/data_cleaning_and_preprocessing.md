# Data Cleaning and Pre-processing 
This page serves as a comprehensive resource for data science interview questions, focusing specifically on topics related to data cleaning, preprocessing, and analysis. Here, you'll find a curated collection of questions that span various aspects of data preparation and exploratory data analysis. 

Whether you're preparing for an interview or seeking to deepen your understanding of essential data science concepts, this notebook aims to provide a structured and informative guide to help you navigate through key challenges in the field.
Explore the questions, test your knowledge, and enhance your proficiency in the fundamental stages of data science workflows.
## List of Questions:
### 1-What are the main tasks of data cleaning in Data Science?
Here are the main tasks to perform in the cleaning phase :
- Finding and handling missing data
- Finding and handling duplicates
- Finding and handling outliers
  
Note : encoding categorical data can be done in feature engineering phase. 
### 2- How to deal with missing values ?

Handling missing values is a crucial step in data preprocessing to ensure accurate and unbiased analysis. Here are two main methods to deal with missing values: 
- Remove missing values 
- Impute missing values 
- Forward or Backward Fill


**Note:** to make better predictions we can add an extension to imputation.

Choosing the right method is based on:
- The characteristics of the data
- The percentage of missing values
- The goals of the performed analysis.

No single method is suitable for all situations, so it's essential to understand the context and implications of each approach.

### 3- How to detect or identify missing values? 
- Identifying missing values is the first step to perform when dealing with them. 
- Using Pandas functions like `isnull()` or `info()`.

### 4- How to remove missing values? 

Here is how to remove missing values :
- Remove Rows with nan/null values using `df = df.dropna()`
- Remove Columns with nan/null values using `df = df.dropna(axis=1)`

Dropping rows or columns is not too advantageous because most values are going to be lost and they contain important information

### 5- How to impute missing values? 

We have four main methods:

    - Impute with statistical measures
    - Impute with a Placeholder 
    - Impute with Machine Learning Algorithms
    - Impute using Interpolation
    - Multiple Imputation
Imputed value won't be exactly right in most cases but it usually leads to more accurate models than you would get from dropping the column entirely  
    
#### 5. 1- What does impute with a statistical measures mean ? 
- Fill missing values with statistical measures (mean, median, mode) or using more advanced imputation methods.
- Example: `df['column'] = df['column'].fillna(df['column'].mean())`
    
#### 5. 2- What does impute with a Placeholder mean ? 
- Replace with a specific value that does not occur naturally in the dataset. 
- Example: `df = df.fillna(-1)`

#### 5. 3- What does impute with a Machine Learning Algorithm mean ?    
- **Solution 1:**  use `KNNImputer()` class from the scikit-learn Python library.
- **Solution 2:**
    - Train a machine learning model to predict missing values based on other features in the dataset.
    - Example : Random Forest 
#### 5. 4- What does Impute using Interpolation mean ?
- Interpolation is a technique used to estimate missing values based on the observed values in a dataset.
- It works by filling in the gaps between known data points, assuming some underlying pattern or relationship.
- Here are some interpolation techniques:
    - Linear Interpolation 
    - Polynomial Interpolation
    - Quadratic 
    - Etc.

Note : the choice of the right interpolation method depends on:
- The nature of the data.
- The assumptions about its behavior
#### 5. 5- What does multiple imputation mean ? 
- It is a statistical technique used to handle missing data via creating multiple imputed datasets. 
- Multiple datasets are created by imputing missing values using a chosen imputation method. 
- Examples : mean imputation, regression imputation, k-Nearest Neighbors imputation, or more sophisticated methods.
- Each dataset represents a set of values for the missing entries.
- Instead of imputing a single value for each missing observation, multiple imputation illustrates the uncertainty associated with missing data by generating several imputed datasets. 
- The results from the analyses conducted on the imputed datasets are combined, or "pooled," to obtain an overall estimate of the parameter of interest.
- The combined results provide not only a point estimate but also an estimate of the uncertainty associated with the missing data. This incorporates both the imputation variability and the variability due to analyzing different imputed datasets.
- `fancyimpute()` Python library can be employed to implement multiple imputation efficiently.

### 6- Why do we need an extension to imputation? 

- Sometimes, missing values themselves can be indicative. Create a new binary column indicating whether a value is missing. 
- For each column with missing entries in the original dataset, we add a new column that shows the location of imputed entries. 
- Models would make better predictions by considering which values were originally missing.   
- Example:  `df['column_missing'] = df['column'].isnull().astype(int)` 

### 7- Why it is better to use the median value for imputation in the case of outliers?
- Using the median for imputation in case of outliers is often considered a better solution compared to the mean.
- The median is a measure of central tendency that has: 
    - **Robustness to Outliers:** it is less influenced by extreme values because it is not affected by the actual values of data points but rather their order. Outliers have a minimal impact on the median.
    - **Resilient to Skewness:** in a skewed distribution, where the tail is longer in one direction, the mean can be heavily influenced by the skewness. The median, being the middle value, is less affected by the skewness and provides a more representative measure in such situations.
    - **Ability to avoid Biased Estimates:** in the presence of outliers, using the mean for imputation might lead to biased estimates, especially when the distribution is not symmetric. The median provides a more balanced estimate in skewed or asymmetric distributions.
    - **Ability to maintain Robustness in Non-Normal Distributions:** in case our data does not have a normal distribution, the median is often a more reliable measure of central tendencyas it helps in producing more accurate imputations.
    
### 8-  How to perform Forward or Backward Fill   ? 
Propagate the last valid observation forward or use the next valid observation to fill missing values: 

- Forward fill using : `df = df.ffill()`  or `df.fillna(method='ffill')`
- Backward fill using : `df = df.bfill()` or `df.fillna(method='bfill')`
### 3-
### 4-
### 5-
### 6-
### 7-
### 8-
### 9-
### 10-
### 11-
### 12-
### 13-
### 14-
### 15-
### 16-
