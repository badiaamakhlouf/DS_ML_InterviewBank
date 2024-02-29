# Data Cleaning and Pre-processing 
This page serves as a comprehensive resource for data science interview questions, focusing specifically on topics related to data cleaning, preprocessing, and analysis. Here, you'll find a curated collection of questions that span various aspects of data preparation and exploratory data analysis. 

Whether you're preparing for an interview or seeking to deepen your understanding of essential data science concepts, this notebook aims to provide a structured and informative guide to help you navigate through key challenges in the field.
Explore the questions, test your knowledge, and enhance your proficiency in the fundamental stages of data science workflows.
## List of Questions:
### Q1-What are the main tasks of data cleaning in Data Science?
Here are the main tasks to perform in the cleaning phase :
- Finding and handling missing data
- Finding and handling duplicates
- Finding and handling outliers
  
Note : encoding categorical data can be done in feature engineering phase. 
### Q2- How to deal with missing values ?

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

### Q3- How to detect or identify missing values? 
- Identifying missing values is the first step to perform when dealing with them. 
- Using Pandas functions like `isnull()` or `info()`.

### Q4- How to remove missing values? 

Here is how to remove missing values :
- Remove Rows with nan/null values using `df = df.dropna()`
- Remove Columns with nan/null values using `df = df.dropna(axis=1)`

Dropping rows or columns is not too advantageous because most values are going to be lost and they contain important information

### Q5- How to impute missing values? 

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

### Q6- Why do we need an extension to imputation? 

- Sometimes, missing values themselves can be indicative. Create a new binary column indicating whether a value is missing. 
- For each column with missing entries in the original dataset, we add a new column that shows the location of imputed entries. 
- Models would make better predictions by considering which values were originally missing.   
- Example:  `df['column_missing'] = df['column'].isnull().astype(int)` 

### Q7- Why it is better to use the median value for imputation in the case of outliers?
- Using the median for imputation in case of outliers is often considered a better solution compared to the mean.
- The median is a measure of central tendency that has: 
    - **Robustness to Outliers:** it is less influenced by extreme values because it is not affected by the actual values of data points but rather their order. Outliers have a minimal impact on the median.
    - **Resilient to Skewness:** in a skewed distribution, where the tail is longer in one direction, the mean can be heavily influenced by the skewness. The median, being the middle value, is less affected by the skewness and provides a more representative measure in such situations.
    - **Ability to avoid Biased Estimates:** in the presence of outliers, using the mean for imputation might lead to biased estimates, especially when the distribution is not symmetric. The median provides a more balanced estimate in skewed or asymmetric distributions.
    - **Ability to maintain Robustness in Non-Normal Distributions:** in case our data does not have a normal distribution, the median is often a more reliable measure of central tendency as it helps in producing more accurate imputations.
    
### Q8-  How to perform Forward or Backward Fill   ? 
Propagate the last valid observation forward or use the next valid observation to fill missing values: 

- Forward fill using : `df = df.ffill()`  or `df.fillna(method='ffill')`
- Backward fill using : `df = df.bfill()` or `df.fillna(method='bfill')`
  
### Q9- How to handle duplicates ? 
Handling duplicates in data science is an essential step to ensure data quality and avoid biases or inaccuracies in analysis. Here are common methods to handle duplicates:
- 1- Identifying Duplicates using `duplicated()` using Pandas
- 2- Removing Duplicates - all : `df = df.drop_duplicates()`
- 3- Removing Duplicates - Keep first Occurrences : `df = df.drop_duplicates(keep='first')`
- 4- Removing Duplicates - Keep last Occurrences : `df = df.drop_duplicates(keep='last')`
- 5- Handling Duplicates Based on Columns
  
### Q10- How to find outliers?
To find outliers, only numerical columns are considered in our analysis. Here are the common methods to do that :
- Visualization technique :  Box Plot, Scatter Plot and Histogram Plot (the most used ones).
- Mathematical approach :
    - Z-score
    - Interquartile range : IQR score 
- Machine Learning Models :
    - Clustering Algorithms : Kmeans, DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    - One-Class SVM (Support Vector Machine)
    - Autoencoders
- Histogram-based Methods:
    - Isolation Forest
- Domain-Specific Knowledge

It's better to try various outlier detection methods and evaluate their performance based on your specific data characteristics that are:

- Data distribution
- Data dimensionality
- The type of outliers you expect to encounter. 

It's often a good practice to combine multiple methods for a more robust outlier detection approach.

### Q11- What Visualization techniques can be used to determine outliers?

- Performing some plots and analysis:   
    - Box plot is considered as Uni-variate analysis 
    - Scatter plot is considered as Bi-variate analysis
- Box plot represents five statistical measures :
    - Q1 : Lower quartile 
    - Q2 : Median
    - Q3 : Upper quartile 
    - Min
    - Max
  - Here is an example of Box plot :
    
    ![title](images/boxplot.png)

  - Here an example of Scatter plot :
 
    ![title](images/scatter-plot.png) 
      
### Q12- How to handle outliers in dataset ? 
Here are some methods about how we handle outliers :

- **Deleting the values:** removing the value completely, if we are sure that this value is wrong and it will never occur again, we remove it using either Interquartile range or Z-score.
- **Replace the values:** change the values if we know the reason for the outliers. (Example: using 99th percentile)
- **Data transformation:** some times data transformation such as natural log reduces the variation caused by the extreme values. Most used for highly skewed data sets.

### Q13- What does Z-Score mean?
- It calculates the Z-score for each data point.
- Z-score measures how many standard deviations a data point is from the mean.
- Typically, a threshold of 2 to 3 standard deviations is used to identify outliers.
- Formula: $Z ={ X - \mu \over\sigma}$

### Q14- What does IQR : interquartile range mean? 
- The IQR is the difference between the third quartile (Q3) and the first quartile (Q1): IQR = Q3 - Q1
- Q1: It represents the median of the lower 50% of the data. Represents 0.25 percentile
- Q3 : It represents the median of the upper 50% of the data. Represents 0.75 percentile

To calculate percentiles or quantiles, we need to sort the data in ascending order and finding the value below which a certain percentage of the data falls.
![title](images/boxplot.png) 

### Q15- What are the limitations of IQR?
Here are the list of limitations : 

- IQR is sensitive to the size of the dataset : may not accurately represent the spread of the data in case of smaller dataset
- It assumes that the data is symmetrically distributed. In case the distribution is skewed, IQR may not accurately represent the spread of the data.
- For IQR, all data points outside the defined range are identified as outliers. However, in some datasets we expect a certain degree of variability, and not all deviations should be considered outliers.
- It does not provide information about Outlier Magnitude. It consider all values outside the defined range equally, without providing a measure of how extreme they are.
- It doesn't consider the shape of the overall data distribution. It may not perform well in detecting outliers in non-Gaussian distributions or distributions with multiple modes.
- It is possible to loose information. With deleting values that are outside the IQR range you sacrifice detailed knowledge about them. Depending on the analytical goals, this loss of detailed information may or may not be significant.
- IQR is considered robust to outliers within its calculated range. This means that if there are extreme values within this range, they have less impact on the calculation of IQR.
-  IQR provides robustness within its calculated range, it is not robust to the influence of extreme values outside that range, and such extreme values may still affect the identification of potential outliers. 

### Q16- How to mitigate these limitations ?
Here are some solutions:
- In scenarios where the nature and cause of outliers matter, the IQR alone might not be sufficient. Other methods that retain specific value information, such as boxplots or more advanced outlier detection techniques, might be more suitable for a detailed diagnostic analysis.
- In situations where extreme values might exist, and their impact needs to be minimized, other outlier detection methods that are more robust to extreme values, such as modified Z-scores or robust regression techniques, might be considered.

<img src="images/distribution_modes.png" width="600">

### Q17- How ML Algorithms used for outliers detection ?
We have two main methods: 
- **Clustering Algorithms:** for example k-means can be used to detect outliers where points that do not belong to any cluster or are in small clusters can be identified as outliers.
- **Isolation Forest:** designed specifically for outlier detection. It isolates outliers by recursively partitioning the data.
  
### Q18- What does Exploratory Data Analysis (EDA) mean? 
It is a critical step in the data analysis process and is often the second step after cleaning the provided dataset. The primary goal of EDA is to summarize the main characteristics of a dataset, gain insights into the underlying structure, identify patterns, detect anomalies, and formulate hypotheses for further analysis.

**Key aspects of Exploratory Data Analysis include:**
- Summary Statistics using `.describe()` pandas library.
- Data Visualization
- Distribution Analysis
- Correlation Analysis
Effective EDA aims to perform more targeted and informed analyses, leading to better decision-making and valuable insights from the data.

### Q19- What does Distribution Analysis mean?
- This analysis aims to examine the distribution of values within a dataset.
- Understanding the distribution of data is essential for gaining insights into its underlying characteristics, identifying patterns, and making informed decisions about subsequent analyses or modeling.
- Here are some examples of distribution analysis: 
    - Frequency Distribution:  It provides a summary of how often each value appears. We can use `.value_counts()` Pandas library.
    - Univariate and Bivariate Analysis : distplot, histplot and X versus Y etc.
    - Probability Distribution
    - Spread or Dispersion analysis
    - Skewness and Kurtosis analysis
    
- Understanding the data distribution is very important in many tasks, including identifying outliers, assessing the appropriateness of statistical models, and making decisions about data transformations.
- Different types of distributions may require different approaches in data analysis and modeling, and distribution analysis helps inform these decisions.

#### Q19.1- What does Skewness and Kurtosis mean ?
**Skewness:**
- It is a measure of the asymmetry of a distribution.
- A distribution is asymmetrical when its left and right side are not mirror images.
- A skewed data can not be used to generate normal distribution. 
- It provides insights into the shape of a distribution.
- The three types of skewness are:
    - **Skewness > 0 :** right (or positive) skewness. This indicates that the tail on the right side is longer or fatter than the left side, and the majority of the data points are concentrated on the left side.
    - **Skewness < 0 :** left (or negative) skewness. It means the tail on the left side is longer or fatter than the right side, and the majority of the data points are concentrated on the right side.
    - **Skewness=0, Zero skewness :** the distribution is perfectly symmetrical.
    
<img src="images/Skewness.png" width="400">
    
**Kurtosis:**
- A statistical measure that describes the shape or "tailedness" of a distribution. 
- It provides information about the concentration of data points in the tails relative to the center of the distribution. 
- The three types of Kurtosis are:
    - **Kurtosis=0 (Mesokurtic) :** the distribution has the same tail behavior as a normal distribution.
    - **Kurtosis>0 (Leptokurtic):** the distribution has fatter tails (heavier tails) and a sharper peak than a normal distribution. This indicates a higher probability of extreme values.
    - **Kurtosis<0 (Platykurtic):** the distribution has thinner tails (lighter tails) and a flatter peak than a normal distribution. This suggests a lower probability of extreme values.
   
kurtosis measures whether the data is heavy-tailed (more extreme values than a normal distribution) or light-tailed (fewer extreme values than a normal distribution).


<img src="images/Kurtosis.png" width="400">

#### Q19.2- What does Spread or Dispersion mean ?
- Data spread: 
    - It provides information about the range of values in a dataset.
    - It provides information about how dispersed or scattered the individual data points are around a measure of central tendency, such as the mean or median.
    - Spread measures help to understand the variability or dispersion of the data.
    - **Examples: IQR, range, variance, standard deviation** 
    - It is crucial to understand the spread of data for better outliers detection, risk assessment, decision-Making etc.
- Dispersion:
    - It explains how individual data points in a dataset deviate or spread out from a central measure of tendency, such as the mean or median. 
    - Dispersion measures provide insights into the variability or spread of the data and are crucial for understanding the overall distribution.
    - **Examples: IQR, range, variance, standard deviation, Mean Absolute Deviation (MAD), Coefficient of Variation (CV)**

### Q20- How to get statistical description of our data using pandas ? 
- In the statistical description we try to select the next values for each numerical features:
    - Maximum values
    - Minimum
    - Average
    - Standard deviation
    - Median
    - Mean
- Code: `df.describe().transpose()`

### Q21- What does Correlation Analysis mean?
- Correlation analysis is a statistical method used to evaluate the strength and direction of the linear relationship between two quantitative variables.
- The result of a correlation analysis is a correlation coefficient, which quantifies the degree to which changes in one variable correspond to changes in another.
- Correlation analysis is widely used in various fields, including economics, biology, psychology, and data science, to understand relationships between variables and make predictions based on observed patterns.
#### Q21.1- What are the plots used to illustrate correlation?
- Correlation matrix and heatmap 
- Scatter Plot : it provides a visual representation of the relationship between two variables. X versus Y
#### Q21.2- What does correlation matrix mean? 
- It is a table that displays the correlation coefficients between many variables. 
- Each cell Corresponds to the correlation coefficient between two variables. 
- This matrix helps detect the presence of any positive or negative correlation between variables.
- The correlation is calculated using the pearson correlation coefficient so values varies from -1 to 1

<img src="images/corr_matrix.png" width="400">
_source: https://www.vertica.com/blog/in-database-machine-learning-2-calculate-a-correlation-matrix-a-data-exploration-post/_

### Q22- What else we can perform in EDA ? 
Here are more analysis to perform during EDA phase:
- Data frame dimension `df.shape`
- Data frame columns: `df.columns`
- Count values: `df['SaleCondition'].value_counts().to_frame()`
- Data sampling: sometimes, it is required to perform over/undersampling in case we have Imbalanced datasets
- Data Grouping using groupby : df_group=df[['YearRemodAdd','SalePrice']].groupby(by=['YearRemodAdd']).max()
- Data filtering :
    - `df_filter =df[df.column>200000]` 
    - `df_filter =df[(df.column1>150000) & (df.column2==2008)]`
    - `df_filter =df[(df.column1>2011) | (df.column2==2008)]`
- Data analysis: 
    - Univariate Analysis : `distplot` and `histplot`
    - Bivariate Analysis `pairplot`, `FacetGrid`, `jointplot` etc.
    - Multivariate Analysis: correlation matrix or heatmap

Notes:
- Multivariate analysis involves analyzing the relationship between three or more variables. We can use scatter matrix plots to visualize the relationship between each pair of features, along with the distribution of each feature.
- Bivariate analysis involves analyzing the relationship between two variables. We can use scatter plots to visualize the relationship between each pair of feature.

### Q23- What is the difference between covariance and correlation?
- These two measures are used during the exploratory data analysis to gain insights from the data.
- **Covariance:** 
   - It measures the degree to which two variables change together.
   - It indicates the direction of the linear relationship between the variables.
   - It can take on any value

- **Correlation:**
   - It is a standardized measure of the strength and direction of the linear relationship between two variables.
   - It ranges from -1 to 1, where :
        - 1: perfect positive linear relationship
        - -1 indicates a perfect negative linear relationship
        - 0 indicates no linear relationship.
