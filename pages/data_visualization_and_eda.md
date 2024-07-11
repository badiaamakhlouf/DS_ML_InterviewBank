# Data Visualisation and Exploratory Data Analysis

![Badiaa Makhlouf](https://img.shields.io/badge/Author-Badiaa%20Makhlouf-green)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?logo=linkedin)](https://www.linkedin.com/in/badiaa-m-b77032116/)
[![Follow](https://img.shields.io/github/followers/badiaamakhlouf?label=Follow&style=social)](https://github.com/badiaamakhlouf)
![License](https://img.shields.io/badge/License-MIT-red)
![Last Updated](https://img.shields.io/badge/last%20updated-July%202024-brightgreen)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![Maintained](https://img.shields.io/badge/maintained-yes-blue.svg) 

This page serves as a valuable resource for interview preparation and enhancing your understanding of data visualization and exploratory data analysis 📊 within the data science lifecycle.

It provides structured questions aimed at improving your proficiency in exploratory data analysis, emphasizing how to select the right visualizations to uncover insights and details within your data ✅.

Here are the social channels I am active on. I would love to connect with and follow others who share the same passion for data science, machine learning, and AI.
Let's exchange ideas, collaborate on projects, and grow together in this exciting field! 🥰 🥳

<div id="badges">
  <a href="https://github.com/badiaamakhlouf">
    <img src="https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white" alt="Github Badge"/>
  </a>
  <a href="https://www.linkedin.com/in/badiaa-m-b77032116/">
    <img src="https://img.shields.io/badge/LinkedIn-blue?logo=linkedin&logoColor=white" alt="LinkedIn Badge"/>
  </a>
  <a href="https://medium.com/@badiaa-makhlouf">
    <img src="https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white" alt="Medium Badge"/>
  </a>
  <a href="https://www.kaggle.com/badiaamakhlouf">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?logo=Kaggle&logoColor=white" alt="Kaggle Badge"/>
  </a>
  <a href="https://stackoverflow.com/users/10863083/baddy">
    <img src="https://img.shields.io/stackexchange/stackoverflow/r/10863083?color=orange&label=reputation&logo=stackoverflow" alt="Stackoverflow Badge"/>
  </a>
</div>

## List of Questions:

### Q1- Explain descriptive, predictive, and prescriptive analytics.


### Q2- What does Exploratory Data Analysis (EDA) mean? 

- It is a critical step in the data analysis process and is often the second step after cleaning the provided dataset.
- It aims to summarize the main characteristics of a dataset, gain insights into the underlying structure, identify patterns, detect anomalies, and formulate hypotheses for further analysis.
- Effective EDA helps to perform more targeted and informed analyses, leading to better decision-making and valuable insights from the data.
- **Key aspects of Exploratory Data Analysis include:**
     - Summary Statistics using `.describe()` pandas library.
     - Data Visualization
     - Distribution Analysis
     - Correlation Analysis
 
### Q3- How to get statistical description of our data using pandas ? 

- In the statistical description we try to select the next values for each numerical features:
    - Maximum values
    - Minimum
    - Average
    - Standard deviation
    - Median
    - Mean
- Code: `df.describe().transpose()`
- Example:

<img src="images/summary_statistic.png" width="600">


### Q4- What does Distribution Analysis mean?

- This analysis aims to examine the distribution of values within a dataset.
- Understanding the distribution of data is essential for gaining insights into its underlying characteristics, identifying patterns, and making informed decisions about subsequent analyses or modeling.
- Here are some examples of distribution analysis: 
    - Frequency Distribution:  it provides a summary of how often each value appears. 
    - Univariate and Bivariate Analysis : distplot, histplot and X versus Y etc.
    - Probability Distribution
    - Spread or Dispersion analysis
    - Skewness and Kurtosis analysis
    
- Understanding the data distribution is very important in many tasks, including identifying outliers, assessing the appropriateness of statistical models, and making decisions about data transformations.
- Different types of distributions may require different approaches in data analysis and modeling, and distribution analysis helps inform these decisions.

### Q5- What does Skewness mean?

- It is a measure of the asymmetry of a distribution.
- A distribution is asymmetrical when its left and right side are not mirror images.
- A skewed data can not be used to generate normal distribution. 
- It provides insights into the shape of a distribution.
- The three types of skewness are:
    - **Skewness > 0 :** right (or positive) skewness. This indicates that the tail on the right side is longer or fatter than the left side, and the majority of the data points are concentrated on the left side.
    - **Skewness < 0 :** left (or negative) skewness. It means the tail on the left side is longer or fatter than the right side, and the majority of the data points are concentrated on the right side.
    - **Skewness=0, Zero skewness :** the distribution is perfectly symmetrical.
    
<img src="images/Skewness.png" width="500">

_Source: https://en.wikipedia.org/wiki/Skewness_


- If we have a positively skewed distribution:  Mode < Median < Mean 
- If we have a left-skewed distribution: Mean < Median < Mode 

### Q6- What does Kurtosis mean ?  

- A statistical measure that describes the shape or "tailedness" of a distribution. 
- It provides information about the concentration of data points in the tails relative to the center of the distribution:
   - The data can be heavy-tailed means it has more extreme values than a normal distribution.
   - Or it can be light-tailed means it has fewer extreme values than a normal distribution.
- The three types of Kurtosis are:
    - **Kurtosis=0 (Mesokurtic) :** the distribution has the same tail behavior as a normal distribution.
    - **Kurtosis>0 (Leptokurtic):** the distribution has fatter tails (heavier tails) and a sharper peak than a normal distribution. This indicates a higher probability of extreme values.
    - **Kurtosis<0 (Platykurtic):** the distribution has thinner tails (lighter tails) and a flatter peak than a normal distribution. This suggests a lower probability of extreme values.

<img src="images/Kurtosis.png" width="400">

_Source: https://www.researchgate.net/figure/Examples-of-positive-negative-and-zero-excess-kurtosis_fig4_373105776_

### Q7- What does Spread or Dispersion analysis mean ?

- Spread analysis and dispersion analysis are usually used interchangeably, but they can have subtle differences depending on the context.
- Dispersion analysis is a more specific term used in statistics to describe the variability within a dataset.
- Spread analysis might be used more broadly depending on the context, particularly outside of pure statistical analysis.
- **Dispersion analysis:**
    - It explains how individual data points in a dataset deviate or spread out from a central measure of tendency, such as the mean or median. 
    - Dispersion measures provide insights into the variability or spread of the data and are crucial for understanding the overall distribution.
    -  Common measures of dispersion include:
       - Range: The difference between the maximum and minimum values.
       - Interquartile Range (IQR): The range of the middle 50% of the data, calculated as the difference between the 75th and 25th percentiles.
       - Variance: The average squared deviation from the mean, showing how data points are spread out.
       - Standard Deviation: The square root of the variance, providing a measure of spread in the same units as the data.
       - etc.

- **Spread analysis:** 
    - It might not only focus on statistical measures of dispersion but also consider how the spread affects decision-making, risk assessment, or comparison between different datasets.
    - It is used in various area such as finance, economics, and marketing to refer to differences between values (e.g., bid-ask spread, yield spread) beyond just statistical dispersion.
    - It can include the Dispersion analysis measures but may also consider broader implications and specific contexts.

### Q8- What does Correlation Analysis mean?

- Correlation analysis is a statistical method used to evaluate the strength and direction of the linear relationship between two quantitative variables.
- The result of a correlation analysis is a correlation coefficient, which quantifies the degree to which changes in one variable correspond to changes in another.
- The most common correlation coefficient is Pearson's r, which ranges from -1 to 1:
   - +1 indicates a perfect positive linear relationship.
   - -1 indicates a perfect negative linear relationship.
   - 0 indicates no linear relationship.
- Correlation analysis include three cases:
   - **Positive Correlation:** indicates that when one variable increases, the other increases.
   - **Negative Correlation:** indicates when one variable increases, the other decreases.
   - **No Correlation:** it indicates when there is no apparent relationship between the variables. 
- Correlation analysis is widely used in various fields, including economics, biology, psychology, and data science, to understand relationships between variables and make predictions based on observed patterns.
- Both correlation matrix heatmap and Scatter Plots are used to illustrate correlation analysis.

### Q9- What is the Scatter Plot ?

- It is a graphical representation of the relationship between two variables, X versus Y.
- Each point on the scatter plot represents an observation. Patterns in the scatter plot can indicate the type and strength of the correlation.
- It is typically used for visualizing and analyzing the relationship between two quantitative (numerical) variables.
- It is mainly used for trend identification, correlation analysis and outliers detection
- Scatter plots are a fundamental tool in exploratory data analysis (EDA) for uncovering relationships between variables and guiding further statistical analysis and modeling.
  
<img src="images/scatterplot.png" width="500">

_Source: https://www.data-to-viz.com/graph/scatter.html_

### Q10- What does correlation matrix mean? 

- It is a table that displays the correlation coefficients between many variables. 
- Each cell Corresponds to the correlation coefficient between two variables. 
- This matrix helps detect the presence of any positive or negative correlation between variables.
- The correlation is calculated using the pearson correlation coefficient so values varies from -1 to 1
- It is often visualized using heatmaps, where colors represent the magnitude and direction of correlations. This can make it easier to identify strong and weak correlations at a glance.

<img src="images/corr_matrix.png" width="500">

_Source: https://www.vertica.com/blog/in-database-machine-learning-2-calculate-a-correlation-matrix-a-data-exploration-post/_

    
### Q11- Why removing highly correlated features is very important?
- Removing highly correlated features is a preprocessing step and it is important for several reasons, especially in the context of statistical modeling and machine learning.
- It can be beneficial for both classification and regression tasks.
- Here are the key reasons why this is crucial:
   - Highly correlated features make it difficult to determine the individual effect of each feature on the target variable.
   - Besides, highly correlated features can introduce redundancy into the model, leading to overfitting and reduced generalization performance.
   - Removing them simplifies the model, makes it easier to interpret and improve its generalization to new data.
   - Reducing the number of features by removing highly correlated ones can lead to faster training times and reduced computational resource requirements via focusing on the most informative and relevant features.
  
### Q12- What does Normal distribution mean ?
- It is also known as the Gaussian distribution and it is a continuous probability distribution that is symmetric about its mean.
- It describes how the values of a variable are distributed.
- It is very useful in machine learning because it has deterministic statistical characteristics and it helps detect linear relationship between variables.
- Here are the key characteristics and properties of the normal distribution:
   - It has a Bell-shaped and symmetric curve about the mean.
   - The mean, median, and mode of the distribution are all equal.
   - The tails of the normal distribution approach, but never touch, the horizontal axis (x-axis)
   - The normal distribution has a single peak (one mode)
   - Empirical Rule:
      - 68% within 1σ of the mean.
      - 95% within 2σ of the mean.
      - 99.7% within 3σ of the mean
        

<img src="images/normal_dist.PNG" width="500">

_Source: https://medicoapps.org/biostatistics-normal-curve-test-of-significance-standard-error-2/_

### Q13- How to perform Frequency and Probability distribution analysis in EDA ?

- **Frequency Distribution in EDA:**
   - Frequency Distribution shows how often each value or range of values occurs in a dataset.
   - **For Numerical Data:**
      - Choose Bins and decide on intervals for grouping data.
      - Count Frequencies via calculating how many data points fall into each bin.
      - Visualize frequently distribution for numerical data using histograms.
   - **For Categorical Data:**
      - Count Frequencies via counting occurrences of each category.
      - Visualize using bar charts or pie charts.
        
- **Probability Distribution in EDA:**
   - Probability Distribution shows how probabilities are distributed over values of a random variable.
   - **For Numerical Data:**
      - Fit Distribution via identifying the theoretical distribution (e.g., normal, binomial).
      - Visualize via using density plots or probability density functions (PDFs).
   - **For Categorical Data:**
      - Calculate Probabilities via dividing frequencies by total count.
      - Visualize using bar charts.

### Q14- What else we can perform in EDA ? 
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



### Q15- What is the Difference between Univariate, Bivariate, and Multivariate analysis

- **Univariate Analysis:**
    - Analysis of a single variable.
    - To describe the distribution, central tendency, and variability of the variable.
    - Techniques:
        - **Descriptive Statistics:** Mean, median, mode, standard deviation, variance.
        - **Visualization:** Histograms, box plots, frequency distributions.
    - Example: Analyzing the house price in a certain region .

- **Bivariate Analysis:**
    - Analysis of two variables to understand the relationship between them.
    - To explore the association, correlation, or causation between the variables.
    - Techniques:
         - **Descriptive Statistics:** Correlation coefficient, cross-tabulation.
         - **Visualization:** Scatter plots, bar charts, line graphs.
         - **Statistical Tests:** t-tests, chi-square tests, ANOVA.
    - Example: Analyzing the relationship between height and weight of a group of people.
 
- **Multivariate Analysis:**
    - Analysis of more than two variables simultaneously.
    - To understand complex relationships, interactions, and the effect of multiple variables on outcomes.
    - Techniques:
        - **Descriptive Statistics:** Multivariate means, covariance matrices.
        - **Visualization:** Multidimensional scaling, parallel coordinates plots.
        - **Statistical Models:** Multiple regression, principal component analysis (PCA), factor analysis, MANOVA (multivariate analysis of variance).
    - Example: Analyzing the relationship between height, weight, age, and income of a group of people.
      
### Q16- How to perform univariate analysis for numerical and categorical variables?

- **Univariate Analysis for Numerical Variables:**
  
    - **Descriptive Statistics:**
      -  Determine mean, median, mode, standard deviation, variance, range, quartiles, percentiles
      -  Code: `df.describe().transpose()`
    - **Visualization:**
      - Histogram: Shows the distribution of the data. `histplot`
      - Box Plot: Visualizes the median, quartiles, and potential outliers.
      - Density Plot: Smooths out the histogram into a continuous curve. `distplot`
 
- **Univariate Analysis for Categorical Variables:**
    
   - **Descriptive Statistics:**
      - Frequency Distribution: count values of each category.
      - Mode: most frequently occurring category.
      - Proportions/Percentages: relative frequency of each category.
   - **Visualization:**
      - Bar Chart: displays the frequency or proportion of each category.
      - Pie Chart: shows the proportion of each category as slices of a pie.


### Q17- How to perform Bivariate analysis for Numerical-numerical, Categorical-Categorical, and Numerical-Categorical variables?

- **Numerical-Numerical Analysis:**
  - Scatter Plot to visualizes the relationship between two numerical variables and to identify patterns, trends, and potential correlations.
  - Correlation Coefficient to measure the strength and direction of the linear relationship between two variables. Pearson's correlation coefficient is commonly used.
  - Sometimes a Linear Regression is used to model the relationship between two variables and to provide insights into the nature and strength of the relationship.
    
- **Categorical-Categorical Analysis:** 
  - Contingency Table: displays the frequency distribution of two categorical variables and helps in understanding the joint distribution. `pd.crosstab(data['Category1'], data['Category2'])`
  - Chi-Square Test of Independence to tests whether there is a significant association between two categorical variables. `chi2, p, dof, expected = chi2_contingency(contingency_table)`
  - Stacked Bar Chart to visualizes the relationship by stacking bars for each category combination.

- **Numerical-Categorical Analysis:**
  - Box Plot to visualizes the distribution of a numerical variable across different categories and helps identify differences in central tendency and variability.
  - Violin Plot which is similar to a box plot but also shows the density of the data at different values.
  - ANOVA (Analysis of Variance) to test whether there are significant differences between the means of different categories.

### Q18- Mention the two kinds of target variables for predictive modeling.

- In predictive modeling, target variables can be classified into two main types: 
    - Continuous Target Variables:
        - They numerical variables that can take on any value within a range.
        - They are used in regression problems with regression models like Linear Regression, Random Forest, Decision Trees etc.
        - Applications: Predicting house prices, forecasting stock prices, etc.
          
    - Categorical Target Variables:
        - They are variables that represent categories or groups.
        - They are used in classification problems with classification models such as Logistic Regression, Decision Trees, Random Forests etc.
        - Applications: Spam detection (spam or not spam), image recognition (cat, dog, or other), medical diagnosis (disease present or not), etc.


### Q19- What information could you gain from a box-plot?

### Q20- What type of data is box-plots usually used for? Why?

### Q21- What is Violin Plot and what is used for ?

### Q22- When will you use a histogram and when will you use a bar chart? Explain with an example.





