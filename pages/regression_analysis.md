# ML : Regression Analysis
This page is a one-stop resource for mastering regression analysis, offering essential information and detailed insights tailored for technical interviews. Whether you're reinforcing foundational knowledge or delving into specific regression concepts, this page serves as a concise yet comprehensive guide. It's a valuable resource for anyone aspiring to excel in data science interviews, providing an edge in understanding regression analysis within the machine learning domain.
## List of questions:

### Q0- What does regression analysis mean?

- It is a statistical technique used in data science and machine learning fields. 
- It aims to model the relationship between a dependent variable and one or more independent variables.
- By modeling the relationship between inputs and output, it is easy to understand the nature and strength of the relationship and to make predictions based on that understanding.

- Mainly, we use regression analysis to resolve problems and answer questions such as:
    - How does a change in one variable (independent variable) impact another variable (dependent variable)?
    - Can we predict the value of the dependent variable based on the values of one or more independent variables?
- It is widely used in various fields, including economics, finance, biology, psychology, and machine learning.

### Q1- Examples of well-known machine learning algorithms used to solve regression problems
Here are some well-known machine learning algorithms commonly used to solve regression problems:

- Linear Regression
- Decision Trees
- Bayesian Regression
- Lasso Regression
- Ridge Regression
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest
- Gradient Boosting Algorithms (e.g., XGBoost, LightGBM)
- Neural Networks (Deep Learning)
  
### Q2- What is linear regression, and how does it work?
- It is the easiest and one of the most popular Machine Learning algorithms for predictive analysis
- LR, is a statistical method to model the relationship between a dependent variable (target) and one or more independent variables (inputs).
- Called "linear" because we assume the existence of linear relationship between previous variables.
- It aims to predict continuous/real or numeric variables such as temperature, salary, quantity, price etc. using the remaining features
- It can be classified into two main types : 
    - Simple Linear Regression : to model relationship between an independent variable (x) and a dependent variable (y).
    - Multiple Linear Regression : involves using more than one independent variable (X) to model the relationship with the dependent variable (y).
- It can be used for both continuous and categorical dependent variables (y) and can handle multiple independent variables.
## Q4- How Simple Linear Regression works? 

- It is used to model relationship between an independent variable (x) and a dependent variable (y).
- Example: Predicting the price of a house based on its size.

<img src="images/lin_reg.png" width="400">   

- The line of regression, is a line of best fit is plotted on a scatter plot of the data points as it is shown in the Figure below
- The equation of this line is : $$y=w \times x + b$$

    - Where : 
        - y: dependent/response/target variable, we want to predict it or explain it.
        - x: independent/input/predictor variable(s), it is (they are) used to predict or explain the variability of y
        - w: regression coefficients: the parameters in the regression equation that indicate the strength and direction of the relationship between variables.
        - b:bias term which represents patterns that do not pass through the origin
        
- The line is determined by finding the values of the slope (w) and intercept (b) that minimize the sum of residuals.
- Residuals: 
    - Corresponds to the prediction error which is differences between the observed (y) and predicted values ($\hat y$), .
    - Formula : $e=y-\hat y$
    - We calculate the Sum
- Our main goal is to find the best fit line where the error between predicted values and actual values should be minimized.

*Source:https://www.javatpoint.com/linear-regression-in-machine-learning

### Q5- How Multiple Linear Regression works? 
- The unique difference between Simple and Multiple Linear Regression lies in the number of independent variables used in the regression model.
- We have multiple independent variables $x_1, x_2, ..., x_n$
- New equation: $y=b_0+b_1 x_1 + b_2x_2+ ...+b_n x_n$
- Where $b_0$ represents the intercept, and $b_1, b_2, ..., b_n$ represent the coefficients of the independent variables.
- Simple linear regression involves one independent variable, while multiple linear regression involves two or more independent variables.
- Example: Predicting the performance of a student based on their age, gender, IQ, etc.
