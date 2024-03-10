# ML : Supervised Learning 
# Part II: Regression Analysis
This page presents the second part of the comprehensive guide to mastering regression analysis for your technical interview. It offers tailored questions and detailed answers to help you understand specific concepts thoroughly. If you missed the first part of the guide, please check [Regression Analysis I](./regression_analysis_I.md) for a complete understanding. 

## List of Questions 
### Q0-What is Lasso Regression and how it works?
- It is short for Least Absolute Shrinkage and Selection Operator.
- It is a linear regression technique used for: 
   - Feature selection
   - L1 regularization : used over regression methods for a more accurate prediction.
- It consists on adding a penalty term to the objective /cost function, which is ordinary least squares (OLS). 
- This penalty term penalizes the absolute values of the coefficients of the regression variables.
- Here is the formula of Lasso Regression objective function : $$RSS + λ \times (sum\ of\ the\ absolute\ values\ of\ the\ coefficients)$$
$$RSS_{Lasso}= {RSS + Lasso penality\ term  = \sum_{i=1}^{n}(y_i -\hat y_i)^2} + λ \sum_{j=1}^{p}|\hat W_j| $$
- Where :
 - RSS (Residual Sum of Squares) is the residual error between the predicted and actual values.
 - λ controls the strength of the penalty term.
- It works as follow : 
 - Reducing the impact of less important features on the model by shrinking their coefficients towards zero.
 - It encourages sparsity in the model which involves selecting only the most important features (p<n) while setting the coefficients of less important features to zero.
 
 
### Q1- How to choose a suitable or optimal λ value for Lasso Regression?
- λ is called Shrinkage coefficient or regularization parameter
- It controls the strength of the penalty term.
- Choosing the optimal value involves using cross-validation technique like K-fold cross validation
- Then, iteratively train and evaluate the model for various λ values using K-fold cross validation.
- The optimal regularization parameter is identified based on balancing model complexity with predictive accuracy. 
 
### Q2- Advantages of Lasso Regression
- The ability to perform feature selection automatically. 
- Very useful when dealing with high-dimensional datasets (large number of features)
- Prevent overfitting by penalizing large coefficients
- Gives models that are more generalizable to unseen data.
  
### Q3- What is Ridge Regression and how it works?
- It is a linear regression technique used for L2 regularization. 
- It adds penalty equivalent to the square of the magnitude of coefficients
- It penalizes the square of the coefficients of the regression variables by adding a penalty term to the ordinary least squares (OLS) objective function.
- Here is the formula of Ridge Regression objective function : $$RSS + λ * (sum\ of\ the\ squares\ of\ the\ coefficients)$$
$$RSS_{Ridge}= {RSS + Ridge\ penality\ term  = \sum_{i=1}^{n}(y_i -\hat y_i)^2 + λ \sum_{j=1}^{p} \hat W_j^2}$$
- Where : 
 - RSS (Residual Sum of Squares) is the residual error between the predicted and actual values.
 - $RSS = \sum_{i=1}^{n}(y_i -\hat y_i)^2$, where $y_i$ true value, $\hat y_i$ predicted value.
 - λ (lambda) is the regularization parameter or coefficient that controls the strength of the penalty term.
- It works by shrinking the coefficients of less important features towards zero. 
- However, unlike Lasso Regression, it does not set coefficients exactly to zero.
- Instead, it shrinks them towards zero while still keeping them in the model.

### Q4- Advantages of Ridge Regression :
- Reducing the model complexity.
- Improving model performance by reducing variance, especially in cases of multicollinearity.
- Stabilizes the model and reduces sensitivity to changes in training data.
- More robust predictions on unseen data.
- It helps prevent overfitting in predictive models.

**Notes:**
- Multicollinearity means high correlation between predictor /input variables .

### Q5- Ridge Regression Vs Lasso Regression
Ridge Regression penalizes coefficients based on their squares, while Lasso Regression penalizes coefficients based on their absolute values, potentially selecting variables by driving some coefficients to zero.

### Q6- How to choose the right Shrinkage coefficient for Lasso and Ridge?
- It is important to choose the right shrinkage coefficient (know as regularization parameter or penalty parameter).
- By leveraging the next approaches, you can systematically select the shrinkage coefficient that optimizes the trade-off between model complexity and performance: 
   - **Cross-Validation:** use cross-validation to assess model performance with various shrinkage coefficients.
   - **Grid Search:** perform a grid search to test different values of the coefficient and selecting the one that yields the best performance metrics. 
   - **Regularization Path:** plot the regularization path to visualize how feature coefficients change with varying shrinkage coefficients.
   - **Information Criteria:** use AIC or BIC to compare models with different shrinkage coefficients. They penalize model complexity and aid in selecting the best balance between fit and complexity.
   - **Domain Knowledge:** any prior knowledge or domain expertise that may inform the choice of the shrinkage coefficient. For example, in case of Lasso regression,  if you know that certain features are likely to be more important than others, you may want to use a higher shrinkage coefficient to encourage sparsity in the coefficients.
 
**Notes:**
- AIC : Akaike Information Criterion
- BIC : Bayesian Information Criterion

### Q7- What is Elastic Net Regression and how it works?
- It is a hybrid approach that combines both Ridge and Lasso Regression techniques.
- It involves adding both the L1 (Lasso) and L2 (Ridge) penalties to the loss function.
- It involves adjusting the mixing parameter to find the right balance between L1 and L2 penalties.
- It can select groups of correlated features while still offering regularization benefits to prevent overfitting.
- There are no limitations on the number of selected variables. 
- Formula : $$\min_{\beta} \left( \frac{1}{N} \sum_{i=1}^{N} (y_i - X_i \beta)^2 + \lambda_1 \sum_{j=1}^{p} |\beta_j| + \lambda_2 \sum_{j=1}^{p} \beta_j^2 \right)$$
- Where : 
    - β represents the coefficients.
    - $y_i$ represents the observed values.
    - $X_i$ represents the predictor variables.
    - N represents the number of observations.
    - p represents the number of predictors.
    - $λ_1$ regularization parameter controlling the strength of the L1 (Lasso).
    - $λ_2$ regularization parameter controlling the strength of the L2 (Ridge) penalties.

### Q8- Advantages Vs Disadvantages of Elastic Net Regression
- **Advantages:**
    - Highly effective when dealing with datasets that have many predictors.
    - Handles multicollinearity. Especially, when predictors are highly correlated.
    - Feature selection
    - Prevent overfitting
    - Balance between Lasso and Ridge: combining their strengths to overcome their individual limitations.
    - Greater flexibility in controlling model complexity
- **Disadvantages :**
    - Complexity in tuning parameters: selecting the optimal values for the mixing parameter and regularization strength can be challenging.
    - It can suffer with double shrinkage
    - Computational complexity: requires more computational resources, especially for large datasets.
     - Interpretability:  more challenging to interpret the coefficients of the model compared to simple regression model

### Q9- What does Bayesian Regression mean? 
- It is a type of linear regression that uses Bayesian statistics/inference to estimate the unknown parameters of a model.
- It involves modelling the distribution of possible coefficient values based on prior knowledge and observed data. 
- So, it does not estimate a single set of coefficients for the predictors
- It aims to find the best estimate of the parameters of a linear model that describes the relationship between the independent and dependent variables.
- It provides more accurate predictive measures from fewer data points.

### Q10- How Bayesian Regression works?
- Starting with **Prior Distribution** that respresents uncertainty about the coefficients before observing any data and it reflects any prior knowledge or beliefs about the coefficients. 
- The **likelihood** function represents the connection between the predictor variables and the response variable, expressing the probability of observing the data given the model parameters.
- The prior distribution is updated with the observed data and using Bayes’ theorem to obtain the **posterior distribution** of the coefficients and estimate the likelihood of a set of parameters given observed data. 
- Parameter Estimation, which involves deriving the most likely values for the coefficients along with their uncertainties from the posterior distribution.
- BR uses the posterior distribution of coefficients to make predictions. 
- Predictions are made by averaging over the predictions from all possible parameter values weighted by their posterior probabilities.

**Notes:**
- Traditional linear regression assumes Gaussian or normal distribution for data.
- Bayesian regression imposes stronger assumptions and prior distributions on parameters.
- Traditional linear regressions are easier to implement and faster with simpler models.
- Effective results for traditional models when data assumptions are valid.

### Q11- The advantages and disadvantages of Bayesian regression
- **Advantages:**
    - Effective for small datasets.
    - Well-suited for online learning due to its real-time data processing capabilities.
    - Robust and mathematically sound approach without requiring extensive prior knowledge.
    - Employs skewed distributions to incorporate external information into the model.

- **Disadvantages:**
    - Model inference can be time-consuming.
    - Less efficient than frequentist approaches for large datasets.
    - Dependency on external packages can pose challenges in certain environments.
    - Vulnerable to common modeling mistakes like linearity assumptions.

### Q12- When to use Bayesian regression ?
Bayesian regression is particularly useful in the following scenarios:
   - **Limited Data:** in case of small sample sizes, BR is a great choice for complex models
   - **Prior Knowledge:** if you have strong prior knowledge to incorporate into your model, BR offers a straightforward way to do so.
   - **Real-time Learning**

### Q13 - How determine whether a predictive model is underfitting
To determine if a predictive model is underfitting, consider these indicators:
- **Low Training and Test Accuracy:** If both the training and test accuracies are low, it suggests that the model is not capturing the underlying patterns in the data, indicating underfitting.
- **High Bias, Low Variance:** Underfit models typically have high bias and low variance, meaning they make simplistic assumptions about the data and are unable to capture its complexity.
- **Poor Performance on Training Data:** If the model performs poorly on the training data, failing to fit even the basic patterns, it is likely underfitting.
- **Simple Model with Few Parameters:** Models that are too simplistic or have too few parameters may underfit the data, as they cannot capture its underlying structure.
- **Visual Inspection of Learning Curves:** Learning curves showing the model's performance on training and test data as a function of training set size can indicate underfitting if both error rates converge to a high value.
- **Increasing Training Size Doesn't Improve Performance:** If increasing the size of the training set does not significantly improve model performance, it suggests underfitting, as the model is unable to learn from additional data.

### Q14 - How to determine whether a predictive model is overfitting ?
To determine if a predictive model is overfitting, consider these indicators:
- **High Training Accuracy, Low Test Accuracy:** If the model performs significantly better on the training data compared to unseen test data, it may be overfitting.
- **Large Gap Between Training and Test Error:** A substantial difference between the error rates of the training and test datasets suggests overfitting.
- **Visual Inspection of Learning Curves:** Plotting learning curves showing the model's performance on training and test data as a function of training set size can reveal overfitting if the test error remains high or decreases slowly as the training set size increases.
- **Regularization Parameter Tuning:** Tuning the regularization parameter can indicate if the model is overfitting or underfitting.
- **Complex Model with Many Parameters:** Overly complex models with a large number of parameters are prone to overfitting as they can capture noise in the training data.
- **Unstable Model Performance:** If the model's performance fluctuates significantly with small changes in the training data or model parameters, it might be overfitting.

### Q15 - Which factors do we consider while selecting Regression Model ?
To select the right regression model, consider these key factors:
- **Data Exploration:** Begin by exploring your data to understand variable relationships and impacts.
- **Model Evaluation Metrics:** Compare models using metrics like statistical significance, R-square, Adjusted R-square, AIC, BIC, and Mallow’s Cp criterion to assess goodness of fit and potential bias.
- **Cross-Validation:** Use cross-validation to evaluate predictive models, dividing data into training and validation sets to measure prediction accuracy.
- **Confounding Variables:** Exercise caution with automatic model selection methods if your dataset includes multiple confounding variables, as including them simultaneously may lead to issues.
- **Objective Alignment:** Consider your objective; sometimes a less powerful model may be more suitable for implementation than a highly statistically significant one.
- **Regularization Methods:** Employ regression regularization methods like Lasso, Ridge, and ElasticNet when dealing with high dimensionality and multicollinearity among variables.

### Q16- How can we use Neural Network to resolve regression problem?
- Neural networks can be used to solve both regression and classification problems.
- For regression probelms, it involves training a network to learn the mapping between input features and continuous output values.
- Here are the steps to use a neural network for regression:
1. **Data Preparation:** organize your dataset with input features and corresponding continuous output values.
2. **Model Architecture:** design the neural network architecture, including the number of input nodes (features), hidden layers, and output nodes. In case of regression, we use an input layer, one or more hidden layers, and an output layer.
3. **Initialization:** initialize the weights and biases of the neural network randomly or using predefined methods.
4. **Forward Propagation:** pass input data through the network to compute output predictions.
5. **Loss Calculation:** calculate the difference between predicted and actual output values using a loss function (e.g., mean squared error).
6. **Backpropagation:** propagate the error backward through the network to update the weights and biases using optimization algorithms like gradient descent.
7. **Iterative Training:** repeat steps 4-6 for multiple iterations (epochs) or until convergence, adjusting the model parameters to minimize the loss function.
8. **Prediction:** once the model is trained, use it to make predictions on new data by passing input features through the trained network.
9. **Evaluation:** Evaluate the performance of the model using metrics such as mean squared error, mean absolute error, or R-squared value on a separate validation or test dataset.
 
**Notes:**
- Sometimes, it is important to fine-tune the model architecture, hyperparameters, and training process to improve performance if needed.
- You can find more in-depth information about neural networks in the sections dedicated to deep learning and advanced machine learning. 

### Q17- What are the activation functions commonly used in the output layer of neural networks?
- They help in transforming the output of the neural network into a suitable format for the specific problem domain.
- The choice of the activation function depends on the tak: 
   - Binary Classification (Single Output Neuron): Sigmoid or Logistic function.
   - Multiclass Classification (Multiple Output Neurons): Softmax function.
   - Regression (Single Output Neuron): ReLU (Rectified Linear Unit) or no activation function (identity function).

**Notes:**
- Sigmoid: 
   - Formula : $σ(x) = {1 \over 1 + e^{-x}}$
   - Illustration :
     
<img src="images/sigmoid-function.png" width="350"/>
   
- Softmax :
   - Formula : $softmax(x_i) = {e^{x_i} \over \sum_{j=1}^{n}e^{x_j}}$
   - Illustration :
     
<img src="images/Softmax1.png" width="350"/>
      
- ReLU (Rectified Linear Unit):
   - Formula : f(x)=max(0,x), f  returns x if x is positive, and 0 otherwise.
   - Illustration :
     
<img src="images/Relu.png" width="350"/>

- ** Source (1): https://www.codecademy.com/resources/docs/ai/neural-networks/sigmoid-activation-function
- ** Source (2) : https://botpenguin.com/glossary/softmax-function
- ** Source (3) : https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/

### Q18- What does Support Vector Regression SVR mean and how it works?
- It is a type of support vector machines (SVM) used perform regression tasks.
- It is useful when the relationship between the input variables and the target variable is complex or nonlinear. 
- It works by finding the hyperplane that best fits the data while also maximizing the margin, which is the distance between the hyperplane and the nearest data points. 
- It is very useful when dealing with datasets that have a high dimensionality or contain outliers.
- SVR can use both linear and non-linear kernels : 
 - A linear kernel is a simple dot product between two input vectors.
 - A non-linear kernel is a more complex function that employs a more sophisticated function to capture more intricate patterns in the data.
- The choice of kernel depends on the data’s characteristics and the task’s complexity.

### Q19- What are advantages of Support Vector Regression (SVR) ?
- **Effective in High-Dimensional Spaces:** data with many features.
- **Robust to Overfitting:** less prone to overfitting, especially when using appropriate regularization parameters and kernel functions.
- **Handles Nonlinear Relationships:**  captures complex nonlinear relationships through the use of kernel tricks.
- **Works with Small Datasets:** works well with relatively small datasets especially when combined with appropriate kernel functions to capture complex patterns.
- **Flexibility with Kernel Functions:** different kernel functions could be used to find nonlinear relationships between variables.
- **Optimal Margin Maximization:** find the hyperplane that maximizes the margin while still fitting the data within a specified tolerance level, leading to robust generalization performance.

### Q20- What are the disadvantages of Support Vector Regression (SVR) ? 
- **Sensitivity to outliers:** outliers can significantly impact the position and orientation of the hyperplane and for sure the model performance
- **Computationally intensive:** computationally demanding, especially for large datasets +can become slow with increasing data size.
- **Difficulty in interpreting parameters:** complex models, interpreting the meaning of the parameters, such as support vectors, can be challenging compared to simpler linear regression models.
- **Sensitivity to kernel choice:** performance depends on the kernel function, and selecting the appropriate kernel can be challenging.

### Q21- What does K-Nearest Neighbors mean?
- Simple supervised ML algorithm
- Used for both :
   - Regression
   - Classification
- For classification, the prediction for a new data point is made based on the majority class
- For regression, the prediction for a new data point is made based on the average of the nearest K neighbors
- K :  the number of nearest neighbors considered for making predictions.
- It works as follow:
   - Calculating the distance between the new data point and all other data points in the training set, typically using Euclidean distance
   - Then selecting the K nearest neighbors
   - The prediction is then determined based on the majority class of the K neighbors
- It is a non-parametric and instance-based learning algorithm : no strong assumptions about the data distribution and instead relies on the local structure of the data to make predictions.

### Q22- How to select the best value for the number of neighbors (K)?
- It is important to find optimal value that balance between bias and variance.
- Here's a simple approach: 
   - **Cross-Validation:** split the data and for each value of K, train the KNN model on the training data and evaluate its performance on the validation data.
   - **Grid Search:** use a range of K values to test.
   - **Evaluate Performance:** evaluate each model using the appropriate evaluation metric such as accuracy (classification) or MSE (regression).
 - Choose Optimal K that gives the best performance first validation set then, test it on testing sets.

### Q23- The advantages Vs disadvantages of K-Nearest Neighbors (KNN)
- **Advantages:**
    - **Simple**
    - **No Training Phase:** doesn't need training; it uses stored data for predictions based on neighbor proximity.
    - **Non-Parametric:** does not make any assumptions about the underlying data distribution.
    - **Versatile:** used for both classification and regression tasks
    - **Interpretable:** predictions are easily interpreted, as they are based on the majority class or the average of neighboring points.
- **Disadvantages:**
    - High computational cost during prediction :as it needs to calculate distances to all training samples
    - Sensitivity to irrelevant features
    - Inefficiency with high-dimensional data
    
###  Q24- What does Principal Component Regression mean?
- It is a technique used in regression analysis.
- It combines principal component analysis (PCA) with linear regression.
- It reduces the dimensionality of predictor variables by transforming them into principal components.
- These components are linearly uncorrelated and capture the maximum variance in the data.
- PCR is used when dealing with multicollinearity among predictors or when the number of predictors exceeds the number of observations.
- The principal components are then used as predictors in a linear regression model to predict the response variable.


**Notes:**
- Multicollinearity refers to the situation where two or more predictor variables in a regression model are highly correlated with each other. 
- It can affect the accuracy and reliability of the regression model's predictions. 
  

