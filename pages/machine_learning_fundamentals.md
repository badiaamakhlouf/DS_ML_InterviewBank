# Machine Learning Fundamentals 

This GitHub page is a guide for those gearing up for machine learning technical interviews. It covers the basics and essentials, focusing on questions related to machine learning fundamentals and model evaluation. The content includes detailed questions about three main machine learning approaches: regression, classification, and clustering.

Whether you're polishing your interview skills or seeking insightful questions as an interviewer, this page is a valuable resource to strengthen your grasp of machine learning basics.

**Note:** Your suggestions for improvement are welcome; please feel free to provide feedback to help maintain the precision and coherence of this page.
## List Of Questions:
### PART I: ML Basics  
#### Q1- What does Machine Learning mean ? 
Machine Learning (ML) is a subset of artificial intelligence (AI) that focuses on the development of algorithms and statistical models that enable computers to perform tasks without explicit programming. The core idea behind machine learning is to allow machines to learn patterns, make predictions, or optimize decisions based on data.

Key concepts in machine learning include:
- Types of Machine Learning: supervised, unsupervised and semi-supervised
- Types of machine learning problems: classification, regression and clustering
- Split data into Training, validation and testing sets (case of supervised)
- Choose the right algorithm depends on the problem you want to solve
- Model Evaluation
- Hyperparameter Tuning
- Deployment
  
#### Q2- What are the types of ML algorithms ? 
Machine learning algorithms can be categorized into several types based on their learning styles and the nature of the task they are designed to solve.

Here are some common types of machine learning algorithms:
- **Supervised Learning** 
- **Unsupervised Learning**
- **Semi-Supervised Learning**
- **Deep Learning** 
- **Reinforcement Learning** 
- **Ensemble learning**  
- **Ranking**
- **Recommendation system**

#### Q3- What does supervised, unsupervised and semi-supervised mean in ML? 

In machine learning, the terms "supervised learning," "unsupervised learning," and "semi-supervised learning" refer to different approaches based on the type of training data available and the learning task at hand:

- **Supervised Learning :** training a model on a labeled dataset, where the algorithm learns the relationship between input features and corresponding target labels. Can be used for Regression (continous output) or Classification (discrete output). 
- **Unsupervised Learning :** Deals with unlabeled data and aims to find patterns, structures, or relationships within the data. Can be used for Clustering (Groups similar data points together) or association
- **Semi-Supervised Learning:** Utilizes a combination of labeled and unlabeled data to improve learning performance, often in situations where obtaining labeled data is challenging and expensive.

#### Q4- What are Unsupervised Learning techniques ?
 We have two techniques, Clustering and association: 
 - Custering :  involves grouping similar data points together based on inherent patterns or similarities. Example: grouping customers with similar purchasing behavior for targeted marketing.. 
 - Association : identifying patterns of associations between different variables or items. Example: e-commerse website suggest other items for you to buy based on prior purchases.
 
#### Q5- What are Supervised Learning techniques ? 
We have two techniques: classfication and regression: 
- Regression : involves predicting a continuous output or numerical value based on input features. Examples : predicting house prices, temperature, stock prices etc.
- Classification : is the task of assigning predefined labels or categories to input data. We have two types of classification algorithms: 
    - Binary classification (two classes). Example: identifying whether an email is spam or not.
    - Multiclass classification (multiple classes). Example: classifying images of animals into different species.

#### Q6- Examples of well-known machine learning algorithms used to solve Regression problems
Here are some well-known machine learning algorithms commonly used to solve regression problems:

- Linear Regression
- Decision Trees
- Random Forest
- Gradient Boosting Algorithms (e.g., XGBoost, LightGBM)
- K-Nearest Neighbors (KNN)
- Bayesian Regression
- Lasso Regression
- Ridge Regression
- Neural Networks (Deep Learning)

More details regarding each algorithm could be found in [Regression Analysis](pages/regression_analysis.md)

#### Q7- Examples of well-known machine learning algorithms used to solve Classification problems
Here are some well-known machine learning algorithms commonly used to solve classification problems:

- Logistic Regression
- Decision Trees
- Random Forest
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- AdaBoost
- Gradient Boosting Machines (GBM)
- XGBoost
- CatBoost
- LightGBM
- Neural Networks (Deep Learning)

More details regarding each algorithm could be found in [Classification Analysis](pages/classification_analysis.md)

#### Q8- Examples of well-known machine learning algorithms used to solve Clustering problems
Several well-known machine learning algorithms are commonly used for solving clustering problems. Here are some examples:

- K-Means Clustering 
- Hierarchical Clustering
- Agglomerative Clustering
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- Mean Shift
- Gaussian Mixture Model (GMM)

These algorithms address different types of clustering scenarios and have varying strengths depending on the nature of the data and the desired outcomes. The choice of clustering algorithm often depends on factors such as the shape of clusters, noise in the data, and the number of clusters expected.


#### Q9- What is Ensemble learning?

Ensemble learning is a machine learning technique that involves combining the predictions of multiple individual models to improve overall performance and accuracy. Instead of relying on a single model, ensemble methods leverage the strengths of diverse models to compensate for each other's weaknesses. The idea is that by aggregating the predictions of multiple models, the ensemble can achieve better generalization and make more robust predictions than any individual model.

There are several ensemble learning methods, with two primary types being:
- **Bagging (Bootstrap Aggregating) :** 
    - Involves training multiple instances of the same model on different subsets of the training data, typically sampled with replacement. 
    - Examples : Random Forest, Bagged Decision Trees, Bagged SVM (Support Vector Machines), Bagged K-Nearest Neighbors, Bagged Neural Networks
- **Boosting :**
    - Focuses on sequentially training models, with each subsequent model giving more attention to the instances that the previous models misclassified. 
    - Examples: AdaBoost (Adaptive Boosting), Gradient Boosting, XGBoost (Extreme Gradient Boosting), LightGBM (Light Gradient Boosting Machine), CatBoost, GBM (Gradient Boosting Machine)
      
#### Q10- What is Reinforcement Learning?

Is a type of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives feedback in the form of rewards or penalties based on its actions, allowing it to learn optimal strategies over time to maximize cumulative rewards. It is inspired by the way humans and animals learn from trial and error.

Here are some applications of Reinforcement Learning : 
- Automated Robots
- Natural Language Processing
- Marketing and Advertising 
- Image Processing
- Recommendation Systems
- Traffic Control 
- Healthcare 
- Etc.
#### Q11- What is Recommender Systems
Also known as recommendation systems or engines, are applications or algorithms designed to suggest items or content to users based on their preferences and behavior. These systems leverage data about users and items to make personalized recommendations, aiming to enhance user experience and satisfaction. There are two main types of recommender systems:

- Content-Based Recommender Systems
- Collaborative Filtering Recommender Systems
Recommender systems are widely used in various industries, including e-commerce, streaming services, social media, and more. They help users discover new items, increase user engagement, and contribute to business success by promoting relevant content and products

#### Q12- What is Content-Based Recommender Systems ? 
#### Q13- What is Collaborative Filtering Recommender Systems ?
#### Q14- What is Ranking ? 
Ranking in machine learning refers to the process of assigning a meaningful order or ranking to a set of items based on their relevance or importance. This is often used in scenarios where the goal is to prioritize or sort items based on their predicted or observed characteristics.

Ranking problems are common in various applications, including information retrieval, recommendation systems, and search engines.

### Part II: Model building
#### Q1- What are three stages of building a machine learning model ? 
- The process of building a machine learning model includes three main stages, These stages are:
    - **Training phase:** after splitting the data into training and testing sets, training data is used to train our model on a labeled dataset. During the training phase, the model tries to learn relationships between input data and the corresponding output target values while adjusting its internal parameters. Throughout this phase, the model aims to maximise the accuracy of making precise predictions or classifications when exposed to unseen data.
    - **Validation phase:** after the model is well trained, we evaluate it on a seperate dataset known as the validation set (maximum 10% of our data). This dataset is not used during the training process. Validation stage helps identify the existence of certain overfitting (model performing well on training data but poorly on new data) or certain underfitting (model needs more training to capture the underlying patterns in the data).
    - **Testing (Inference) phase:** during this phase, the trained and validated model is applied to unseen dataset, called test dataset. This phase aims to evaluate the model's performance and provides a measure regarding the model's effectiveness and its ability to make accurate predictions in a production environment.

#### Q2- How to split your data while building a machine learning model ?    
- During the model building phase, it is required to split the data into three main sets to evaluate the model's performance and effectiveness. The three sets are: 
    - Training: used to train the model and learn relationship between inputs and outputs, contains 70-80% of our total dataset
    - Validation: used to validate the model, fine-tune the model's hyperparameters and assess its performance during training, it helps to prevent overfitting and underfitting. It contains 10-15% of the total data
    - Testing: used to test and evaluate the model's performance against unseen data and after validation phase. It is used to measure how effective will our built model be in a production environment. It contains 10-15% of the total data.

- Splitting data is accomplished after the preprocessing phase (handle missing values, categorical features, scale features, etc.). 
- It is important to ensure that the split is representative of the overall distribution of the data to avoid biased results.
- It is favorable to use cross-validation technique. 
- No fixed rule to split data between training, validation and testing, portions can vary based on individual preferences.
  
#### Q3- How to choose which ML algorithm to use given a dataset?
- Choosing the right machine learning algorithm for a given dataset involves considering various factors related to the nature of the data and the problem at hand.
- No master algorithm it all depends on the situation
- Here's a step-by-step guide to take the right decision : 
    - **Understand the Problem or the situation :**
        - Understanding the nature of taregt variable (continous, categorical?, is all output variables labled or mixed?). 
        - Determine the problem we are trying to solve ( is it classification, regression or clustering?)
    - **Domain Knowledge:**
        - Trying to find any domain-specific knowledge that might influence the choice of algorithm. 
        - Certain algorithms may be well-suited to specific industries or types of problems.
    - **Explore the Data:**
        - Determine data dimension 
        - Perform exploratory data analysis (EDA) to understand the characteristics of the dataset.
        - Understand features distribution, identify patterns and detect outliers etc.
    - **Consider the Size of the Dataset:**
        - Small: simpler models or models with fewer parameters may be more suitable to avoid overfitting.
        - Large: more complex models can be considered.
    - **Check for Linearity:** 
        - Studying the relationships between features and the target variable are linear or nonlinear.
        - If linear: then use linear models as they are more effective in this case.
        - If nonlinear: then non linear models (e.g., decision trees, support vector machines) may be suitable for more complex relationships.
    - **Data pre-processing :**
        - Handle Categorical Data : some algorithms handle categorical data naturally (e.g., decision trees), while others may require encoding.
        - Dealing with Missing Values: some algorithms can handle missing data, while others may require imputation or removal of missing values.
        - Check for Outliers: some algorithms may be sensitive to outliers, while others are more robust.
    - **Consider the Speed and Scalability:**
        - Take into account the computational requirements of the algorithm.
        - Some algorithms are faster and more scalable than others, making them suitable for large datasets.
    - **Evaluate Model Complexity:**
        - Simple models like linear regression are interpretable but may not capture complex patterns. 
        - More complex models like ensemble methods (e.g., random forests, gradient boosting) can capture intricate relationships but may be prone to overfitting.
    - **Validation and Cross-Validation:**
        - Use validation techniques, such as cross-validation, to assess the performance of different algorithms.
        - This helps you choose the one that generalizes well to new, unseen data.
    - **Experiment and Iterate:**
        - It's often beneficial to experiment with multiple algorithms and compare their performance.
        - Iterate on the choice of algorithm based on performance metrics and insights gained during the modeling process.

**Note:**
- There is no one-size-fits-all solution, and the choice of the algorithm may involve some trial and error.
- Additionally, ensemble methods, which combine multiple models, can sometimes provide robust solutions.
- Keep in mind that the performance of an algorithm depends on the specific characteristics of your dataset and the goals of your analysis.      

#### Q4- What is Overfitting, causes and mitigation? 

- Overfitting is a common challenges in machine learning that relate to the performance of a model on unseen data.
- It occurs when a machine learning model learns the training data too well, capturing noise and random fluctuations in addition to the underlying patterns (as concept).
- High error on testing dataset.

#### Q4. 1-Key characteristics of Overfitting :

- Excellent Performance on Training Data.
- Poor Generalization to New Data
- Low Bias, High Variance: the model is not biased toward any particular assumption, but its predictions are highly sensitive to variations in the training data.
- Complex Model: overfit models are often overly complex and may capture noise or outliers as if they were significant patterns.
- Memorization of Training Data: instead of learning the underlying patterns, an overfit model may memorize specific details of the training data, including noise and outliers.

#### Q4. 2- Causes of Overfitting : 
- Too many features or parameters.
- Model is too complex for the available data.
- Training on a small dataset or training for too many iterations

#### Q4. 3- Overfitting mitigation :
- Regularization techniques (e.g., L1 or L2 regularization).
- Feature selection or dimensionality reduction.
- Increasing the amount of training data.
- Using simpler model architectures with less variables and parameters so variance can be reduced.
- Use of Cross-validation method

**Note:**
- It is important to find balance between model complexity and the ability to generalize to new, unseen data.

#### Q5- What is Underfitting, causes and mitigation? 
- It is the case when the model is too simple to capture the underlying patterns in the training data.
- Besides, the model performs poorly not only on the training data but also on new, unseen data.
- High error rate on both training and testing datasets.
- It occurs when the model lacks the complexity or flexibility to represent the underlying relationships between the features and the target variable.

#### Q5. 1-Key characteristics of underfitting :
- Poor Performance on Training Data
- Poor Generalization to New Data
- High Bias, Low Variance : The model is biased toward making overly simple assumptions about the data.
- Inability to Capture Patterns
- Simplistic Model: underfit models are often too simplistic and may not capture the nuances or complexities present in the data.

#### Q5. 2- Causes of Underfitting: 
- Too few features or parameters: inadequate feature representation.
- Insufficient model complexity: using a model that is too basic for the complexity of the data.
- Inadequate training time or data.


#### Q5. 3- Underfitting mitigation: 
- Increasing the complexity of the model
- Adding relevant features.
- Training for a longer duration.
- Considering more sophisticated model architectures.

**Note:**
- Increasing model complexity excessively may lead to overfitting. 
- Achieving a balance between overfitting and underfitting is crucial.
- This balance, often referred to as the model's "sweet spot", results in a model that generalizes well to new, unseen data. 
- Techniques like cross-validation, hyperparameter tuning, and monitoring learning curves can help strike this balance during the model development process.
  
#### Q6- What are the types of Regularization in Machine Learning?

- Regularization is a technique used in machine learning to prevent overfitting by adding a penalty term to the objective/loss function.
- It consists on adding a cost term that penalize the large weights of model.
- There are mainly two types of regularization commonly used: L1 regularization (Lasso) and L2 regularization (Ridge). - Additionally, Elastic Net is a combination of both L1 and L2 regularization. 

Here are all the used techniques in ML : 
- L1 Regularization (Lasso)
- L2 Regularization (Ridge)
- Elastic Net

#### Q6. 1- What is L1 Regularization (Lasso)?
L1 regularization tends to shrink some coefficients exactly to zero, effectively excluding the corresponding features from the model. It is often used when there is a belief that some features are irrelevant. The penalty term is the sum of the absolute values of the regression coefficients.

#### Q6. 2- What is L2 Regularization (Ridge)?
L2 regularization tends to shrink coefficients toward zero without eliminating them entirely. It is effective in dealing with multicollinearity (high correlation between predictors) and preventing overfitting. The penalty term is the sum of the squared values of the regression coefficients.

#### Q6. 3- What is Elastic Net?

Elastic Net combines both L1 and L2 penalties in the objective function. It has two control parameters, alpha (which controls the overall strength of regularization) and the mixing parameter, which determines the ratio between L1 and L2 penalties. It is useful when there are many correlated features, and it provides a balance between Lasso and Ridge.

**Note:**
- These regularization techniques help improve the generalization performance of machine learning models by preventing them from becoming too complex and fitting noise in the training data.
- The choice between L1, L2, or Elastic Net depends on the specific characteristics of the dataset and the modeling goals.

#### Q7- What is Model Validation Technique?

Validation techniques in machine learning are essential for assessing the performance of a model and ensuring its ability to generalize well to unseen data. 

Here are some common validation techniques:
- Train-Test Split 
- K-Fold Cross-Validation 
- Stratified K-Fold Cross-Validation
- Leave-One-Out Cross-Validation (LOOCV)
- Holdout Validation
- Time Series Cross-Validation

#### Q8- What is train-test-validation split?
- It is an important step to indicate how well a model will perform with real-world, unseen data.
- A good train-test-validation split helps mitigate overfitting and ensures that evaluation is not biased.
- It consists on dividing input dataset into three subsets:
    - Training: 70-80% of the data
    - Validation: 10-15% of the data
    - Testing: 10-15% of the data
- This split aims to ensure that the model is trained on a sufficiently large dataset, validated on a separate set to fine-tune parameters, and tested on a completely independent set to provide an unbiased evaluation of its performance.

#### Q9- What is K-Fold Cross-Validation?
- It is a technique used to assess the performance and generalization ability of a model. 
- The input dataset will be divided into k equally sized folds/groups.
- (K-1) folds are used for training and one fold is used for testing. Then, we evaluate the model. 
- Repeating the training and evaluation K times.
- Each time a different fold is taken as the test set while the remaining data is used for training.
- Here are the steps of the process :
    - Data Splitting
    - Model Training and Evaluation : iteration
    - Performance Metrics : error, accuracy, recall, precision etc is evaluated for each iteration.
    - Average Performance : average performance (error) is evaluated across all K iterations ==> provide a more reliable estimate of the model's performance.
- Error formula : $e(n)={y(n)-\hat y(n)}$ is calculated for each iteration where $\hat y$ is the predicted value.
- Ideally, K is 5 or 10. The optimal value may depend on the size and nature of the dataset.
- A higher K value can result in a more reliable performance estimate but may increase computational costs.
- K-fold is very helpful to limit issues related to the variability of a single train-test split.==> It provides a more robust evaluation of a model's performance by ensuring that every data point is used for testing exactly once.
  
 #### Q10- What is Stratified K-Fold Cross-Validation? 
- It is an extension of K-Fold Cross-Validation that ensures the distribution of the target variable's classes is approximately the same in each fold as it is in the entire dataset.
- In case of imbalanced datasets, this technique is prefered because some classes may be underrepresented.
- It helps in addressing issues related to  overrepresented or underrepresented classes in specific folds, which could lead to biased model evaluations.
- Here are the main steps for Stratified K-Fold Cross-Validation :
    - Data Splitting : ensuring that each fold has an equal distribution for each class samples.
    - Model Training and Evaluation: the same K-fold cross-validation, steps repeated K times
    - Average Performance : the average performance is calculated at the end of all K iterations to provide a robust performance estimate.
    
#### Q11- What is Leave-One-Out Cross-Validation (LOOCV)?
- It is a specific case of k-fold cross-validation where the number of folds (K) is set equal to the number of data points in the dataset. 
- Each iteration one point is dedicated to testing while the remaining samples are dedicated for training
- The same as k-fold, we calculate the performance metric for each iteration then we evaluate the average.
- The process is repeated until each data point has been used as a test set exactly once.
- It has the next advantages: 
    - It minimizes bias introduced by the choice of a specific train-test split.
    - It provides a robust estimate of a model's performance since each data point serves as both training and test data.
    - It is computationally expensive, especially for large datasets, as we are going to create a model for each sample.
    - It is preferably to be used with only small datasets.
      
#### Q12- What is Holdout Validation ?
- It is known as a train-test split. 
- The input dataset will be divided into two subsets: a training set (70-80%) and a testing set (20-30%).
- The exact split ratio depends on factors such as the size of the dataset and the nature of the machine learning task.
- The testing set is called Holdout Set also and it helps gathering an initial estimate of a model's performance.
- The performance metrics are accuracy, precision, recall, error, etc
- This technique is suitable if the input dataset is large enough to provide sufficient data for both training and testing, and when computational resources are limited compared to more computationally intensive methods like cross-validation.
- This technique could be not too reliable as the model performance can be influenced by the specific random split of data into training and testing sets. 
- To address this variability, multiple iterations of the holdout process can be performed, and the results can be averaged.

#### Q13- What does bias term mean?
- In the context of machine learning, the term "bias" can refer to two different concepts:
    - Data bias
    - bias in the context of bias-variance tradeoff
    - And bias term in the context of linear models. 
    
#### Q13. 1-  What does data bias mean?
- It is when the available data used in the training phase is not representative of the real-world population or phenomen of study.
- It refers to the presence of systematic errors or inaccuracies in a dataset that can lead to unfair, unrepresentative, or skewed results when analyzing or modeling the data.
- The existence of biased data can lead to undesired and often unfair outcomes (discriminatory results) when the model is applied to testing data because the model will learn these biases too. 
- A high-bias model makes strong assumptions about the underlying relationships in the data, and as a result, it may overlook complexity and fail to capture the true patterns.
- Example: a biased facial recognition model may perform poorly for certain demographic groups.
- Various types of bias are existing :
    - Selection bias : when the process of selecting data points for the dataset is not random, leading to a non-representative sample
    - Measurement bias : errors or inconsistencies in how data is measured or recorded. Examples: errors in sensors, discrepancies in data collection methods, or differences in measurement standards.
    - Sampling Bias : if the method used to collect or sample data introduces a bias. 
    - Observer Bias: occurs when the individuals collecting or interpreting the data have subjective opinions or expectations that influence the data
    - Historical Bias : when historical data includes biased decisions or reflects societal biases, machine learning models trained on such data may perpetuate or even exacerbate those biases.
    - Algorithmic Bias : occurs when machine learning algorithms learn and perpetuate biases present in the training data. 

**Note:**
    - Addressing data bias is an ongoing challenge in the field of machine learning, and researchers and practitioners are actively working to develop methods and tools to identify, measure, and mitigate bias in models.
    
#### Q13. 2- What does Bias-variance trade off mean?
- It is a fundamental concept in machine learning that involves finding the right balance between two sources of error, namely bias and variance, when building predictive models.
- The tradeoff arises because decreasing bias often increases variance, and vice versa. 
- Key points about the bias-variance tradeoff: 
    - **High bias : underfitting:** a model is too simple and it missunderstand the relevant relations between features and target outputs. It leads to systematic errors on both the training and test data.
    - **High variance : overfitting:** a model is too complex and fits the training data too closely, capturing noise as if it were a real pattern. It perform poorly on new, unseen data. 
- The goal is to find the optimal model complexity that minimizes both bias and variance, resulting in good generalization to new data.

![title](images/bias_variance_tradeoff.jpeg)  

#### Q13. 3- How to find the right balance between variance and bias?
- Cross-validation techniques, such as k-fold cross-validation, can be used to estimate a model's bias and variance and guide the selection of an appropriate model complexity.
- Techniques like regularization can be employed to penalize overly complex models, helping to mitigate overfitting and find a better bias-variance tradeoff.

####  Q13. 4- What is Bias-Variance Decomposition? 
- It is a mathematical expression that breaks down the mean squared error (MSE) of a predictive model into three components: bias squared, variance, and irreducible error. 
- Formula : $$MSE=Bias^2 +Variance+Irreducible Error$$
- Irreducible error is the inherent noise or randomness in the data that cannot be reduced by any model. It represents the minimum achievable error, even with a perfect model.
- This decomposition provides insights into the sources of error in a model and helps to illustrate the bias-variance tradeoff.
- We aim to find the optimal model complexity that minimizes the sum of bias squared and variance.

**Note:**
- Balancing bias and variance is a central challenge in machine learning, and understanding this tradeoff is essential for model selection, training, and evaluation. 
    
#### Q13. 5- How to mitigate Bias in ML?      
- To mitigate bias it's crucial to accomplish well studied steps:
    - Collecting diverse and representative data.
    - Implement ethical data collection practices.
    - Thoroughly processing it to detect and identify biases in the data. 
    - Regularly checking model predictions to ensure fairness and to detect and rectify biases in the model predictions.
    - Develop and use algorithms that are designed to be aware of and mitigate biases : `Fairness-aware Algorithms`


