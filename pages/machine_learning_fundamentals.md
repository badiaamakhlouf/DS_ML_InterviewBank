# Machine Learning Fundamentals 

This GitHub page is a guide for those gearing up for machine learning technical interviews. It covers the basics and essentials, focusing on questions related to machine learning fundamentals and model evaluation. The content includes detailed questions about three main machine learning approaches: regression, classification, and clustering.

Whether you're polishing your interview skills or seeking insightful questions as an interviewer, this page is a valuable resource to strengthen your grasp of machine learning basics.

**Note:** Your suggestions for improvement are welcome; please feel free to provide feedback to help maintain the precision and coherence of this page.
## List Of Questions:

### Q1- What does Machine Learning mean ? 
Machine Learning (ML) is a subset of artificial intelligence (AI) that focuses on the development of algorithms and statistical models that enable computers to perform tasks without explicit programming. The core idea behind machine learning is to allow machines to learn patterns, make predictions, or optimize decisions based on data.

Key concepts in machine learning include:
- Types of Machine Learning: supervised, unsupervised and semi-supervised
- Types of machine learning problems: classification, regression and clustering
- Split data into Training, validation and testing sets (case of supervised)
- Choose the right algorithm depends on the problem you want to solve
- Model Evaluation
- Hyperparameter Tuning
- Deployment
  
### Q2- What are three stages of building a machine learning model ? 
- The process of building a machine learning model includes three main stages, These stages are:
    - **Training phase:** after splitting the data into training and testing sets, training data is used to train our model on a labeled dataset. During the training phase, the model tries to learn relationships between input data and the corresponding output target values while adjusting its internal parameters. Throughout this phase, the model aims to maximise the accuracy of making precise predictions or classifications when exposed to unseen data.
    - **Validation phase:** after the model is well trained, we evaluate it on a seperate dataset known as the validation set (maximum 10% of our data). This dataset is not used during the training process. Validation stage helps identify the existence of certain overfitting (model performing well on training data but poorly on new data) or certain underfitting (model needs more training to capture the underlying patterns in the data).
    - **Testing (Inference) phase:** during this phase, the trained and validated model is applied to unseen dataset, called test dataset. This phase aims to evaluate the model's performance and provides a measure regarding the model's effectiveness and its ability to make accurate predictions in a production environment.
      
## Q3- What are the types of ML algorithms ? 
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

### Q4- What does supervised, unsupervised and semi-supervised mean in ML? 

In machine learning, the terms "supervised learning," "unsupervised learning," and "semi-supervised learning" refer to different approaches based on the type of training data available and the learning task at hand:

- **Supervised Learning :** training a model on a labeled dataset, where the algorithm learns the relationship between input features and corresponding target labels. Can be used for Regression (continous output) or Classification (discrete output). 
- **Unsupervised Learning :** Deals with unlabeled data and aims to find patterns, structures, or relationships within the data. Can be used for Clustering (Groups similar data points together) or association
- **Semi-Supervised Learning:** Utilizes a combination of labeled and unlabeled data to improve learning performance, often in situations where obtaining labeled data is challenging and expensive.

### Q5- What are Unsupervised Learning techniques ?
 We have two techniques, Clustering and association: 
 - Custering :  involves grouping similar data points together based on inherent patterns or similarities. Example: grouping customers with similar purchasing behavior for targeted marketing.. 
 - Association : identifying patterns of associations between different variables or items. Example: e-commerse website suggest other items for you to buy based on prior purchases.
 
### Q6- What are Supervised Learning techniques ? 
We have two techniques: classfication and regression: 
- Regression : involves predicting a continuous output or numerical value based on input features. Examples : predicting house prices, temperature, stock prices etc.
- Classification : is the task of assigning predefined labels or categories to input data. We have two types of classification algorithms: 
    - Binary classification (two classes). Example: identifying whether an email is spam or not.
    - Multiclass classification (multiple classes). Example: classifying images of animals into different species.

### Q7- Examples of well-known machine learning algorithms used to solve Regression problems

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

### Q8- Examples of well-known machine learning algorithms used to solve Classification problems

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

### Q9- Examples of well-known machine learning algorithms used to solve Clustering problems

Several well-known machine learning algorithms are commonly used for solving clustering problems. Here are some examples:

- K-Means Clustering 
- Hierarchical Clustering
- Agglomerative Clustering
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- Mean Shift
- Gaussian Mixture Model (GMM)

These algorithms address different types of clustering scenarios and have varying strengths depending on the nature of the data and the desired outcomes. The choice of clustering algorithm often depends on factors such as the shape of clusters, noise in the data, and the number of clusters expected.

### Q10- How to split your data while building a machine learning model ?    
- During the model building phase, it is required to split the data into three main sets to evaluate the model's performance and effectiveness. The three sets are: 
    - Training: used to train the model and learn relationship between inputs and outputs, contains 70-80% of our total dataset
    - Validation: used to validate the model, fine-tune the model's hyperparameters and assess its performance during training, it helps to prevent overfitting and underfitting. It contains 10-15% of the total data
    - Testing: used to test and evaluate the model's performance against unseen data and after validation phase. It is used to measure how effective will our built model be in a production environment. It contains 10-15% of the total data.

- Splitting data is accomplished after the preprocessing phase (handle missing values, categorical features, scale features, etc.). 
- It is important to ensure that the split is representative of the overall distribution of the data to avoid biased results.
- It is favorable to use cross-validation technique. 
- No fixed rule to split data between training, validation and testing, portions can vary based on individual preferences.
  
### Q11- How to choose which ML algorithm to use given a dataset?
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
