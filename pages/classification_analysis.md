# ML : Supervised Learning 
# Classification Analysis
Supervised learning encompasses two main techniques: classification and regression. This page is all about classification analysis. It provides essential information and it detailed insights into machine learning models used for classification tasks. This resource is a valuable tool for aspiring data scientists and machine learning engineers preparing for technical interviews, offering a concise yet thorough guide whether you're reviewing basics or diving into specific concepts.

### Q0- What does classification analysis mean?
- It is one fundamental task in machine learning and data science. 
- It uses the supervised learning approach which consists on splitting input data into training and testing
- Commonly used in building predictive models to categorize data into predefined classes or categories based on input features or attributes. 
- Algorithm learns patterns and relationships between input features and class labels.
- Once the model is trained, it can predict the class labels of new or unseen data instances. 
- This predictive capability enables various applications such as spam detection, sentiment analysis, medical diagnosis, and fraud detection.

### Q1- What are some real-life applications of classification algorithms?
- Classification algorithms have numerous real-life applications across various fields.
- Here are some examples:
   - **Email Spam Detection:** classifying emails as spam or not spam 
   - **Medical Diagnosis:** predicting diseases or medical conditions 
   - **Sentiment Analysis:** classifying text data (e.g., customer reviews, social media posts) as positive, negative, or neutral sentiments.
   - **Credit Risk Assessment:** predicting the creditworthiness of individuals or businesses
   - **Image Recognition:** identifying objects like facial recognition, object detection, and medical imaging.
   - **Fraud Detection:** detecting fraudulent transactions or activities.
   - **Customer Churn Prediction:** predicting whether customers are likely to churn or leave a service. 
   - **Disease Outbreak Prediction:** identifying potential disease outbreaks or epidemics 
   - **Predictive Maintenance:** predicting equipment failures or malfunctions in industrial machinery, vehicles, or infrastructure

### Q2- Examples of well-known machine learning algorithms used to solve classification problems
Here are some well-known machine learning algorithms commonly used to solve classification problems:
- Logistic Regression
- Decision Trees
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Random Forest
- AdaBoost
- Gradient Boosting Machines (GBM)
- XGBoost
- CatBoost
- LightGBM
- Neural Networks (Deep Learning)

### Q3. how to choose a classifier based on training dataset size?
- For small training datasets:
  - Choose simpler classifiers like Naive Bayes or k-Nearest Neighbors (kNN).
  - It is better to use simple model with high bias and low variance.
  - They work better because they are less likely to overfit. 
- For Larger datasets:
  - It is better to use more complex classifiers such as Support Vector Machines (SVMs) or Random Forests.
  - It is better to use model with low bias and high variance
  - This model type will tend to perform better with complex relationships. 
-  Balancing variance and bias is essential for developing models that perform well on both training and unseen data.
- Simple classifiers are preferred to avoid overfitting and reduce computational complexity with limited data.
- Complex classifiers excel in capturing intricate patterns in larger datasets and handling higher dimensionality.

### Q4- How to address class imbalance in classification problems
- Two techniques are used to address class imbalance : 
   - Downsampling
   - Upsampling
- This two techniques are part of the feature engineering and the preprocessing phase.
- **Downsampling:** involves reducing the size of the majority class by randomly removing instances.
- **Upsampling:** involves increasing the size of the minority class by duplicating instances or generating synthetic samples.
- By balancing the class distribution, we improve the performance of machine learning models.

### Q5- How to evaluate a Classification model?
- Many metrics are commonly used to evaluate the performance of classification models in machine learning.
- The choice of metrics depends on the specific goals and characteristics of the classification problem.
- Here are some classification metrics:
    - Confusion matrix
    - Accuracy
    - Precision
    - F1 Score
    - Recall (Sensitivity or True Positive Rate)
    - Specificity or True Negative Rate
    - Area Under the Receiver Operating Characteristic (ROC) Curve (AUC-ROC)
    - Area Under the Precision-Recall Curve (AUC-PR) 
- The choice of metrics depends on the specific requirements of the classification problem (binary classification or multiclass classification).
- For example, in imbalanced datasets, where one class significantly has large number of samples than the second class, precision, recall, and F1 score are often more informative than accuracy.

### Q6- What is confusion matrix in classification problems?
- Confusion matrix is a table used to measure the performance of classification model
- It gives more details regarding the number of instances that were correctly or incorrectly classified for each class.
- The confusion matrix is a valuable tool for assessing the strengths and weaknesses of a classification model and guiding further optimization efforts.
- Here is an example of confusion matrix for a binary classification problem : 
![title](images/confusion-matrix1.jpeg)
##### 1. True Positive : 
- Samples that are from the positive class and were correctly classified or predicted as positive by the model.
##### 2. True Negative :  
- Samples that are from the negative class and were correctly classified or predicted as negative by the model.
##### 3. False Positive : 
- Samples that are from  the negative class but were incorrectly classified or predicted as positive by the model.
##### 4. False Negative : 
- Samples that are from  the positive class but were incorrectly classified or predicted as negative by the model

### Q7- How to define Accuracy?

- An evaluation metric used to evaluate the performance of classification model.

- Divides the number of correctly classified observations by the total number of samples.

- **Formula:** $$Accuracy ={ Number\ of\ Correct\ Predictions \over Total\ number\ of\ predictions }$$


- Here a second formula : $$Accuracy ={ TP + TN \over TP + TN + FP + FN }$$

### Q8- How to define Precision ?
- An evaluation metric that measures the accuracy of the positive predictions made by the model. 
- It divides the number of true positive predictions by the sum of true positives and false positives.
- It belongs to [0,1] interval, 0 corresponds to no precision and 1 corresponds to perfect precision.
- Precision = Positive Predictive Power
- **Formula:** $$Precision = {True\ Positives \over True\ Positives + False\ Positives}$$ 

### Q9- How to define Recall, Sensitivity or True Positive Rate?
- An evaluation metric that measures the ability of the model to capture all the positive samples.
- It divides number of true positives samples by the sum of true positives and false negatives.
- Recall = Sensitivity = True Positive Rate. 
- **Formula:** $$Recall= {True\ Positives \over True\ Positives + False\ Negatives}$$
  
### Q10- How to define F1-score? 
- An evaluation metric that combines both Precision and Recall.
- Wighted average of Precision and Recall.
- It can be calculated using the `f1_score()` function of `scikit-learn`
- F1 belongs to [0,1]: 0 is the worst case and 1 is the best.
- **Formula :** $$F1= {2×Precision×Recall \over Precision+Recall}$$
  
### Q11- How to define Specificity or True Negative Rate ?
- Specificity measures the ability of the model to correctly identify negative instances.
- It divides the true negatives samples by the sum of true negatives observations and false positives observations.
- True Negative Rate = Specificity
- **Formula:** $$Specificity={True\ Negatives \over True\ Negatives + False\ Positives}$$ 

### Q12- What is Receiver Operating Characteristic (ROC) and Area under-ROC curve (AUC-ROC)?
- ROC curve is a graphical representation of the model's performance across different classification thresholds.
- The shape of the curve contains a lot of information
- Area under the ROC curve : AUC-ROC provides a single metric indicating the model's ability to distinguish between classes.
- Here is ROC and AUC-ROC illustration:
<img src="images/roc-curve-original.png" width="400"> 
- If AUC-ROC is high, then we have better model. Else, we have poor model performance.
- Smaller values on the x-axis of the curve point out lower false positives and higher true negatives.
- Larger values on the y-axis of the plot indicate higher true positives and lower false negatives.
- We can plot the ROC curve using the `roc_curve()` scikit-learn function.
- To calculate the accuracy, we use `roc_auc_score()` function of `scikit-learn`.
* Note: False Positive Rate = 1- Specificity

*source: https://sefiks.com/2020/12/10/a-gentle-introduction-to-roc-curve-and-auc/

### Q13- What is Area Under the Precision-Recall Curve (AUC-PR)?
- Similar to AUC-ROC, AUC-PR represents the area under the precision-recall curve.
- It provides a summary measure of a model's performance across various levels of precision and recall.
- It can be calculated using the `precision_recall_curve()` function of `scikit-learn`.
- The area under the precision-recall curve can be calculated using the `auc()` function of `scikit-learn` taking the recall and precision as input.

<img src="images/precision_recall_curve.png" width="400"> 

*source: https://analyticsindiamag.com/complete-guide-to-understanding-precision-and-recall-curves/

- The same here if AUC-PR is high, then we have better model. Else, we have poor model performance.
- The recall is provided as the x-axis and precision is provided as the y-axis.

### Q14- When to Use ROC vs. Precision-Recall Curves?

- Choosing either the ROC curves or precision-recall curves depends on your data distribution:
    - ROC curves: preferable to be used when there are roughly equal numbers of observations for each class.
    - ROC curves provide a good picture of the model when the dataset has large class imbalance.
    - Precision-Recall curves should be used when there is a moderate to large class imbalance.

### Q15- Classification Report Scikit-learn? 
- The `classification_report` function of `scikit-learn` provides a detailed summary of classification metrics for each class in a classification problem. 
- The report contains the next metrics:
    - Precision
    - Recall- sensitivity
    - F1-score
    - Specificity
    - Support
- Support: the number of actual instances of each class in the dataset.
- Classification reports are evaluated using classification metrics that have precision, recall, and f1-score on a per-class basis.
  
### Q16 - How do we evaluate a classification report?
- High recall + high precision ==> the class is perfectly handled by the model. 
- Low recall + high precision ==> the model can not detect the class well but is highly trustable when it does.
- High recall + low precision ==> the class is well detected but model also includes points of other class in it. 
- Low recall + low precision ==> class is poorly handled by the model
  
### Q17- What is log loss fucntion?
- It is an evaluation metric used in logistic regression
- Called logistic regression loss or cross-entropy loss
- Input of this loss function is probability value that belongs to [0,1].
- It measures the uncertaintly of our prediction based on how much it varies from the actual label.

### Q18- Is accuracy always a reliable metric for evaluating classification models?
- No, it is not true. 
- In cases of imbalanced datasets, accuracy may not be reliable for evaluating model performance.
- Instead, precision and recall are preferred metrics for classification models. 
- Additionally, the f1-score, which combines precision and recall, provides a comprehensive measure of performance.

### Q19- What is Logistic Regression and how it works?
- It is a classification algorithm used to predict a discret output.
- Types of outputs: 
    - Binary (2 classes)
    - Multiple (>2 classes)
    - Ordianl (Low, medium, High)
- An activation function is applied to transform the linear combination of features into probabilities (predictions). 
- The activation function = the logistic function = The sigmoid function.
- Output:mx+b
- Sigmoid function formula: $$S(z)={1\over 1+ e^{-z}}$$
<div>
<img src="images/sigmoid-function.png" width="350"/>
</div>
Source (1): https://www.codecademy.com/resources/docs/ai/neural-networks/sigmoid-activation-function
### Q20- What is Decision Trees?
- It is a non-parametric supervised learning algorithm. 
- It has the tree structure: root node, edges (branches), internal and leaf nodes
- It can be used to solve both Classification and Regression problems.
- We build a tree with datasets broken up into smaller subsets while developing the decision tree
- It can handle both categorical and numerical data 

<img src="images/tree_structure.png" width="400"> 
### Q21- What are the types of decision tree 
- We have three main types :
    - **ID3:** Iterative Dichotomiser 3: splitting datasets is based on metrics like entropy and information gain. 
    - **C4.5:** it is identified as a later iteration of ID3, where it uses information gain or gain ratios to split datasets.
    - **CART:** Classification And Regression Trees: it utilizes Gini impurity to identify the ideal attribute to split the dataset on.
    
### Q22- What are the different DT menthods to split datasets?
Actually, you can find more details in the regression section. Here is the list of methods:
- **Variance** 
- **Entropy** 
- **Information gain**
- **Gini Impurity**
- **Chi-Square**

### Q23- How can we use Decision Trees to resolve classification tasks ?
- Decision trees are extensively for classification tasks.
- They work by recursively splitting the data into subsets based on the value of features.
- At each split, the algorithm selects the feature that best separates the data into classes.
- This process continues until a stopping criterion is met, such as reaching a maximum tree depth or having only samples from one class in a node.
- Decision trees are intuitive, easy to interpret, and can handle both numerical and categorical data.
