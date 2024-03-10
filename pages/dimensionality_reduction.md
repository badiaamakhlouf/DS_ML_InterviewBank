# ML: Unsupervised Learning 
# Dimensionality Reduction Techniques 

This page is your go-to resource for understanding dimensionality reduction techniques. It offers in-depth explanations of various concepts and provides a comprehensive range of questions and answers, from basics to advanced techniques. By delving into dimensionality reduction algorithms and their applications, it serves as an invaluable tool for interview preparation and skill enhancement in dimensionality reduction analysis.

## List Of Question 
### Q0- What does Dimensionality reduction mean?
- It is a technique used in machine learning and statistics to reduce the number of input variables or features in a dataset.
- The goal is to simplify the dataset while retaining its essential information and patterns.
- High-dimensional datasets, where the number of features is large, can suffer from the curse of dimensionality, leading to increased computational complexity and potential overfitting.
- Dimensionality reduction methods aim to overcome these challenges by transforming or selecting a subset of features, preserving the most relevant information and reducing noise in the data.
  
### Q1- What are the most common dimensionality reduction techniques include:
Here are popular examples of Dimensionality reduction techniques:
- Principal Component Analysis (PCA)
- Singular Value Decomposition (SVD)
- Independent Component Analysis (ICA)
- Linear Discriminant Analysis (LDA)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Autoencoders
- Locally Linear Embedding (LLE)
- Isomap (Isometric Mapping)
  
### Q2- What does Principal Component Analysis (PCA) mean? 
- It is an unsupervised dimensionality reduction technique that aims to transforms input data into a new set of uncorrelated features while keeping the maximum variance in the data.
- It can be applied to both supervised and unsupervised machine learning tasks
- To calculate it, we can use various python libraries such as `NumPy`, `SciPy`, and `scikit-learn`
- PCA serves primarily in two key use cases:
    - **Data Visualization:** it aids in visualizing complex datasets, providing valuable insights into the underlying patterns.
    - **Algorithm Optimization:** it can significantly accelerate the learning process of algorithms that may otherwise exhibit slow training speeds.
- Multiple methods and libraries are available for applying PCA:
  - **Method 1:** Using scikit-learn library: `sklearn.decomposition.PCA()`
  - **Method 2:** Using NumPy, SciPy libraries 
    
Here are the steps of calculating PCA using the covariance matrix and use eigenvalue decomposition to obtain the eigenvectors and eigenvalues. Here are the steps to apply :
 - 1. Standardise the data
 - 2. Compute the covariance matrix and use eigenvalue decomposition to obtain the eigenvectors and eigenvalues.
 - 3. Select the k largest eigenvalues and their associated eigenvectors.
 - 4. Transform the data into a k dimensional subspace using those k eigenvectors.
    
#### Q1.1- Case 1 : Using scikit-learn library
- This method is based on using `sklearn.decomposition.PCA()`, where n is the number of PCA components 
- How to choose the correct number of PCA Components ?
  - The first principal components that capture the most significant variance in the data
  - Example 97% or 99% of data variability.
- If we found the correct component number, example: n=2,  then we use the next code : PCA(n_components=2)  

#### Q1.2- Case 2: Using NumPy, SciPy libraries: 
- This method consists on applying PCA using the eigenvalue decomposition, which is based on finding the eigenvectors and eigenvalues.
-  Here are the consecutive steps to apply this method:
    1. Standardise the data
    2. Compute the covariance matrix and use eigenvalue decomposition to obtain the eigenvectors and eigenvalues.
    3. Select the k largest eigenvalues and their associated eigenvectors.
    4. Transform the data into a k dimensional subspace using those k eigenvectors.
       
### Q3- Why do we need to find eigenvalues and eigenvectors in PCA?
- In PCA, finding eigenvalues and eigenvectors is a crucial step in transforming the original data into a new coordinate system (principal components) that captures the maximum variance.
- PCA begins by calculating the **covariance matrix** of the original data. This matrix summarizes the relationships between different variables.
- **Eigenvalues:** The eigenvalues of the covariance matrix represent the variance of the data along the corresponding eigenvectors. Larger eigenvalues indicate directions of maximum variance, and smaller eigenvalues indicate directions of minimum variance.
- **Eigenvectors:** Eigenvectors are the directions (principal components) in which the data varies the most. Each eigenvector corresponds to a principal component, 
- The principal component directions are given by the eigenvectors of the matrix, and the magnitudes of the components are given by the eigenvalues.
- The eigenvectors are ranked in order of their corresponding eigenvalues.
- The first few eigenvectors (principal components) with the largest eigenvalues capture the most significant variance in the data ( 97% or 99% of data variability).
- By choosing a subset of these components, you can effectively reduce the dimensionality of the data while retaining the most important information.

**Note:**
- Eigenvectors are orthogonal, meaning they are perpendicular to each other. This orthogonality ensures that the principal components are uncorrelated, simplifying the interpretation of the transformed data.

### Q4- What does Singular Value Decomposition (SVD) means ? 
- Singular Value Decomposition (SVD) is a mathematical technique widely used in linear algebra and numerical analysis.
- is often used to reduce the number of features or dimensions in a dataset.
- The singular values obtained from the decomposition can be used to identify the most important components or features, and the corresponding vectors can be used to transform the data into a lower-dimensional space.
- Here are steps to apply this method:
  - It aims to represent the original matrix A  with fewer dimensions via decomposing it into three other matrices U, V and Σ.
  - The SVD of a matrix A is represented as: $A = U Σ V^T$ :
    - A: The original matrix that we want to decompose.
    - **Left Singular Vectors U:** These vectors form an orthonormal basis for the column space of the original matrix A. They capture the relationships between the rows of A.
    - **Singular Values Σ:** The singular values on the diagonal of Σ are the square roots of the eigenvalues of $AA^T$ (or $A^TA$). They represent the amount of variability or importance associated with each singular vector. The singular values are arranged in descending order.
    - **Right Singular Vectors $V^T$:** These vectors form an orthonormal basis for the row space of the original matrix A. They capture the relationships between the columns of A.
- This decomposition is widely used in signal processing, data analysis, and machine learning. Examples:
  - **Dimensionality Reduction** 
  - **Image Compression**
  - **Pseudo-Inverse** 
  - **Collaborative Filtering**
  - **Latent Semantic Analysis (LSA)**
    
### Q5- PCA Versus SVD? 
- PCA is a specific method for dimensionality reduction and data analysis, SVD is a more general matrix decomposition technique.
- PCA can be viewed as a special case of SVD when applied to the covariance matrix of the data.
- Both techniques have their applications and are widely used in various fields, often complementing each other in data analysis and modeling.
- Here some key differences in their formulations and applications:
  - PCA: aims to find the principal components (or directions) along which the data varies the most. SVD decomposes a matrix into three other matrices, capturing the inherent structure and relationships within the data.
  - PCA is a specific application of SVD where the input matrix is the covariance matrix of the data. SVD is a more general matrix decomposition technique applicable to any matrix.
  - PCA typically involves centering the data (subtracting the mean) before computing the covariance matrix. SVD can be applied directly to the original data matrix without the need for centering.
  - PCA Commonly used for dimensionality reduction, data visualization, and noise reduction. SVD Applied in a broader range of applications, including matrix inversion, image compression, collaborative filtering, and solving linear least squares problems.

### Q6- What does Independent Component Analysis (ICA) mean ? 
- ICA is a computational technique used in signal processing and data analysis.
- It aims to separate a multivariate signal into additive, independent components, with the assumption that the original signals are statistically independent and non-Gaussian.
- Here's a breakdown of key concepts related to Independent Component Analysis:
  - **Statistical Independence:** ICA assumes that the observed signals are composed of independent source signals. Independence is a crucial assumption, as it allows ICA to uncover the underlying sources.
  - **Non-Gaussianity:** Unlike Principal Component Analysis (PCA), which assumes that the components are orthogonal and Gaussian, ICA relies on the non-Gaussian nature of the sources. Non-Gaussianity is exploited as a criterion for finding independent components.
- It consists on finding independent components.
- Does not focus on variance issues, it focuses on the mutual Independence of the component.
- The general form of the linear mixing model in ICA is expressed as:
  - Form: $X=A⋅S$
  - **X:** is the observed signal (mixture).
  - **A:** is the mixing matrix (often unknown).
  - **S:** is the vector of independent source signals.
- Solution of previous form is :
  - $S=W⋅X=A^{-1} ⋅X$
  - S: represents the independent components identified by ICA
  - W: the inverse of the mixing matriX
- Here some examples of ICA Applications:
  - **Blind Source Separation:** Unmixing signals when the mixing matrix is unknown.
  - **Image Processing:** Separating mixed images into their constituent sources.
  - **Biomedical Signal Processing:** Separating brain signals (e.g., EEG or fMRI) into independent components.
    
**Note:**
- ICA is a powerful technique, especially when dealing with scenarios where the sources are mixed together, and the mixing process is unknown or complex.
  
### Q7- How to measure non-Gaussianity in ICA? 
ICA aims to break down a multivariate signal into independent components. It relies on the assumption that observed data stems from independent sources, and it's essential that these sources exhibit non-Gaussian behavior. This non-Gaussianity enables ICA to effectively discern and isolate the independent components.

Non-Gaussianity in Independent Component Analysis (ICA) can be measured using various statistical metrics or tests. Here are the main measures for non-Gaussianity: 

- Kurtosis: it quantifies the "tailedness" or peakedness of a distribution. In a non-Gaussian distribution, the kurtosis will deviate from the expected value for a Gaussian distribution, which is 3. Higher kurtosis indicates heavier tails than a Gaussian distribution, while lower kurtosis indicates lighter tails.
- Skewness: it measures the asymmetry of the distribution. In a non-Gaussian distribution, the skewness will deviate from 0, which is the expected value for a symmetric Gaussian distribution. Positive skewness indicates a longer tail on the right side of the distribution, while negative skewness indicates a longer tail on the left side.
- Negentropy: Negentropy is a measure of non-Gaussianity. It quantifies the difference between the entropy of a Gaussian distribution and the observed distribution. Lower negentropy values indicate closer resemblance to a Gaussian distribution.
- Mutual Information: it measures the amount of information that one random variable contains about another random variable. Since Gaussian distributions are maximally non-informative, components with low mutual information are more likely to be closer to Gaussian, while those with high mutual information are more likely to be non-Gaussian.
- Jarque-Bera Test: The Jarque-Bera test is a statistical test that assesses whether sample data have the skewness and kurtosis matching a normal distribution. A low p-value indicates departure from normality.

These measures and tests provide insights into the departure of the data distribution from a Gaussian distribution, which is essential for successful ICA decomposition.

### Q8- Fast-ICA versus ICA ?
Fast Independent Component Analysis (Fast-ICA) and Independent Component Analysis (ICA) are both techniques used for separating mixed signals into statistically independent components.
Here's a comparison between the two:

- **Speed and Efficiency:** Fast-ICA is optimized for computational efficiency (fast). It employs techniques such as negentropy maximization and gradient ascent to rapidly converge to a solution. However, traditional ICA algorithms may be computationally slower compared to Fast-ICA, especially for large datasets or high-dimensional signal sources.
- **Optimization Method:** Fast-ICA uses optimization methods like negentropy maximization or fixed-point iteration to find the separating matrix that maximizes independence. Traditional ICA methods may employ various optimization techniques, including gradient ascent, kurtosis maximization, or mutual information minimization.
- **Assumptions:** Fast-ICA often assumes that the sources are non-Gaussian and have non-quadratic cumulants. It aims to exploit non-Gaussianity to separate the signals. However, ICA assumes generally that the sources are statistically independent and aims to find a linear transformation that maximizes their independence.
- **Convergence:** Fast-ICA tends to converge faster due to its efficient optimization techniques, making it suitable for real-time applications or large datasets. However, traditional ICA convergence speed may vary depending on the optimization method and the complexity of the dataset.
- **Applications:** Fast-ICA used in signal processing tasks such as blind source separation, audio processing, and image analysis where computational efficiency is crucial. ICA applied in fields like neuroscience, telecommunications, and biomedical signal processing for separating mixed sources into meaningful components.
- **Robustness:** Fast-ICA is efficient but more sensitive to the choice of initialization and parameters, requiring careful tuning for optimal performance. However, traditional ICA methods may offer more robustness and flexibility, allowing for customization based on specific dataset characteristics.
  
Fast-ICA and ICA both serve the purpose of separating mixed signals into independent components, but they differ in terms of computational efficiency, optimization methods, assumptions, convergence speed, applications, and robustness. The choice between the two depends on the specific requirements of the task at hand and the characteristics of the dataset.


**Note:**
Please be aware that I am committed to maintaining the accuracy and relevance of this page. If you come across any errors, potential adjustments, or missing details, kindly bring them to my attention. Your feedback is invaluable, and together, we can ensure the continual improvement and accuracy of this resource.


