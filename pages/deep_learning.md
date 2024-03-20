# Advanced Machine Learning Topics 
# Deep Learning 

This Github page is a valuable resource for technical interviews on deep learning and neural networks 🧠. It consolidates essential questions, offering a comprehensive study tool 📚. Whether reinforcing foundational knowledge or preparing for an interview, this page It explores fundamental principles and inquiries relevant to algorithms, techniques, concepts etc., ensuring success in technical interviews.

## List of Questions
## Part 1: Deep Learning and Neural Networks
### Q0 -What is Deep Learning?

- It is a subset of machine learning that involves training artificial neural networks with multiple layers (deep neural networks). 
- It is used to model complex patterns and representations.
- DL models are trained using large amounts of data.
- The term "deep" refers to the depth of the neural network, which consists of multiple hidden layers through which data is processed.
- It eliminates the need for manual feature engineering by automatically learning hierarchical representations from raw data during the training process.

    
### Q1- What are some real-life applications of deep learning algorithms?
- DL is used in various real-life scenarios across different fields.
- Here are some common applications for DL:
    - **Computer Vision:** object detection, image classification, facial recognition, and autonomous driving
    - **Natural Language Processing (NLP):** translation, chatbots, sentiment analysis etc.
    - **Healthcare:** medical image analysis, disease diagnosis, personalized treatment etc.
    - **Finance:** fraud detection, credit scoring, customer service automation etc. 
    - **Recommendation Systems:** for e-commerce, social media and advertising etc.
    - **Robotics**
    - **Autonomous Vehicles**
    - **Manufacturing:** such as predictive maintenance, process optimisation, quality control

### Q2- What are the common types of deep learning ?
- Artificial Neural Network (ANN)
- Convolutional Neural Networks (CNNs):
- Recurrent Neural Networks (RNNs)
- Autoencoders
- Transformers
- Generative Adversarial Networks (GANs)
- Graph Neural Networks (GNNs)

###  Q3- What is neural network?
- A neural network is a computational model which, has the same operational structure like the way biological neural networks in the human brain works.
- Neural networks are designed to recognize patterns, perform tasks, and make decisions by learning from data.
- It is an important component of deep learning and artificial intelligence.
- Here are the characteristics and components of neural networks:
    - Neurons
    - Layers
    - Connections (Weights)
    - Activation Function
    - Feedforward and Backpropagation
    - Learning
    
<div>
<img src="images/neuralnet2.png" width="500"/>
</div>

*source : https://medium.com/@parekhdhruvish1331/getting-into-deep-learning-c6b270e43055

### Q4- What does Neurons mean in NN?
- They are nodes or computational units that are considered as basic building blocks of a neural network.
- They are inspired by the neurons in the human brain.
- They are used to process and transmit information. 
- They works as follow:
   - They receive input 
   - Then, apply to them a mathematical operation
   - Finally, they produce an output.
- Each layer of NN has multiple number of neurons which are interconnected together.
- The strength of these connections defined as weights that are used to determine the contribution of each neuron's output to the next layer. 

### Q5- What are the main Layers of NN?
- An ANN has three main layers:
    - **Input Layer:** receives the initial input data and passes it on to the next layer.
    - **Hidden Layers:** intermediate layers that process the input and generate output.
    - **Output Layer:** produces the final output or prediction based on the computations performed in the hidden layers..   
- Each subset of neurons belongs to a layer. 
- Depending on the architecture of our NN, the number and configuration of network hidden layers can vary.
- Knowing that, we have different types of NN architectures, such as feedforward, recurrent, and convolutional etc. where each architecture has its own specific layer configurations

<div>
<img src="images/neural_network.png" width="300"/>
</div>

### Q6- What does Connections (Weights) mean in NN?
- They are the parameters used to determine the relationship strength between neurons in consecutive layers. 
- Each connection is associated with a weight which is used to determine the contribution of the neuron's output to the next layer.
- During training, NNs learn from data through the adjustment of weights based on the error between the predicted output and the actual output. 
- This adjustment is accomplished using optimization algorithms such as gradient descent.
- The learning process enables the network to recognize patterns, generalize from examples, and make predictions on new, unseen data.

### Q7- What are the various activation functions used in NN?
- Each neuron has an activation function that determines its output based on the weighted sum of its inputs.
- They help in transforming the output of the neural network into a suitable format for the specific problem domain.
- Common activation functions are:
    - Sigmoid
    - Hyperbolic tangent (tanh)
    - Rectified linear unit (ReLU)
    - Leaky ReLU
    - Softmax
    - Swish
    
- The choice of the activation function depends on the tak: 
   - Binary Classification (Single Output Neuron): Sigmoid or Logistic function.
   - Multiclass Classification (Multiple Output Neurons): Softmax function.
   - Regression (Single Output Neuron): ReLU (Rectified Linear Unit) or no activation function (identity function).

    
<div>
<img src="images/neural1.png" width="500"/>
</div>

### Q8- Explain the Sigmoid function ?
- Formula : $σ(x) = {1 \over 1 + e^{-x}}$
- It compresses the input values, making sure they're all between 0 and 1.
- Used for binary Classification. 
- However, it suffers from the vanishing gradient problem.
- Illustration :
     
<img src="images/sigmoid-function.png" width="350"/>
   
** Source: https://www.codecademy.com/resources/docs/ai/neural-networks/sigmoid-activation-function   

### Q9- Explain the Softmax function ?
- Formula : $softmax(x_i) = {e^{x_i} \over \sum_{j=1}^{n}e^{x_j}}$
- It converts the output of a NN into a probability distribution over multiple classes. 
- Suitable for multi-class classification problems.
- Illustration :

<img src="images/Softmax1.png" width="350"/>

** Source : https://botpenguin.com/glossary/softmax-function

### Q10- Explain ReLU (Rectified Linear Unit) function      
- Formula : ReLU(x)=max(0,x), ReLU  returns x if x is positive, and 0 otherwise.
- It sets all negative values to zero and leaves positive values unchanged.
- It is widely used due to its simplicity and effectiveness in training deep neural networks.
- Illustration :
     
<img src="images/Relu.png" width="350"/>

** Source : https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/

### Q11- Explain Hyperbolic tangent (tanh) function
- Formula: $tanh(x) = {e^x - e^{-x} \over e^x + e^{-x}}$
- Similar to the sigmoid function, but it squashes the input values between -1 and 1.
- It is often used in recurrent neural network
- Illustration :

<img src="images/tanh.png" width="350"/>

** Source: https://en.m.wikipedia.org/wiki/File:Hyperbolic_Tangent.svg

### Q12- Explain Leaky ReLU function

- Formula: $ReLU_{Leaky}(x) = { x\  if\ x > 0,\ alpha \times x\ if\ x <= 0 }$, where is a small positive constant, typically in the range of 0.01 to 0.3, which determines the slope of the function for negative input values.
- It is similar to ReLU.
- However, it allows a small, positive gradient for negative inputs, which helps mitigate the "dying ReLU" problem.
- Illustration :
<img src="images/LeakyReLU.png" width="350"/>

** Source: https://www.researchgate.net/figure/ReLU-activation-function-vs-LeakyReLU-activation-function_fig2_358306930

### Q13- Explain Swish function
- Formula: $Swish(x) ={x . sigmoid(x)} = {x \over 1 + e^{-x}}$
- Proposed as an alternative to ReLU. 
- Swish is a smooth, non-monotonic activation function. 
- It is preferred to be used in various neural networks because it often works really well.
- Illustration :
<img src="images/Swish.png" width="400"/>

** Source: https://www.researchgate.net/figure/Self-Gated-Swish-Activation-Function-3_fig1_359571434

### Q14- What Is a Multi-layer Perceptron(MLP)?
- It is a type of ANN composed of multiple layers of neurons or nodes. 
- It has the next architecture:
    - One input layer
    - One or more hidden layers
    - One output layer
- As defined previously, weight is associated to each connection and each neuron is connected to all neurons in the consecutive layer.
- It has significant role in ML as it is able to learn complex patterns and nonlinear relationships in the data. 
- Actually, neurons use nonlinear activation functions which make the network able to learn complex patterns in data. 
- MLP is trained using backpropagation technique, where weights are adjusted to minimize the difference between its predictions and the true target values.
- MLP is used for multiple tasks: 
  - Classification
  - Regression 
  - Pattern recognition
  
### Q15- What is Shallow neural network?

- It is a single-layer neural network that has only one hidden layer between the input and output layers.
- It is simple and computationally efficient. 
- Each neuron in the hidden layer receives inputs from the input layer.
- Then, performs a weighted sum of these inputs, applies an activation function
- Finally, passes the result to the output layer.
- Shallow neural network can be used for simple linear classification or regression tasks.
  
### Q16- What is Deep neural network?

- It is a type of ANN that has multiple hidden layers
- It consist of an input layer, one or more hidden layers, and an output layer.
- Each layer has multiple neurons which perform computations on the input data using weighted connections and activation functions.
- During the training phase, DNN uses backpropagation and gradient descent algorithm to adjust the weights and connections biases based on minimizing the difference between predicted and actual outputs.
- DNN is able to learn complicated patterns and representations in data.
- They are more suitable for complex tasks such as : natural language processing, image recognition and speech recognition.


### Q17- Deep neural networks versus Shallow neural networks :

- **Shallow neural networks:**
    - Simple architecture : only one hidden layer.
    - They are more used with simple tasks, where the data has simple patterns or relationships.
    - They are suitable for cases when the dataset is small.  
    - They have a limited capacity to learn complex patterns in data.
    - They are simple, easier to train and computationally efficient. 
    
- **Deep neural networks:** 
    - They have complex architecture with multiple hidden layers.
    - They can capture and learn more complex patterns and non-linear relationships in data.
    - They are suitable for more challenging and complicated tasks with large amount of data.
    - They require more computational resources and large amount of labeled data to achieve optimal performance during training phase. 
    
### Q18- What types of data processing are typically performed in neural networks?
Typically, data processing in neural networks involves several steps:
- **Data Cleaning:** handling missing values, ensuring data coherence and removing outliers
- **Data Normalization:** scaling numerical features to a standard range.
- **Feature Engineering:** creating new features or transforming existing ones
- **Encoding Categorical features:** converting categorical variables into numerical form suitable for neural networks.
- **Data Augmentation:** generating additional training samples 
- **Dimensionality Reduction:** reducing the number of features
- **Train-Test Split:** splitting the dataset into training and testing subsets. 
- **Data Balancing:** addressing class imbalances by oversampling, undersampling, or using techniques like SMOTE.

### Q19- Why do we need Data Normalization in neural networks?
- It is used to achieve stable and fast training of the model
- It aims to bring all the features to a certain scale or range of values, usually between 0 and 1. 
- Without normalization, there's a higher risk of the gradient descent failing to converge to the global or local minima and instead oscillating back and forth.
- Normalized data reduces the likelihood of numerical instability that can occur when working with features with vastly different scales.
- It is used to prevent overfitting and improve the generalization ability of the model.
- Normalizing features ensures that they contribute equally to the model's learning process. 

### Q20- Why do we need Data Augmentation in neural networks?
Here are some points why Data Augmentation is important in neural network: 
- Increase the model robustness and improve performance via exposing it to a larger number of various data samples.
- Enhance the model generalization ability and improve performance. 
- Mitigating and reducing overfitting.
- It increases the effective size of the dataset to overcome the lack of data issue. Especially, when the provided dataset is so small. 
- It helps in reducing the dependency on Large Datasets -> the need for collecting and annotating large datasets is reduced.
- Training process is more cost effective 

### Q21- What does Image augmentation mean?

- An efficacious Technique when we do not have enough amount of data to train a DL model.
- It aims to increase the diversity of images in a dataset via applying various transformations to images : 
   - Rotation
   - Scaling and resizing
   - Cropping and flipping
   - Changing color and brightness
- It increases the diversity of the dataset without collecting new data.
- It helps improve the robustness and generalization ability of models by exposing it to a wider range of variations.
- It is widely used in computer vision tasks such as object detection and classification.
- It is very useful when dealing with limited or imbalanced datasets.
- It helps prevent overfitting and improves model performance on unseen data.
- DL frameworks such as TensorFlow and PyTorch provide built-in support for image augmentation.
- Augmentation parameters must be chosen carefully 


### Q22- How to address the problem of class imbalance ?
- It is important to address the class imbalance issue in neural networks to ensure that the model learns effectively from all exisitng classes. 
- Here are some common techniques to mitigate class imbalance:
    - **Data Augmentation:** generate artificial samples for minority classes.
    - **Resampling Techniques:**
        - Oversampling: increase the number of samples in the minority class by replicating existing samples or generating synthetic data.
        - Undersampling: Decrease the number of samples in the majority class to match the size of the minority class.
    - **Ensemble Methods:** various models are trained on different subsets of data or different algorithms are used, then merge their predictions to handle class imbalance.
    - **Cross-Validation techniques:** use validation techniques such as k-fold cross-validation.
    - **Weighted loss functions** which give more importance to minority class samples (Assign higher weights to the loss) during training, helping the model better learn from these instances and improve performance on imbalanced datasets.
    - **Use Specific algorithms:** which are designed to handle class imbalance, such as SMOTE (Synthetic Minority Over-sampling Technique) or ADASYN (Adaptive Synthetic Sampling).
    - **Choose the right Evaluation Metrics:** in case of class imbalance accuracy alone is not a robust or an accurate metric. Instead, we can use precision, recall, F1-score or ROC-AUC. 
