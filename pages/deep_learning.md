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
      
### Q23- What Are Hyperparameters and parameters of neural network?
- Neural network has several important hyperparameters and parameters. 
- Hyperparameters guide the learning process, while parameters are learned from data during training.
- **Hyperparameters:**
   - Set before the training process.
   - Control the model's architecture and learning process.
   - Tuning hyperparameters affects model performance.
   - Examples: learning rate, batch size, and number of layers.
- **Parameters:**
   - Learned during the training process.
   - Adjusted iteratively to minimize the loss/cost function.
   - Define the model's ability to capture patterns in the data.
   - Examples: weights and biases in the network's connections.

### Q24- What is the Learning rate in neural network and how  to choose its optimal value? 
- It is an hyperparameter of the model. 
- It aims to control the step size taken during updating the weights of the network. 
- It determines how much the model's parameters (weights) are adjusted with respect to the loss / cost function. 
- During the optimization process, we aim to minimize the gradient descent loss function
- The loss function corresponds to the difference between predicted and actual outputs.
- Choosing the optimal learning rate is crucial for achieving better performance :
  - High : faster convergence and might lead to overshooting the optimal solution.
  - Low: slower convergence but more stable training. 
- To choose the optimal learning rate, we use techniques like :
   - Grid search
   - Random search
   - Leverage adaptive learning rate algorithms : AdaGrad, RMSprop, or Adam.

### Q25- What is batch size in neural network ? 
- It corresponds to the number of training samples used in one iteration.
- It determines how many samples are propagated through the network before updating the model parameters during training.
- It is considered as an hyperparameter in neural network training.
- Adjusting the batch size can affect the convergence speed and generalization of the model
- It is an important aspect to tune during the training process.

### Q26- How to choose the best batch size ?
- Several factors must be considered while choosing the best batch size. 
- Factors can be : dataset size, model complexity, available computational resources, and training objectives
- However, it is important to try with different batch sizes to identify the best one for your specific task and constraints.
- Here are some ideas how to choose the right value :
   - Consider hardware limitations and dataset size: large datasets requires large memory and expensive computational resources.
   - Try various batch sizes : a range between 10 and 250 and choose the size that optimizes training speed and accuracy.
   - Balance between speed and accuracy: 
       - Smaller batch sizes may lead to slower convergence but: can provide more noise in gradient estimation + potentially leading to better generalization. 
       - Larger batch sizes may speed up training but could result in poorer generalization.
   - Consider total dataset size :
       - For large datasets: smaller batches can still offer sufficient randomness
       - For smaller datasets: bigger batches might be better.
   - Smaller batch sizes can help in mitigating overfitting during the optimization process

### Q27- What Is the difference between Epoch, Batch, and Iteration in Deep Learning?
In deep learning:
- Epoch: One pass through the entire dataset.--> numder of epochs defines the number of times the algorithm sees the entire dataset.
- Batch: Subset of the dataset used in one iteration.
- Iteration: One update of the model's parameters using one batch of data.
- An epoch consists of multiple iterations, and each iteration processes one batch of data.
### Q28- What is the significance of using the Fourier transform in Deep Learning tasks?
- It plays a crucial role in deep learning as it used for several tasks:
    -  **Feature Extraction:** extract frequency-domain features from signals or images, aiding in the representation of data for deep learning models.
    - **Data Preprocessing:** such as denoising, smoothing signals and remove irrelevant components, before feeding them into neural networks. 
    - **Data Augmentation:** applying transformations (Fourier transform and its variants, such as the Short-Time Fourier Transform (STFT) like rotation or scaling in the frequency domain.
    - **Efficient Convolution:** it can be used to implement convolution operations in convolutional neural networks (CNNs) and reducing computational complexity.
    - **Time-Series Analysis:** used to analyze time-series data or audio signals, and identify patterns or anomalies effectively.
      
## Part 2: Tensorflow, Keras and Pytorch
### Q1- What is Tensorflow and what it is used for?
- It is an Open-source machine learning framework by Google.
- It aims to Build, train, and deploy machine learning models.
- Key features: it supports neural networks, including deep learning models.
- Itallows users to build complex neural network architectures using high-level APIs like Keras or through its own lower-level API.
- It perform efficient training on large datasets, including distributed training across multiple GPUs or TPUs (Tensor Processing Units).
- Used to develop and deploy models for making predictions on new data.
- It offers an advanced visualization tools that are used for monitoring and analyzing model performance.
- It supports various deployment options, including exporting models to different formats for deployment on different platforms, such as mobile devices or the web.
- It is widely used in research and industry for tasks like image recognition and NLP.

### Q2- What are the programming elements in Tensorflow?

In TensorFlow, the main programming elements include:

- **Tensors:** fundamental data structures representing multi-dimensional arrays.
- **Variables:** mutable tensors that can hold state that can be updated during computation.
- **Constants:** parameters whose value does not change.
- **Placeholders:** allow us to feed data to a tensorflow model from outside a model
- **Layers:** high-level abstractions for building neural network layers.
- **Sessions:** execution environments where operations are evaluated and tensors are computed.
- **Operations:** mathematical operations that can be performed on tensors.
- **Graphs:** computational graphs that define the flow of data and operations.
- **Estimators:** high-level API for training and evaluating TensorFlow models.
- **Optimizers:** algorithms for optimizing the parameters of a model during training.
- **Loss Functions:** functions that compute the error or loss between predicted and actual values.
- **Metrics:** functions for evaluating the performance of a model.
### Q3- What does Tensor mean in Tensorflow?
- Tensor corresponds to fundamental data structures representing a multi-dimensional array that can be vectors, matrices and even complex data structures.
- The term "tensor" is a mathematical concept that represents data with multiple dimensions.
- In Tensorflow setup, a tensor can be considered as a container that can hold data in multiple dimensions.
- The data could be input data, model parameters, and outputs that is passed between operations in a computational graph.
- Tensorflow, perform mathematical operations on this data efficiently, even if it is of large-scale data. 

### Q4- What do placeholders mean in Tensorflow ? 
- First, they are used to create a computational graph. 
- At the beginning, they are considered as empty variables, that will be fill with data during execution (running the graph).
- Then, they allow us to feed data to a tensorflow model (a computational graph) from outside a model, when running the graph. 
- They are typically used to define the input and target data for training a machine learning model.
- To define a placeholder, we use the `tf.placeholder()` command. 
- This separation of graph definition and data feeding enables flexibility and efficiency in TensorFlow's execution.

### Q5-  Explain a Computational Graph.
- TensorFlow operates via constructing a computational graph.
- In the computational graph we have: 
    - Interconnected Nodes that correspond to mathematical operations such as  addition, multiplication, or convolution
    - Edges represent tensors, which are multi-dimensional arrays carrying data between nodes (data flow).
- By using operations within the graph, TensorFlow allows for dynamic computation and optimization during execution.
- This structure forms a "DataFlow Graph," enabling efficient computation and optimization of machine learning models.
- This graph-based approach facilitates distributed computing and parallel execution, enhancing scalability and performance.

### Q6- What do Variables and Constants mean in TensorFlow? 
- First of all, Variables and Constants are two important elements in building and training machine learning models using TensorFlow.
- **Variables:**
    - Mutable tensors that hold values that can be updated during computation,
    - They are typically used to represent trainable parameters, such as weights, biases, in machine learning models.
    - To define a variable, we use the `tf.Variable()` command and initialize them before running the graph in a session
    - Example : W = tf.Variable([.5].dtype=tf.float32)
- **Constants:**
    - Immutable tensors whose values remain fixed and constant during the execution of a TensorFlow graph
    - They are typically used to represent fixed values or hyperparameters in a model.
    - To define a constant we use  `tf.constant()` command.
    - Example: `a = tf.constant(6.0,tf.float32)`

### Q7- Explain Session in TensorFlow 
- It is an execution environment for running operations or evaluating tensors.
- It encapsulates the control and state of the TensorFlow runtime, allowing to perform computations on the defined computational graph.
- It manages the resources (memory allocation and device management) required for running the computations efficiently.
- It maintains the state of variables and other resources throughout the execution
- Here are how you should use Session:
    - First create a Session object in your TensorFlow program using `with tf.Session() as sess:` command
    - Then, run operations and evaluate tensors within the context of the Session using `sess.run(...)`
- TensorFlow setup gives you the flexibility in managing the flow of computations via having control over when to start and end the execution of operations within the Session.

### Q8- What is Keras and what it is used for ?
- It is a Python-based open-source framework, that simplifies the process of developing deep learning models.
- It offers a user-friendly API and a high-level interface for building, training, and deploying neural networks enabling rapid development and experimentation.
- Models in Keras are built using layers, which can be easily stacked and configured to create complex architectures.
- It supports both **convolutional and recurrent neural networks**, as well as combinations of the two.
- Also, it provides support for custom layers, loss functions, and metrics.
- It allows creating custom layers, callbacks, and regularizers tailored to specific requirements or tasks.
- Compatible with multiple backends, including TensorFlow, Theano, etc.
- Simplifies the process of developing deep learning models for various applications.

### Q9- What is Pytorch and what it is used for ?
- It is an open-source machine learning framework used for building deep learning models.
- It supports both:
    - Traditional feedforward networks
    - Advanced architectures like RNNs and CNNs.
- It has an automatic differentiation engine that enables gradient-based optimization methods for training neural networks. 
- It allows efficient gradients computation and complex optimization algorithms implementation simplification.
- It is widely used in research (enable explore new ideas and algorithms quickly) and industry (model serialization, inference optimization, and integration with other frameworks and platforms) for various machine learning tasks.
- It integrates smoothly with GPUs, accelerating computation for training large-scale models and handling massive datasets efficiently.
- It is used in many fields such as: 
    - Computer vision
    - Natural language processing
    - Reinforcement learning

**Notes:**
- RNNs: Recurrent Neural Networks
- CNNs: Convolutional Neural Networks

### Q10- Why is Tensorflow the most preferred Library in Deep Learning?
- **Flexibility:** it allows building a wide range of neural networks, from simple to complex architectures.
- **Community Support:** it has a large community with extensive documentation and resources available.
- **Performance:** it is optimized for speed and efficiency, utilizing hardware accelerators like GPUs and TPUs.
- **Ease of Use:** high-level APIs like Keras make it easy to build and train neural networks with an intuitive interface.
- **Deployment Options:** models can be deployed across various platforms, including mobile devices, desktops, servers, and the cloud.
- **Continuous Development:** it is actively maintained, receiving regular updates and improvements to stay current with the latest advancements.

### Q11- Tensorflow Versus Pytorch
- TensorFlow and PyTorch are two popular deep learning frameworks 
- The choice between them depends on:
   - Specific project requirements
   - Familiarity with the framework
   - Personal preference
- **Tensorflow :**
    - Based on theano library.
    - Developed By Google and has a larger user base and extensive documentation.
    - Easy to use
    - It Offers both:
      - high-level APIs like Keras for quick development of neural networks
      - lower-level APIs for fine-grained control over model architecture.
    - Has Tensorboard for visualizing deep learning models
    - Better support for deployment in production environments, has tools for mobile and embedded devices: 
       - TensorFlow Serving 
       - TensorFlow Lite 
    - Very performant, particularly on large-scale distributed training and deployment scenarios. 
- **Pytorch :**
    - Based on Torch library
    - Developed by Facebook/ Meta and has an active research community, particularly in academia.
    - Offers dynamic computation graphs, which makes it flexible, and easy to debug. 
    - It is more Pythonic and intuitive for developers.
    - Visualization features are existing
    - Historically, deployment in production has been more challenging compared to TensorFlow.
    - Competitive performance for its efficient GPU utilization and dynamic graph execution.
- **In summary:**
   - TensorFlow is often favored for its ease of use, deployment capabilities, and extensive ecosystem
   - PyTorch is more preferred by researchers and developers who value flexibility, dynamic computation graphs, and Pythonic design.

## Part 3: Model training and Evaluation

### Q1- What is the Cost Function? 
- It is also called loss function
- It measures the performance of the neural network model during training phase. 
- It calculates the difference between predicted and actual/true values. 
- During the training phase, the main goal is to minimize the cost function via adjusting the model's parameters to improve its accuracy

### Q2- What  do you  understand by Feedforward
- To train a neural network, we use two main algorithms:
  - **Feedforward**
  - **Backpropagation**
- It aims to find predictions and outputs via passing data through the neural network. 
- The information is following one direction, from input layer to output layer.
- It has no feedback loops or recurrent connections. 
- Each layer in the network takes the input data and transforms it, making it more abstract as it goes.
- The final /output layer provides the predictions that are compared to the true / target values to identify the loss or error.  
- It does not have memory, therefore, it is used for tasks where the input-output mapping is fixed and does not depend on previous states or inputs.

**Notes:**
- "More abstract" means that the features become less directly related to the raw input data and instead capture higher-level patterns or concepts
- Unlike recurrent networks, feedforward networks do not have memory and are typically used for tasks where the input-output mapping is fixed and does not depend on previous states or inputs.

### Q3- What  do you  understand by Backpropagation?
- It is a key algorithm used to train a neural network. 
- It aims to adjust the model's weights and biases to minimise the cost function and improve the model's accuracy.
- It involves propagating the error backward from the output layer to the input layer.
- It calculates the gradient of the loss function with respect to each parameter in the network.
- It enables efficient optimization using gradient descent or its variants.
- Iteratively, it adjusts the model's parameters based on these gradients to improve performance over time.

<div>
<img src="images/feed_back.png" width="500"/>
</div>
