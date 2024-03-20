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
