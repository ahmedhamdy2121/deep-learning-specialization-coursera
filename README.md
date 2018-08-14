# [Deep Learning Specialization on Coursera](https://www.coursera.org/specializations/deep-learning)

**Master Deep Learning, and Break into AI**

Instructor: [Andrew Ng](http://www.andrewng.org/)

This repo contains all the lecture notes and supplemental notes for this specialization. ***I won't update it with my solutions because that would be ethically wrong!*** I am using it as a reminder for me in case I forgot something, so instead of go through the whole videos again, I will just check it first here ;)

## Programming Assignments
You can solve the assignments yourself even if you are just Auditing, sometime notebook won't open, so these are quick links for open-able notebook, then you shall choose **"File > Open"** and you will find all the weeks' materials, programming assignments, and datasets.
- [Open Python Notebook (Course 1)](https://www.coursera.org/learn/neural-networks-deep-learning/notebook/Zh0CU/python-basics-with-numpy-optional)
- [Open Python Notebook (Course 2)](https://www.coursera.org/learn/deep-neural-network/notebook/UAwhh/regularization)
- [Open Python Notebook (Course 4)](https://www.coursera.org/learn/convolutional-neural-networks/notebook/7XDi8/convolutional-model-step-by-step)


## Goals
- Learn the foundations of Deep Learning
- Understand how to build neural networks
- Learn how to lead successful machine learning projects
- Learn about Convolutional networks, RNNs, LSTM, Adam, Dropout, BatchNorm, Xavier/He initialization, and more.
- Work on case studies from health-care, autonomous driving, sign language reading, music generation, and natural language processing.
- Practice all these ideas in Python and in TensorFlow.


## Courses
### [Course 1: Neural Networks and Deep Learning](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/tree/master/01-Neural-Networks-and-Deep-Learning)
  
  - [Week 1 - Introduction to deep learning](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/tree/master/01-Neural-Networks-and-Deep-Learning/week1)
    - **Learning Objectives**
      - Understand the major trends driving the rise of deep learning.
      - Be able to explain how deep learning is applied to supervised learning.
      - Understand what are the major categories of models (such as CNNs and RNNs), and when they should be applied.
      - Be able to recognize the basics of when deep learning will (or will not) work well.
    - [Notes 1 - Welcome to the Deep Learning Specialization](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/blob/master/01-Neural-Networks-and-Deep-Learning/week1/Welcome-to-the-Deep-Learning-Specialization.md)
    - [Notes 2 - Frequently Asked Questions](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/blob/master/01-Neural-Networks-and-Deep-Learning/week1/Frequently-Asked-Questions_Coursera.md)
  
  - [Week 2 - Neural Networks Basics](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/tree/master/01-Neural-Networks-and-Deep-Learning/week2)
    - **Learning Objectives**
      - Build a logistic regression model, structured as a shallow neural network
      - Implement the main steps of an ML algorithm, including making predictions, derivative computation, and gradient descent.
      - Implement computationally efficient, highly vectorized, versions of models.
      - Understand how to compute derivatives for logistic regression, using a backpropagation mindset.
      - Become familiar with Python and Numpy
      - Work with iPython Notebooks
      - Be able to implement vectorization across multiple training examples
    - [Notes 1 - Logistic Regression as a Neural Network](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/blob/master/01-Neural-Networks-and-Deep-Learning/week2/01-Logistic-Regression-as-a-Neural-Network.ipynb)
    - [Notes 2 - Vectorization](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/blob/master/01-Neural-Networks-and-Deep-Learning/week2/02-vectorization.ipynb)
    - [Notes 3 - Standard Notation](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/tree/master/01-Neural-Networks-and-Deep-Learning/week2/standard-notation.pdf)

  - [Week 3 - Shallow Neural Networks](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/tree/master/01-Neural-Networks-and-Deep-Learning/week3)
    - **Learning Objectives**
      - Understand hidden units and hidden layers
      - Be able to apply a variety of activation functions in a neural network.
      - Build your first forward and backward propagation with a hidden layer
      - Apply random initialization to your neural network
      - Become fluent with Deep Learning notations and Neural Network Representations
      - Build and train a neural network with one hidden layer.
    - [Notes - Shallow neural networks](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/blob/master/01-Neural-Networks-and-Deep-Learning/week3/shallow-neural-network.ipynb)
      
  - [Week 4 - Deep Neural Networks](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/tree/master/01-Neural-Networks-and-Deep-Learning/week4)
    - **Learning Objectives**
      - Learning Objectives
      - See deep neural networks as successive blocks put one after each other
      - Build and train a deep L-layer Neural Network
      - Analyze matrix and vector dimensions to check neural network implementations.
      - Understand how to use a cache to pass information from forward propagation to back propagation.
      - Understand the role of hyper-parameters in deep learning
    - [Notes - Deep neural networks](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/blob/master/01-Neural-Networks-and-Deep-Learning/week4/deep-neural-networks.ipynb)
    
    
### [Course 2: Improving Deep Neural Networks: Hyper-parameter tuning, Regularization and Optimization](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/tree/master/02-Improving-Deep-Neural-Networks)

  - [Week 1 - Practical aspects of Deep Learning](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/tree/master/02-Improving-Deep-Neural-Networks/week5)
    - **Learning Objectives**
      - Recall that different types of initializations lead to different results
      - Recognize the importance of initialization in complex neural networks.
      - Recognize the difference between train/dev/test sets
      - Diagnose the bias and variance issues in your model
      - Learn when and how to use regularization methods such as dropout or L2 regularization.
      - Understand experimental issues in deep learning such as Vanishing or Exploding gradients and learn how to deal with them
      - Use gradient checking to verify the correctness of your back-propagation implementation
    - [Notes - Practical aspects of Deep Learning](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/blob/master/02-Improving-Deep-Neural-Networks/week5/Practical-aspects-of-Deep-Learning.ipynb)

  - [Week 2 - Optimization algorithms](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/tree/master/02-Improving-Deep-Neural-Networks/week6)
    - **Learning Objectives**
      - Remember different optimization methods such as (Stochastic) Gradient Descent, Momentum, RMSProp and Adam
      - Use random mini-batches to accelerate the convergence and improve the optimization
      - Know the benefits of learning rate decay and apply it to your optimization
    - [Notes - Optimization algorithms](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/blob/master/02-Improving-Deep-Neural-Networks/week6/optimization-algoritihms.ipynb)

  - [Week 3 - Hyper-parameter tuning, Batch Normalization and Programming Frameworks](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/tree/master/02-Improving-Deep-Neural-Networks/week7)
    - **Learning Objectives**
    	- Master the process of hyper-parameter tuning
      - Learning Batch Norm
      - Learn about multi-class classifier
      - Train how to use tensorflow
    - [Notes - Hyper-parameter tuning and programming frameworks](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/blob/master/02-Improving-Deep-Neural-Networks/week7/hyperparameter-tuning-and-programming-frameworks.ipynb)


### [Course 3: Structuring Machine Learning Projects](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/tree/master/03-Structuring-Machine-Learning-Projects)

  - [Week 1 - ML Strategy (1)](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/tree/master/03-Structuring-Machine-Learning-Projects/week8)
    - **Learning Objectives**
      - Understand why Machine Learning strategy is important
      - Apply satisficing and optimizing metrics to set up your goal for ML projects
      - Get to know single number evaluation metrics and how to deal with N metrics
      - Choose a correct train/dev/test split of your dataset
      - Understand how to define human-level performance
      - Use human-level perform to define your key priorities in ML projects
      - Take the correct ML Strategic decision based on observations of performances and dataset
    - [Notes - Introduction to ML Strategy](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/blob/master/03-Structuring-Machine-Learning-Projects/week8/introduction-to-ML-strategy.md)

  - [Week 2 - ML Strategy (2)](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/tree/master/03-Structuring-Machine-Learning-Projects/week9)
    - **Learning Objectives**
      - Understand what multi-task learning and transfer learning are
      - Manual help might be needed to assist in figuring out next steps
      - Building up your system quickly then iterate
      - Recognize bias, variance and data-mismatch by looking at the performances of your algorithm on train/dev/test sets
      - Get to know when to use Transfer Learning and Multi-task learning
      - Introduction to End-to-end deep learning
    - [Notes - Error Analysis](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/tree/master/03-Structuring-Machine-Learning-Projects/week9/error-analysis.md)
    - [Notes - End-to-end Deep Learning](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/tree/master/03-Structuring-Machine-Learning-Projects/week9/What_is_end_to_end_deep_learning.pdf)


### [Course 4: Convolutional Neural Networks](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/tree/master/04-Convolutional-Neural-Networks)

  - [Week 1 - Foundations of Convolutional Neural Networks](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/tree/master/04-Convolutional-Neural-Networks/week10)
    - **Learning Objectives**
      - Understand the convolution operation
      - Understand the pooling operation
      - Remember the vocabulary used in convolutional neural network (padding, stride, filter, ...)
      - Build a convolutional neural network for image multi-class classification
    - [Notes - Convolutional Neural Networks](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/blob/master/04-Convolutional-Neural-Networks/week10/Convolutional-Neural-Networks.md)

  - [Week 2 - Deep convolutional models: case studies](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/tree/master/04-Convolutional-Neural-Networks/week11)
    - **Learning Objectives**
      - Understand multiple foundational papers of convolutional neural networks
      - Analyze the dimensionality reduction of a volume in a very deep network
      - Understand and Implement a Residual network
      - Build a deep neural network using Keras
      - Implement a skip-connection in your network
      - Clone a repository from github and use transfer learning
    - [Notes - Practical advices for using ConvNets](https://github.com/ahmedhamdy90/deep-learning-specialization-coursera/blob/master/04-Convolutional-Neural-Networks/week11/Practical-advices-for-using-ConvNets.md)


Good luck :)



