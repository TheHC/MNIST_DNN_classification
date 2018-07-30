# MNIST_Logistic_classifier

This project aims to test the performance of a logistic classifier network on the MNIST dataset. 
The optimzer used is the  gradient descent optimizer.
The current performance was obtained by using a CPU : AMD Ryzen 1800x.
# Usage
Tune the hyperparameters :

    - Initial learning rate
    - Batch size
    - Number of epochs 
    
by modifying them in the Logistic_classifier.py script : 
```sh
# Tuning happens here : 
batch_size=50
epochs=300
learn_rate=0.01
```
Run the script Logistic_classifier.py and get the graph of the evolution of the accuracy and the loss in the folder Results/graphs under the name accuracy_cost.


### Installation

The Packages needed are :

    - Tensorflow 
    - numpy
    - matplotlib
  
### Todos  
    - Test the performance on diffrent hardwares
    - Include a script that plots the evolution of learning for different values of 
      the hyperparameters to show the effect of each one of them.

