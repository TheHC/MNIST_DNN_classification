# coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf 
import numpy as np 
from Batching import get_batches
import matplotlib.pyplot as plt

size_inputs=784 # images are 28*28
size_classes=10 # 10 digits ( 0-9 ) 

# importing data
mnist=input_data.read_data_sets('./datasets/mnist', one_hot=True)

#preparing data : mnist is already shuffled
#Training images and labels 
T_features=mnist.train.images
T_labels=mnist.train.labels.astype(np.float32)

#Validation images and labels
V_features=mnist.validation.images
V_labels=mnist.validation.labels.astype(np.float32)

#Testing images and labels
TS_features=mnist.test.images
TS_labels=mnist.test.labels.astype(np.float32)

#placeholder for features and labels batches ( placeholder because we don't want to initialize them, the init value will be fed in the session
features=tf.placeholder(tf.float32, [None, size_inputs])
labels=tf.placeholder(tf.float32,[None,size_classes])

#Variables for Weights and biases : init point sir a draw from a normal distribution
W=tf.Variable(tf.random_normal([size_inputs, size_classes]))
b=tf.Variable(tf.random_normal([size_classes]))

#Tensor for logits : xW+b ( not Wx+b because the features are rows not columns)
logits=tf.add(tf.matmul(features,W),b)

#Defining a placeholder for the learning rate ( value will change to use learning decay)
learning_rate=tf.placeholder(tf.float32)

#Defining the softmax ad cross entropy
soft_cross=tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

#Defining the cost or loss
loss=tf.reduce_mean(soft_cross)

#Defining the optimizer GD
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

#Getting if a prediction is true 
#for each row of fed features , representing one input , the netword gives us a row of logits having the same size as the row of labels. By comparing the index of the highest logit or highest probability ( softmax ) and the index of the 1 in the label row we can figure out if the prediction is true. Guys, this is exactly what supervised learning means. 
right_prediction=tf.equal(tf.argmax(logits,1),tf.argmax(labels,1))

#Calculating accuracy : which is the proportion of true predicitons 
accuracy=tf.reduce_mean(tf.cast(right_prediction, tf.float32))

# Tuning happens here : 
batch_size=50
epochs=300
learn_rate=0.01

#Create variables to plot 
Epoch_plot=np.arange(epochs)
Accuracy_plot=[]
Loss_plot=[]

#Define a function to print stats at the end of each epoch : loss and accuracy 
def print_stats(epoch_i,ses, current_features, current_labels):
	current_loss=ses.run(loss,feed_dict={features:current_features, labels:current_labels})
	Loss_plot.append(current_loss)
	validation_accuracy=ses.run(accuracy,feed_dict={features: V_features, labels: V_labels})
	print(' Epoch = {}  | Loss = {:3.4} | Validation Accuracy = {:3.4}'.format(epoch_i,current_loss,validation_accuracy))
	Accuracy_plot.append(validation_accuracy)
# init for variables 
init=tf.global_variables_initializer()




train_batches=get_batches(batch_size, T_features, T_labels)

with tf.Session() as ses:
	ses.run(init)

	#Going over each learning epoch
	for epoch_i in range(epochs):

		#Going over all batches
		for batch_features, batch_labels in train_batches:
			#defining a feed_dict for the placeholders
			train_feed_dict={features: batch_features,labels: batch_labels,learning_rate: learn_rate}
			ses.run(optimizer, feed_dict=train_feed_dict)
		print_stats(epoch_i, ses, batch_features, batch_labels)
	#Now calculating the final accuracy on test dataset 
	test_accuracy=ses.run(accuracy,feed_dict={features: TS_features, labels: TS_labels})
print('Test Accuracy :{}'.format(test_accuracy))

#plot the evolution of accuracy and cost 
plt.plot(Epoch_plot,Accuracy_plot,'b',label="Accuracy")
plt.plot(Epoch_plot,Loss_plot,'r',label="Loss")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
plt.savefig('./Results/graphs/accuracy_cost.png')
plt.show()




