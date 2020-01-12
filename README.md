# Fun-with-TensorFlow

```ruby
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

# Initial Configuration
batch_size = 100
learning_rate = 0.5
training_epochs = 50

# Learning Rate Updates
#  global_step = tf.Variable(0, trainable=False)
#  starter_learning_rate = 0.1
#  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   #  100000, 0.96, staircase=True)

# Inverse Time Decay
#  global_step = tf.Variable(0, trainable=False)
#  decay_steps = 1.0
#  decay_rate = 0.5
#  learning_rate = tf.train.inverse_time_decay(learning_rate, global_step,
#          decay_steps, decay_rate)

# Neural Network Details:
# Layers: 1
# Width: 10 Neurons
# Activation: Softmax

# MNIST Database
# Training Set: 60,000 images and their labels
# Test Set: 10,000 images and their labels

# This command loads the data in from the data/ folder in a one-hot
# representation. 
mnist = mnist_data.read_data_sets("data", one_hot=True, validation_size=0)

# Defining the graph
X = tf.placeholder(tf.float32, [None, 784])
Y_ = tf.placeholder(tf.float32, [None, 10])

# Weight Definitions
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Defining Y as the output of the model.
Y = tf.nn.softmax(tf.matmul(X,W)+b)

# We define the cross entropy to use as a loss function
cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_, logits=Y))

# We define the sigmoid 
#  cross_entropy = tf.reduce_mean(
#        tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_, logits=Y))

# We define the accuracy 
correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# Defining the training steps
#  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
#  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
# train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
#  train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
train_step = tf.train.MomentumOptimizer(learning_rate, 0.06).minimize(cross_entropy)
#  train_step = tf.train.MomentumOptimizer(learning_rate, 0.06).minimize(cross_entropy, global_step=global_step)

# Initialize the network model
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# We loop through the training data in batches
for epoch in range(training_epochs):
    
    # Number of Images in a batch
    batch_count = int(mnist.train.num_examples/batch_size)    

    for i in range(batch_count):
        batch_X, batch_Y = mnist.train.next_batch(batch_size)
        
        # Train through each batch
        sess.run([train_step], feed_dict={X:batch_X, Y_: batch_Y})
       
    if epoch % 10 == 0:
        # Print Updates
        print("Epoch: " + str(epoch))
        a,c = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        print("Accuracy: " + str(a) + ", Loss: ", str(c))

# Check accuracy and check loss
print("Finished training " + str(training_epochs) + " batches of size " +
        str(batch_size))
a,c = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
print("Final Accuracy: " + str(a) + ", Final Loss: ", str(c))
```
