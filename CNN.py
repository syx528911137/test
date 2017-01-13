import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def add_layer(inputs,in_size,out_size,activation_function = None):
    W = tf.Variable(tf.random_normal([in_size,out_size]))
    b = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    result = tf.matmul(inputs,W) + b
    output = 0
    if activation_function is None:
        output = result
    else:
        output = activation_function(result)
    return output


def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    #strides = [1,x_movement,y_movement,1]
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')





def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pred = sess.run(prediction,feed_dict={xs:v_xs})
    correct_predict = tf.equal(tf.argmax(y_pred,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result



mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])


x_image = tf.reshape(xs,[-1,28,28,1])

#conv1 layer
W_conv1 = weight_variable([5,5,1,32]) #patch 5 * 5 , in size 1,out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1) # out size 28 x 28 x 32
h_pool1 = max_pool_2x2(h_conv1)                         # out size 14 x 14 x 32
#conv2 layer

W_conv2 = weight_variable([5,5,32,64]) #patch 5 * 5 , in size 32,out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2) # output size 14 x 14 x 64
h_pool2 = max_pool_2x2(h_conv2)                         # output size 7 x 7 x 64

#func1
W_func1 = weight_variable([7*7*64,1024])
b_func1 = bias_variable([1024])
#[n_samples,7,7,64] -> [n_samples,7*7*64]
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])

h_func1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_func1) + b_func1)

#func2
W_func2 = weight_variable([1024,10])
b_func2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_func1,W_func2) + b_func2)







# prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))


train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)


for step in range(0,1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})

    if step % 50 == 0:
        print compute_accuracy(mnist.test.images,mnist.test.labels)












x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) + 0.5 + noise


xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])


layer1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
predict = add_layer(layer1,10,1,activation_function=tf.nn.relu)


loss = tf.reduce_mean(tf.reduce_sum(tf.square(predict - ys),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)


for step in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if step % 20 == 0:
        print sess.run(loss,feed_dict={xs:x_data,ys:y_data})





