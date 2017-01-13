import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data




def precision(x_data,y_label):
    global predict
    ys_pred = sess.run(predict,feed_dict={xs:x_data})
    correct_pred = tf.equal(tf.argmax(ys_pred,1),tf.argmax(y_label,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    return sess.run(accuracy,feed_dict={xs:x_data,ys:y_label})

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

weight_conv1 = tf.Variable(tf.truncated_normal([5,5,1,32],stddev=0.1),name="wc1")
bias_conv1 = tf.Variable(tf.constant(0.1,shape=[32]),name="bc1")
rc_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,weight_conv1,strides=[1,1,1,1],padding="SAME")) #28 x 28 x 16
rp_conv1 = tf.nn.max_pool(rc_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME") #14 x 14 x 16

weight_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev=0.1),name="wc2")
bias_conv2 = tf.Variable(tf.constant(0.1,shape=[64]),name="bc2")
rc_conv2 = tf.nn.relu(tf.nn.conv2d(rp_conv1,weight_conv2,strides=[1,1,1,1],padding="SAME")) #14 x 14 x 32
rp_conv2 = tf.nn.max_pool(rc_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME") #7 x 7 x 32

weight_func3 = tf.Variable(tf.truncated_normal([7 * 7 * 64,1024],stddev=0.1),name="w3")
bias3 = tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[1024]),name="b3")

rp_conv2_reshape = tf.reshape(rp_conv2,[-1,7*7*64])

func1_output = tf.nn.relu(tf.matmul(rp_conv2_reshape,weight_func3) + bias3)
keep_prob = tf.placeholder(tf.float32)
func_dropout = tf.nn.dropout(func1_output,keep_prob)

weight_func4 = tf.Variable(tf.truncated_normal([1024,10],stddev=0.1),name="w4")
bias4 = tf.Variable(tf.constant(0.1,tf.float32,[10]),name="b4")


predict = tf.nn.softmax(tf.matmul(func1_output,weight_func4) + bias4)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(predict),reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)



init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


for i in range(0,1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.8})
    if i % 10 == 0:
        print compute_accuracy(mnist.test.images,mnist.test.labels)
print compute_accuracy(mnist.test.images,mnist.test.labels)














