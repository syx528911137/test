#-*-coding:UTF-8-*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def compute_accuracy(v_xs,v_ys):
    global pred
    y_pred = sess.run(pred,feed_dict={xs:v_xs})
    correct_predict = tf.equal(tf.argmax(y_pred,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result




mnistdata = input_data.read_data_sets('MNIST_data',one_hot=True)


xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])


w = tf.Variable(tf.random_normal([784,10],dtype=tf.float32))
bias = tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[10]))
# tmp = tf.nn.dropout(tf.matmul(xs,w) + bias,0.6)
# #
# pred = tf.nn.softmax(tmp)
pred = tf.nn.softmax(tf.matmul(xs,w) + bias)

cost = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(pred),reduction_indices=[1]))


optim = tf.train.GradientDescentOptimizer(0.01).minimize(cost)


init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

for i in range(0,20000):
    traindatas,trainlabels = mnistdata.train.next_batch(100)
    sess.run(optim,feed_dict={xs:traindatas,ys:trainlabels})
    if i % 100 == 0:
        print compute_accuracy(mnistdata.test.images,mnistdata.test.labels)

