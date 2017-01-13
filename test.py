# #-*-coding:UTF-8-*-
# import tensorflow as tf
# import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
#
# def add_layer(inputs,in_size,out_size,activation_function = None):
#     W = tf.Variable(tf.random_normal([in_size,out_size]))
#     b = tf.Variable(tf.zeros([1,out_size]) + 0.1)
#     result = tf.matmul(inputs,W) + b
#     output = 0
#     if activation_function is None:
#         output = result
#     else:
#         output = activation_function(result)
#     return output
#
#
# def compute_accuracy(v_xs,v_ys):
#     global prediction
#     y_pred = sess.run(prediction,feed_dict={xs:v_xs})
#     correct_predict = tf.equal(tf.argmax(y_pred,1),tf.argmax(v_ys,1))
#     accuracy = tf.reduce_mean(tf.cast(correct_predict,tf.float32))
#     result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
#     return result
#
#
#
# mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#
# xs = tf.placeholder(tf.float32,[None,784])
# ys = tf.placeholder(tf.float32,[None,10])
#
#
#
#
#
#
#
#
# prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)
#
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))
#
#
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
#
# sess = tf.Session()
# init = tf.initialize_all_variables()
# sess.run(init)
#
#
# for step in range(0,10000):
#     batch_xs,batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
#
#     if step % 50 == 0:
#         print compute_accuracy(mnist.test.images,mnist.test.labels)
#
#
#
#
#
#
#
# # mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
# # x = tf.placeholder(tf.float32,[None,784])
# #
# # w = tf.Variable(tf.zeros([784,10]))
# # b = tf.Variable(tf.zeros([10]))
# #
# # y = tf.nn.softmax(tf.matmul(x,w) + b)
# # y_ = tf.placeholder("float",[None,10])
# #
# # cross_entroy = -tf.reduce_sum(y_ * tf.log(y))
# #
# # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entroy)
# #
# # init = tf.initialize_all_variables()
# #
# #
# # sess = tf.Session()
# # sess.run(init)
# #
# # for i in range(1000):
# #     batch_xs,batch_ys = mnist.train.next_batch(100)
# #     sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})
# #
# #
# # correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
# # accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
# #
# # print sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})





import numpy as np


a = np.linspace(-1,1,5)[:,np.newaxis]
print a
noise = np.random.normal(0,0.5,a.shape)
print noise
b = np.square(a) + 1 + noise
print b