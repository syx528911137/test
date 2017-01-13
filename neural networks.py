# import tensorflow as tf
# import numpy as np
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
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# x_data = np.linspace(-1,1,300)[:,np.newaxis]
# noise = np.random.normal(0,0.05,x_data.shape)
# y_data = np.square(x_data) + 0.5 + noise
#
#
# xs = tf.placeholder(tf.float32,[None,1])
# ys = tf.placeholder(tf.float32,[None,1])
#
#
# layer1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
# predict = add_layer(layer1,10,1,activation_function=tf.nn.relu)
#
#
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(predict - ys),reduction_indices=[1]))
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
# init = tf.initialize_all_variables()
#
# saver = tf.train.Saver()
#
#
# sess = tf.Session()
# sess.run(init)
#
#
# for step in range(1000):
#     sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
#     if step % 20 == 0:
#         print sess.run(loss,feed_dict={xs:x_data,ys:y_data})
#
#
#
# save_path = saver.save(sess,"test_save.ckpt")

import tensorflow as tf
import numpy as np

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
biases = 0.5
y_data = np.square(x_data) + biases + noise


xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])


#layer 1
Weights_layer1 = tf.Variable(tf.random_normal([1,10]),name="Weights_layer1")
biases_layer1 = tf.Variable(tf.zeros([1,10]) + 0.1,name="biases_layer1")
output_layer1 = tf.nn.relu( tf.matmul(xs,Weights_layer1) + biases_layer1)

#layer 2

Weights_layer2 = tf.Variable(tf.random_normal([10,1]),name="Weights_layer2")
biases_layer2 = tf.Variable(tf.zeros([1,1]) + 0.1,name="biases_layer2")
output_layer2 = tf.nn.relu(tf.matmul(output_layer1,Weights_layer2) + biases_layer2)


loss = tf.reduce_mean(tf.reduce_sum(tf.square(output_layer2 - ys),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

init = tf.initialize_all_variables()
# saver = tf.train.Saver()

saver = tf.train.Saver()




sess = tf.Session()
sess.run(init)


for step in range(0,2000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})

    if step % 100 == 0:
        print sess.run(loss,feed_dict={xs:x_data,ys:y_data})



savePath = saver.save(sess,"testnn.ckpt")