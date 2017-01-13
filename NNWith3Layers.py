import tensorflow as tf
import numpy as np


class NNwith3layers:

    def __init__(self,sess,traindatascale,testdatascale):
        self.sess = sess
        self.xs = tf.placeholder(tf.float32,[None,traindatascale])
        self.ys = tf.placeholder(tf.float32,[None,testdatascale])


    def trainModel(self,traindata,testdata,iter_num):
        Weigths_layer1 = tf.Variable(tf.random_normal([1,10]),name="weights_layer1")
        bias_layer1 = tf.Variable(tf.zeros([1,10]) + 0.1,name="bias_layer1")
        output_layer1 = tf.nn.relu(tf.matmul(self.xs,Weigths_layer1) + bias_layer1)

        Weigths_layer2 = tf.Variable(tf.random_normal([10,1]),name="weight_layer2")
        bias_layer2 = tf.Variable(tf.random_normal([1,1]) + 0.1,name="bias_layer2")
        output_layer2 = tf.nn.relu(tf.matmul(output_layer1,Weigths_layer2) + bias_layer2)

        loss = tf.reduce_mean( tf.reduce_sum(tf.square(output_layer2 - self.ys),reduction_indices=[1]))

        trainstep = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        init = tf.initialize_all_variables()
        self.sess.run(init)

        for i in range(0,iter_num):
            self.sess.run(trainstep,feed_dict={self.xs:traindata,self.ys:testdata})
            if i % 100 == 0:
                print self.sess.run(loss,feed_dict={self.xs:traindata,self.ys:testdata})


    def saveModel(self,path):
        saver = tf.train.Saver()
        saver.save(self.sess,path)
        pass

    def loadModel(self,path):
        saver = tf.train.Saver()
        saver.restore(self.sess,path)
    def testcase(self,input,path):
        Weigths_layer1 = tf.Variable(tf.random_normal([1, 10]), name="weights_layer1")
        bias_layer1 = tf.Variable(tf.zeros([1, 10]) + 0.1, name="bias_layer1")
        output_layer1 = tf.nn.relu(tf.matmul(self.xs, Weigths_layer1) + bias_layer1)

        Weigths_layer2 = tf.Variable(tf.random_normal([10, 1]), name="weight_layer2")
        bias_layer2 = tf.Variable(tf.random_normal([1, 1]) + 0.1, name="bias_layer2")
        output_layer2 = tf.nn.relu(tf.matmul(output_layer1, Weigths_layer2) + bias_layer2)
        self.loadModel(path)
        return self.sess.run(output_layer2,feed_dict={self.xs:input})


if __name__ == '__main__':
    x_data = np.linspace(-1,1,300)[:,np.newaxis]
    noise = np.random.normal(0,0.05,x_data.shape)
    bise = 0.5
    y_data = np.square(x_data) + bise + noise

    # sess = tf.Session()
    # model = NNwith3layers(sess,1,1)
    # # model.trainModel(x_data,y_data,2000)
    # # model.saveModel("nnw3.ckpt")
    # # model.loadModel("nnw3.ckpt")
    # print model.testcase([[0.6],[0.7],[0.8]],"nnw3.ckpt")
    xs = tf.placeholder(tf.float32,[None,1])
    ys = tf.placeholder(tf.float32,[None,1])
    Weigths_layer1 = tf.Variable(tf.random_normal([1, 10]), name="weights_layer1")
    bias_layer1 = tf.Variable(tf.zeros([1, 10]) + 0.1, name="bias_layer1")
    output_layer1 = tf.nn.relu(tf.matmul(xs, Weigths_layer1) + bias_layer1)

    Weigths_layer2 = tf.Variable(tf.random_normal([10, 1]), name="weight_layer2")
    bias_layer2 = tf.Variable(tf.random_normal([1, 1]) + 0.1, name="bias_layer2")
    output_layer2 = tf.nn.relu(tf.matmul(output_layer1, Weigths_layer2) + bias_layer2)

    # Weigths_layer3 = tf.Variable(tf.random_normal([6,2]),name="w3")
    # bias_layer3 = tf.Variable(tf.random_normal([1,2]) + 0.1,name="b3")
    # output_layer3 = tf.nn.relu(tf.matmul(output_layer2,Weigths_layer3) + bias_layer3)
    #
    # Weigths_layer4 = tf.Variable(tf.random_normal([2,1]),name="w4")
    # bias_layer4 = tf.Variable(tf.random_normal([1,1]) + 0.1,name="b4")
    # output_layer4 = tf.nn.relu( tf.matmul(output_layer3,Weigths_layer4) + bias_layer4 )
    #
    # varlist = []
    # varlist.append(Weigths_layer4)
    # varlist.append(Weigths_layer3)
    # varlist.append(bias_layer3)
    # varlist.append(bias_layer4)

    loss = tf.reduce_mean( tf.reduce_sum(tf.square(output_layer2 - ys),reduction_indices=[1]))

    trainstep = tf.train.GradientDescentOptimizer(0.3).minimize(loss)

    sess = tf.Session()
    # saver = tf.train.Saver({'weights_layer1':Weigths_layer1,'bias_layer1':bias_layer1,'weight_layer2':Weigths_layer2,'bias_layer2':bias_layer2})
    # saver.restore(sess,"nnw3.ckpt")

    # init = tf.initialize_variables(varlist)
    init = tf.initialize_all_variables()
    sess.run(init)

    for step in range(0, 2000):
        sess.run(trainstep, feed_dict={xs: x_data, ys: y_data})

        if step % 100 == 0:
            print sess.run(loss, feed_dict={xs: x_data, ys: y_data})

