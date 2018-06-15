from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import scipy.io as sio

import pickle
from time import sleep
import matplotlib.pyplot as plt
from random import shuffle
import cloudstorage as gcs;

sess = tf.InteractiveSession()

# dataFile = '17MachineHourlyPeak7DaysInput'
filename = 'gs://polynomial-text-128814-ml/17MachineHourlyPeak7DaysInput-traintest.mat'
with gcs.open(filename, 'r') as gcs_file:
    Data=sio.loadmat(gcs_file)
# Data=sio.loadmat(open( "/Users/lori/Downloads/NNInput/"+dataFile+"-traintest.mat", "rb" ))
data = Data['trainInput']
label = Data['trainLabel']

# testData=sio.loadmat(open( "/Users/lori/Downloads/NNInput/"+dataFile+"-test.mat", "rb" ))
testdata = Data['testInput']
testlabel = Data['testLabel']


# feature1 = pickle.load(open("/Users/lori/Downloads/NNInput/Weights/cpu_hourly_fullyconnected_7days_400_100_50_10_new.pkl", "rb" ))

def randomize(data,change):
    randomData = data+change*np.random.choice([-1, 1], size=[data.shape[0], data.shape[1]])
    return randomData


# Parametersimport input_data
learning_rate = 0.0001
#training_epochs = 15
#batch_size = 100


# Network Parameters
n_hidden_1 = 200 # 1st layer number of features
n_hidden_2 = 30
n_hidden_3 = 10 # 1st layer number of features
n_hidden_4 = 10
# n_hidden_3 = 10# 2nd layer number of features
n_input = data.shape[1]
n_classes = label.shape[1]
data_size=data.shape[0]
# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

#
# def layer(x, input_size, output_size):
#     w = tf.Variable(tf.truncated_normal(shape=[input_size, output_size]))
#     b = tf.Variable(tf.zeros([output_size]))
#     return tf.sigmoid(tf.matmul(x,w) + b)
#
# num_layers = ... #read somewhere
# size_layers = ... #output_dimesion for each layer
# h = x
# for i in range(num_layers):
#     h = layer(h, input_size, size_layers[i])
#     input_size = size_layer[i]



# Create model
def multilayer_perceptron(x, weights, biases,keep_prob):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # layer_1_drop = tf.nn.dropout(layer_1, keep_prob)* keep_prob
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2_drop = tf.nn.dropout(layer_2, keep_prob)* keep_prob

    # layer_3 = tf.add(tf.matmul(layer_2_drop, weights['h3']), biases['b3'])
    # layer_3 = tf.nn.relu(layer_3)
    # layer_3_drop = tf.nn.dropout(layer_3, keep_prob)* keep_prob

    # layer_4 = tf.add(tf.matmul(layer_3_drop, weights['h4']), biases['b4'])
    # layer_4 = tf.nn.relu(layer_4)
    #
    # layer_4_drop = tf.nn.dropout(layer_4, keep_prob)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2_drop, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias

# with tf.name_scope('weights'):
weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], mean=0, stddev=1 / n_hidden_1)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], mean=0, stddev=1 / n_hidden_2)),
       # 'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3],mean=0, stddev=1 / n_hidden_3)),

        # 'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], mean=0, stddev=1 / n_hidden_4)),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], mean=0, stddev=1 / n_classes))
    }
# tf.summary.scalar('output_weights', weights['out'])
biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1], mean=0, stddev=1 / n_hidden_1)),
        'b2': tf.Variable(tf.random_normal([n_hidden_2], mean=0, stddev=1 / n_hidden_2)),
        # 'b3': tf.Variable(tf.random_normal([n_hidden_3], mean=0, stddev=1 / n_hidden_3)),
        # 'b4': tf.Variable(tf.random_normal([n_hidden_4], mean=0, stddev=1 / n_hidden_4)),

        'out': tf.Variable(tf.random_normal([n_classes], mean=0, stddev=1 / n_classes))
    }
# tf.summary.scalar('output_biases', biases['out'])





pred = multilayer_perceptron(x, weights, biases,keep_prob)

# simple_pred = tf.reduce_mean(x, 1)
# Define loss and optimizer

cost_all=tf.nn.softmax_cross_entropy_with_logits(logits =pred, labels=y)


with tf.name_scope('cross_entropy'):
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits =pred, labels=y))

tf.summary.scalar('cross_entropy', cost)


simple_pred = Data['simpleProbAll']
Sp_op = 1-simple_pred[0]
Sp_one = np.column_stack((simple_pred[0],Sp_op))
spp = tf.placeholder("float", [None, n_classes])
simple_cost = tf.nn.softmax_cross_entropy_with_logits(logits =spp, labels=y)


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
maxepoch=10
display_step = 10
train_step=10;

merged = tf.summary.merge_all()

# tf.global_variables_initializer().run()

a1=[];
a2=[];
c1=[];

with tf.Session() as sess:
    # assign_op1 = weights['h1'].assign(feature1['w']['h1'])
    # assign_op2 = weights['h2'].assign(feature1['w']['h2'])
    # assign_op3 = weights['out'].assign(feature1['w']['out'])
    #
    # assign_op4 = biases['b1'].assign(feature1['b']['b1'])
    # assign_op5 = biases['b2'].assign(feature1['b']['b2'])
    # assign_op6 = biases['out'].assign(feature1['b']['out'])
    #
    # assign_op7 = biases['b3'].assign(feature1['b']['b3'])
    # assign_op8 = biases['b4'].assign(feature1['b']['b4'])
    # assign_op9 = weights['h3'].assign(feature1['w']['h3'])
    # assign_op10 = weights['h4'].assign(feature1['w']['h4'])

    ####random Initializer


    init = tf.initialize_all_variables()

    sess.run(init)
    train_writer = tf.summary.FileWriter('/Users/lori/Downloads/NNInput/Tensorboard' + '/train',
                                         sess.graph)
    # #
    # sess.run(assign_op1)
    # sess.run(assign_op2)
    # sess.run(assign_op3)
    # sess.run(assign_op4)
    # sess.run(assign_op5)
    # sess.run(assign_op6)
    # sess.run(assign_op7)
    # sess.run(assign_op8)
    # sess.run(assign_op9)
    # sess.run(assign_op10)
    # Training cycle

    avg_cost = 0.
        #total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
    for epoch in range(maxepoch):
            #batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
        if epoch % train_step == 0:
            randomData = randomize(data,0.1)

        summary,_, ca,c,w,b,acc,cp,fp = sess.run([merged,optimizer, cost_all,cost,weights,biases,accuracy,correct_prediction,pred], feed_dict={x: data,
                                                          y: label,keep_prob : 0.5})
        a1.append(sess.run(accuracy, feed_dict={x: data, y: label, keep_prob: 1}))
        a2.append(sess.run(accuracy, feed_dict={x: testdata, y: testlabel, keep_prob: 1}))
        c1.append(sess.run(cost, feed_dict={x: testdata, y: testlabel, keep_prob: 1}))

            # Compute average loss
        train_writer.add_summary(summary, epoch)

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(c),"accuracy","{:.3f}".format(acc),"originalData_accuracy","{:.3f}".format(a1[-1]),"testData_accuracy","{:.3f}".format(a2[-1]))
    print("Optimization Finished!")


    # print("Accuracy:", accuracy.eval({x: data, y: label, keep_prob: 1}))
    # # sc = simple_cost.eval({x: data,y: label,spp:Sp_one,keep_prob : 1})
    # # Test model
    # print("Cost:", cost.eval({x: testdata, y: testlabel, keep_prob: 1}))
    # cp_test=correct_prediction.eval({x: testdata, y: testlabel, keep_prob: 1})
    # ca_test=cost_all.eval({x: testdata, y: testlabel, keep_prob: 1})

    # print("Accuracy:", accuracy.eval({x: testdata, y: testlabel,keep_prob : 1}))

# output=open('/Users/lori/Downloads/NNInput/Weights/cpu_hourly_fullyconnected_7days_400_100_50_10_new.pkl','wb')
# f1={'w':w,'b':b};
# pickle.dump(f1, output)
# output.close()
#
# sio.savemat('/Users/lori/Downloads/17MachineResult_cost_test.mat', {'cp': cp,'fp':fp,'cost':ca,'cp_test':cp_test,'ca_test':ca_test})
