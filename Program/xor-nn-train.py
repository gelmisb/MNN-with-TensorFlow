from __future__ import print_function
import tensorflow as tf
import time
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

NUM_EXAMPLES = 500  # Number of training or evaluation examples
NUM_FEATURES = 2  # Number of input features
NUM_LABELS = 1  # Number of output features or class labels
NUM_HIDDEN = 5  # Number of hidden/middle layer nodes
LEARNING_RATE = 0.1  # Speed of learning
NUM_EPOCHS = 1000  # Present the training data this number of times
OUTPUT_INTERVAL = 100  # Used for output during training

# Path to training or evaluation file
INPUT_FILE = "data/train.csv"
# Path to the network model save file
MODEL_PATH = "data/xor-model.ckpt"

# Array for the input features
trainx = []
# Array for the output labels/features
trainy = []

# Load inputs and labels from disk
# NOTE: assumes 2 inputs followed by 1 label
# NOTE: files assumed to be located in a data directory
with open(INPUT_FILE, 'r') as csvfile:
    input_data = csv.reader(csvfile, delimiter=',')
    for row in input_data:
        trainx.append([float(row[0]), float(row[1])])
        trainy.append([float(row[2])])

print('Training data ', trainx)
print('Label data ', trainy)

# Define the input layer placeholders
x_ = tf.placeholder(tf.float32, shape=[NUM_EXAMPLES, NUM_FEATURES], name='inputs')
# Define the desired/target output placeholders
y_ = tf.placeholder(tf.float32, shape=[NUM_EXAMPLES, NUM_LABELS], name='labels')

# Define weights
Weights1 = tf.Variable(tf.random_uniform([NUM_FEATURES, NUM_HIDDEN], -1.0, 1.0), name="Weights1")
Weights2 = tf.Variable(tf.random_uniform([NUM_HIDDEN, NUM_LABELS], -1.0, 1.0), name="Weights2")

# Define the BIAS node
Bias1 = tf.Variable(tf.zeros([NUM_HIDDEN]), name="Bias1")
Bias2 = tf.Variable(tf.zeros([NUM_LABELS]), name="Bias2")

# Feed forward to the hidden layer
H1 = tf.sigmoid(tf.matmul(x_, Weights1) + Bias1)

# Feedforward to the output layer - Hypothesis is what the 
# neural network thinks it should output for a 
# given input.
Hypothesis = tf.sigmoid(tf.matmul(H1, Weights2) + Bias2)

# Setup the cost function and set the traning method
# We are using the squared error (ACTUAL - DESIRED)
cost = tf.reduce_sum(tf.square(Hypothesis - y_))

# Choose a training approach - effectively backprop
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

# Initialise the variables and create a session
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

# sess = tf.Session()

# Initialise the session
with tf.Session() as sess:
    sess.run(init)

    # Start the training loop
    t_start = time.clock()
    for i in range(NUM_EPOCHS):
        # Run the training
        sess.run(train_step, feed_dict={x_: trainx, y_: trainy})
        avg_error = 0
        # Do some output ever 1000 epochs
        if i % OUTPUT_INTERVAL == 0:
            print('Epoch ', i)
            results = sess.run(Hypothesis, feed_dict={x_: trainx, y_: trainy})
            hp, ct = sess.run([Hypothesis, cost], feed_dict={x_: trainx, y_: trainy})

            print("Hypothesis\tTarget\tError")
            for j in range(len(hp)):
                hyp_error = abs(hp[j] - trainy[j])
                print(hp[j], '\t', trainy[j], '\t', hyp_error)
                avg_error = sum(hyp_error) / float(len(hyp_error))

            print("Cost = ", ct)
            print("Average error ", avg_error)

            answer = tf.equal(tf.floor(Hypothesis + 0.1), y_)
            accuracy = tf.reduce_mean(tf.cast(answer, "float"))

            print(answer)
            print("Accuracy: ", accuracy.eval({x_: trainx, y_: trainy}) * 100, "%")

    t_end = time.clock()
    print('Elapsed time is', t_end - t_start)

    # Save model weights to disk
    save_path = saver.save(sess, MODEL_PATH)
    print("Model saved in file: %s" % save_path)