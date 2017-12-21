import tensorflow as tf 
from tqdm import tqdm
import input_data

keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
image_w = 50
image_h = 50
training_iters =20000
learning_rate = 0.001
batch_size = 16
display_step = 100

dropout = 0.75
tf_logs = './log'
test_dir = './test'
train_dir = './images'
n_classes = 2

x = tf.placeholder(tf.float32, [None, image_w*image_h])
y = tf.placeholder(tf.float32, [None, n_classes])

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, image_w, image_h, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    'wd1': tf.Variable(tf.random_normal([50 * 50 * 64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,
                                                              labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
    cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

summary_op = tf.summary.merge([
    tf.summary.scalar('loss', cost),
    tf.summary.scalar('accuracy', accuracy),
])

init = tf.global_variables_initializer()
# Launch the graph
writer = tf.summary.FileWriter(tf_logs)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    writer.add_graph(sess.graph)

    # Keep training until reach max iterations
    progress = tqdm(range(training_iters))
    train_images,train_labels = input_data.get_file(train_dir)
    for batch_idx in progress:
        batch_x, batch_y = input_data.get_batch(train_images,train_labels,image_w,image_h,10,50)
        # Run optimization op (backprop)
        sess.run(optimizer,
                 feed_dict={x: batch_x.eval(session=sess),
                            y: batch_y.eval(session=sess),
                            keep_prob: dropout})

        if batch_idx % display_step == 0:
            summary, loss, acc = sess.run([summary_op, cost, accuracy], 
                                        feed_dict={x: batch_x.eval(session=sess),
                                                    y: batch_y.eval(session=sess),
                                                     keep_prob: 1.})
            progress.set_description('Loss(val): %4.4f, Accuracy (val): '
                                     '%4.4f' % (loss, acc))
            writer.add_summary(summary, batch_idx)
            saver.save(sess, tf_logs + '/model.ckpt', global_step=batch_idx)

    print "Optimization Finished!"

    # Calculate accuracy for 256 mnist test images
    test_images ,test_labels = input_data.get_file(test_dir)
    print "Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: test_images.eval(session=sess),
                                      y: test_labels.eval(session=sess),
                                      keep_prob: 1.})

writer.close()
