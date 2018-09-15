# Import libraries
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Load MNIST dataset
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


class feedForwardNet:
    pass


def declare_placeholders(self):
    """Define placeholders for class"""

    # Placeholder for Inputs
    self.input_batch = tf.placeholder(dtype=tf.float32, shape=[None, self.num_input], name="input_batch")
    # Placeholder for outputs
    self.output_batch = tf.placeholder(dtype=tf.float32, shape=[None, self.num_classes], name="output_batch")

    # Placeholder for Learning rate
    self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate")

    # Placeholder for dropouts
    self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name="dropout")


feedForwardNet.__declare_placeholders = classmethod(declare_placeholders)


def build_layers(self):
    # Dictionaries to specify weights and biases
    self.weights = {
        'h1': tf.Variable(tf.random_normal([self.num_input, self.num_hidden_1])),
        'h2': tf.Variable(tf.random_normal([self.num_hidden_1, self.num_hidden_2])),
        'out': tf.Variable(tf.random_normal([self.num_hidden_2, self.num_classes]))
    }
    self.biases = {
        'b1': tf.Variable(tf.random_normal([self.num_hidden_1])),
        'b2': tf.Variable(tf.random_normal([self.num_hidden_2])),
        'out': tf.Variable(tf.random_normal([self.num_classes]))
    }


feedForwardNet.__build_layers = classmethod(build_layers)


def compute_activations(self):
    layer1 = tf.nn.relu(tf.add(tf.matmul(self.input_batch, self.weights['h1']), self.biases['b1']))
    layer1 = tf.nn.dropout(layer1, 1 - self.dropout)
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, self.weights['h2']), self.biases['b2']))
    layer2 = tf.nn.dropout(layer2, 1 - self.dropout)

    self.logits = tf.add(tf.matmul(layer2, self.weights['out']), self.biases['out'])


feedForwardNet.__compute_activations = classmethod(compute_activations)


def compute_loss(self):
    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.output_batch))


feedForwardNet.__compute_loss = classmethod(compute_loss)


def optimize(self):
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


feedForwardNet.__optimize = classmethod(optimize)


def train_on_batch(self, session, x_batch, y_batch, learning_rate, dropout):
    feed_dict = {self.input_batch: x_batch,
                 self.output_batch: y_batch,
                 self.learning_rate: learning_rate,
                 self.dropout: dropout}

    return session.run([self.optimizer, self.loss], feed_dict=feed_dict)


feedForwardNet.train_on_batch = classmethod(train_on_batch)


def predict_for_batch(self, session, x_batch, dropout):
    predictions = session.run(self.logits,
                              feed_dict={self.input_batch: x_batch,
                                         self.dropout: dropout})
    return predictions


feedForwardNet.predict_for_batch = classmethod(predict_for_batch)


def init_model(self, num_input, num_hidden_1, num_hidden_2, num_classes):
    self.num_input = num_input
    self.num_hidden_1 = num_hidden_1
    self.num_hidden_2 = num_hidden_2
    self.num_classes = num_classes

    self.__declare_placeholders()
    self.__build_layers()
    self.__compute_activations()
    self.__compute_loss()
    self.__optimize()


feedForwardNet.__init__ = classmethod(init_model)


def evaluate_model(self, session, x_batch, labels, dropout):
    predictions = self.predict_for_batch(session, x_batch, dropout)
    correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
    return session.run(accuracy, feed_dict={self.input_batch: x_batch, self.dropout: dropout})


# Network Parameters
hidden_1_size = 256
hidden_2_size = 128
input_size = 784
num_classes = 10

# Hyper parameters
lr = 0.01
dropout_prob = 0.2
batch_size = 100
num_epochs = 15
display_step = 1

model = feedForwardNet(input_size, hidden_1_size, hidden_2_size, num_classes)

# Train model
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(num_epochs):
        avg_loss = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for _ in range(total_batch):
            x, y = mnist.train.next_batch(batch_size)
            # Run optimization
            _, c = model.train_on_batch(sess, x, y, lr, dropout_prob)
            avg_loss += c / total_batch

        if epoch % display_step == 0:
            # Model accuracy
            accuracy = evaluate_model(model, sess, x, y, dropout_prob)
            print("Epoch:", epoch + 1, 'Loss:', avg_loss, 'Accuracy:', accuracy)
    print("....Training Finished!")

    # Predict on test set
    x_test = mnist.test.images
    y_test = mnist.test.labels
    test_predictions = model.predict_for_batch(sess, x_test, dropout_prob)

    correct_test_predictions = tf.equal(tf.argmax(test_predictions, 1), tf.argmax(y_test, 1))
    # Calculate accuracy
    test_accuracy = tf.reduce_mean(tf.cast(correct_test_predictions, "float"))
    print("Accuracy:", test_accuracy.eval({model.input_batch: x_test, model.output_batch: y_test}))
