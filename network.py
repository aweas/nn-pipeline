import datetime
import tensorflow as tf
import numpy as np
import tqdm
import pandas as pd
from scipy.signal import spectrogram
from scipy.io import wavfile
from skimage.transform import resize


def standardize(x):
    return (x - np.mean(x)) / np.std(x)


class AbstractNetwork:
    def _inference(self):
        raise NotImplementedError("Must create a deriving class with own architecture implementation")

    def __init__(self, input_shape, classes_num):
        self.input = None
        self.inference = None
        self.sess = None
        self.dropout_prob = None
        self.logits = None

        self.saver = None
        self.train_writer = None

        self.mode = None

        self.input_shape = input_shape

        self.classes_num = classes_num

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def _input_parser(self, filename, label):
        """ Load images from provided paths
        Beginning of training pipeline
        """

        one_hot = tf.one_hot(label, self.classes_num, dtype=np.uint8)

        result = self.__read_image(filename)
        result = tf.cast(result, tf.float32)
        result = tf.divide(result, tf.constant(255, dtype=tf.float32))

        return result, one_hot

    def __read_image(self, filename):
        result = tf.read_file(filename)
        result = tf.image.decode_bmp(result)
        result = tf.image.resize_images(result, (self.input_shape[0], self.input_shape[1]))
        return result

    def set_training_data(self, *args):
        """ Set training and validation data. """
        if len(args) == 4:
            self.X_train = args[0]
            self.y_train = args[1]
            self.X_test = args[2]
            self.y_test = args[3]
        elif len(args) == 2:
            self.X_train = args[0]
            self.y_train = args[1]
        else:
            raise AttributeError("Invalid number of parameters passed")

    def reset(self):
        """ Reset session and graph allowing for clean training from scratch """

        tf.reset_default_graph()
        self.sess = tf.Session()
        self.input = tf.placeholder(tf.float32, shape=(None, *self.input_shape), name='features')
        self.dropout_prob = tf.placeholder_with_default(1.0, shape=())

    def training(self, epochs=100, batch_size=64, iter_before_validation=10, log_training=False):
        """ Train network with provided data """

        y_input, loss, optimize = self._prepare_for_training(batch_size, log_training)

        for i in range(epochs):
            ls = 0
            training_answers = []

            real_answers = []
            self.sess.run(self.training_init_op)
            if self.X_test is not None:
                self.sess.run(self.test_init_op)

            iterations = int(np.ceil(len(self.X_train) / batch_size))

            for _ in tqdm.trange(iterations):
                X_train, y_train = self.sess.run(self.next_training_batch)
                X_train = np.reshape(X_train, (batch_size, *self.input_shape))

                ls_temp, __, answ = self.sess.run([loss, optimize, self.logits],
                                                  feed_dict={self.input: X_train,
                                                             y_input: y_train,
                                                             self.dropout_prob: 0.2})
                ls += ls_temp / iterations
                training_answers.extend(np.argmax(answ, axis=1))
                real_answers.extend(np.argmax(y_train, axis=1))

            if i % iter_before_validation == 0:
                training_answers = np.asarray(training_answers)
                real_answers = np.asarray(real_answers)               
                print(f'Epoch {i+1}:')
                print(f'Training loss: {ls}')
                print('Training metrics:')
                self.show_metrics(real_answers, training_answers)
                if self.X_test is not None:
                    # acc = len(real_answers[real_answers == training_answers]) / len(real_answers)
                    # print(f'Training accuracy: {int(acc*100)}%')
                    print('Test metrics:')
                    test_answers, real_test_answers = self._get_test_results(batch_size)
                    self.show_metrics(real_test_answers, test_answers)                    

    def show_metrics(self, real_answers, training_answers):
        acc = len(real_answers[real_answers == training_answers]) / len(real_answers)
        print(f'Accuracy: {int(acc*100)}%')

        print("No. \t  How many selected items were relevant \t How many relevant items were selected")
        for i in list(set(real_answers)):
            t_p = real_answers[np.logical_and(real_answers == training_answers, training_answers==i)]
            f_p = real_answers[np.logical_and(real_answers != training_answers, training_answers==i)]
            f_n = real_answers[np.logical_and(real_answers != training_answers, real_answers==i)]
            if len(t_p)+len(f_p) != 0:
                precision = len(t_p) / (len(t_p)+len(f_p))
            else:
                precision = 0
            recall = len(t_p)/(len(t_p) + len(f_n))
            if recall+precision != 0:
                f1 = 2*precision*recall/(precision+recall)
            else:
                f1 = 0
            print(f"Class: {i}: \t Precision: {int(precision*100)}% ({len(t_p)}/{(len(t_p)+len(f_p))}) ", end='')
            if(precision == 0):
                print('\t', end='')
            print(f"\t\t\t Recall: {int(recall*100)}% ({len(t_p)}/{len(t_p) + len(f_n)})", end='')
            print(f"\t\t F1: {int(f1*100)}%")
        print()

    def _prepare_for_training(self, batch_size, log_training):
        """ Prepare all necessary variables and fields """

        self.reset()

        if self.X_train is None:
            raise TypeError("Training data cannot be empty!")

        y_input = tf.placeholder(tf.float32, shape=(None, self.classes_num))

        with tf.variable_scope('cnn'):
            self.inference = self._inference()

        self.logits = self.inference

        with tf.name_scope('training'):
            loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y_input, logits=self.logits))
            optimize = tf.train.AdamOptimizer().minimize(loss)

        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        self.saver = tf.train.Saver()

        if log_training:
            self.train_writer = tf.summary.FileWriter('logs/', self.sess.graph)

        # I MIGHT clean this up someday, but this day has yet to come
        # Prepare tensors and dataset for training data
        self.y_train_tensor = tf.constant(self.y_train)
        self.X_train_tensor = tf.constant(self.X_train)

        self.tr_data = tf.data.Dataset.from_tensor_slices((self.X_train_tensor, self.y_train_tensor))
        self.tr_data = self.tr_data.map(self._input_parser, num_parallel_calls=4)
        self.tr_data = self.tr_data.batch(batch_size)
        self.tr_data = self.tr_data.prefetch(buffer_size=2 * batch_size)

        self.training_iterator = tf.data.Iterator.from_structure(self.tr_data.output_types,
                                                                 self.tr_data.output_shapes)
        self.next_training_batch = self.training_iterator.get_next()
        self.training_init_op = self.training_iterator.make_initializer(self.tr_data)

        if self.X_test is not None:
            # Prepare tensors and dataset for test data
            self.y_test_tensor = tf.constant(self.y_test)
            self.X_test_tensor = tf.constant(self.X_test)

            self.val_data = tf.data.Dataset.from_tensor_slices((self.X_test_tensor, self.y_test_tensor))
            self.val_data = self.val_data.map(self._input_parser, num_parallel_calls=4)
            self.val_data = self.val_data.batch(batch_size)
            self.val_data = self.val_data.prefetch(buffer_size=2 * batch_size)

            self.test_iterator = tf.data.Iterator.from_structure(self.val_data.output_types,
                                                                 self.val_data.output_shapes)
            self.next_test_batch = self.test_iterator.get_next()
            self.test_init_op = self.test_iterator.make_initializer(self.val_data)

        return y_input, loss, optimize

    def _get_test_results(self, batch_size):
        """ Calculate current network accuracy for provided validation set """

        res = []
        real_res = []
        for j in range(int(len(self.X_test) / batch_size)):
            X_test, y_test = self.sess.run(self.next_test_batch)
            res.extend(np.argmax(self.predict(X_test), axis=1))
            real_res.extend(np.argmax(y_test, axis=1))

        res = np.asarray(res)
        real_res = np.asarray(real_res)
        return res, real_res

    def predict(self, img):
        """ Predict output with basic inference """

        return self.sess.run(tf.nn.softmax(self.logits), feed_dict={self.input: img})

    def freeze_model(self, location):
        """ Prepare ProtoBuffer file containing graph definition and all variables necessary for inference (and nothing more)
        This method allows to cut down size by up to 80% comparing to full metagraph.
        """

        name = f"{location}_{datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')}.pb"

        inference_nodes = tf.graph_util.remove_training_nodes(tf.get_default_graph().as_graph_def())

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            self.sess,
            inference_nodes,
            ['output']
        )

        with tf.gfile.GFile(name, "wb") as f:
            f.write(output_graph_def.SerializeToString())

    def show_summary(self):
        # Count total number of parameters
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
            print(variable.name, f'\t{variable_parameters} params\t{shape}')
        print(f'\nTotal: {total_parameters} params')