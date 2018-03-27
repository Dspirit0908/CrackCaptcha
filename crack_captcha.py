# coding: utf-8

from utils import *
from setting import *
from PIL import Image
from pylab import array
import tensorflow as tf


class CrackCaptcha(object):

    BASE_MODEL = tf.train.get_checkpoint_state(os.getcwd()).model_checkpoint_path

    def __init__(self, train_dir=r'./train', test_dir=r'./test', logdir='./logs',
                 max_captcha=4, image_height=60, image_width=160):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.logdir = logdir
        self.max_captcha = max_captcha
        self.image_height = image_height
        self.image_width = image_width
        self.keep_prob = tf.placeholder(tf.float32)  # dropout
        with tf.name_scope('Input'):
            self.X = tf.placeholder(tf.float32, [None, self.image_height * self.image_width], name='X_INPUT')
            self.Y = tf.placeholder(tf.float32, [None, self.max_captcha * CHAR_SET_LEN], name='Y_INPUT')

    # Convolutional Neural Networks
    def convolutional_neural_networks(self, w_alpha=0.01, b_alpha=0.1):
        with tf.name_scope('Input_reshape'):
            x = tf.reshape(self.X, shape=[-1, self.image_height, self.image_width, 1])
            image_op = tf.summary.image('input', x, 9)

        # 3 conv layer
        with tf.name_scope('Conv1'):
            w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
            w_c1_his = tf.summary.histogram('w_c1', w_c1)
            b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
            conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
            conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv1 = tf.nn.dropout(conv1, self.keep_prob)

        with tf.name_scope('Conv2'):
            w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
            w_c2_his = tf.summary.histogram('w_c2', w_c2)
            b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
            conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
            conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv2 = tf.nn.dropout(conv2, self.keep_prob)

        with tf.name_scope('Conv3'):
            w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
            w_c3_his = tf.summary.histogram('w_c3', w_c3)
            b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
            conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
            conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            conv3 = tf.nn.dropout(conv3, self.keep_prob)

        # Fully connected layer
        with tf.name_scope('FC'):
            w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))  # need to modify
            w_d_his = tf.summary.histogram('w_d', w_d)
            b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
            dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
            dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
            dense = tf.nn.dropout(dense, self.keep_prob)

        with tf.name_scope('Out'):
            w_out = tf.Variable(w_alpha * tf.random_normal([1024, self.max_captcha * CHAR_SET_LEN]))
            w_out_his = tf.summary.histogram('w_out', w_out)
            b_out = tf.Variable(b_alpha * tf.random_normal([self.max_captcha * CHAR_SET_LEN]))
            output = tf.add(tf.matmul(dense, w_out), b_out)

        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=self.Y))  # need to modify
            loss_scalar = tf.summary.scalar('loss', loss)

        with tf.name_scope('Accuracy'):
            predict = tf.reshape(output, [-1, self.max_captcha, CHAR_SET_LEN])
            max_idx_p = tf.argmax(predict, 2)
            max_idx_l = tf.argmax(tf.reshape(self.Y, [-1, self.max_captcha, CHAR_SET_LEN]), 2)
            correct_pred = tf.equal(max_idx_p, max_idx_l)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            accuracy_scalar = tf.summary.scalar('accuracy', accuracy)

        merge_op = tf.summary.merge([w_c1_his, w_c2_his, w_c3_his, w_d_his, w_out_his, image_op, accuracy_scalar])
        return loss, accuracy, loss_scalar, merge_op, output

    # train model
    def train_model(self, gen_captcha=True, max_step=20000, learning_rate=0.001, acc_record=None, save_prefix='base_model'):
        """
        :param gen_captcha:
        :param max_step:
        :param learning_rate:
        :param acc_record: None or a list
        :param save_prefix:
        :return:
        """
        if acc_record is None:
            acc_record = [0.8]

        loss, accuracy, loss_scalar, merge_op, _ = self.convolutional_neural_networks()
        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(self.logdir, sess.graph)
            step, flag, index = 0, 0, 0
            for i in range(max_step):
                if gen_captcha is True:
                    batch_x, batch_y = get_next_batch(self.max_captcha, self.image_height, self.image_width)
                else:
                    batch_x, batch_y = get_next_image(self.train_dir, self.max_captcha, self.image_height, self.image_width)
                loss_s, _, loss_ = sess.run([loss_scalar, optimizer, loss], feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 0.5})
                summary_writer.add_summary(loss_s, step)
                print("After %d training step(s), loss on training batch is %g." % (step, loss_))
                step += 1

                # 每100 step计算一次准确率
                if step % 100 == 0:
                    if gen_captcha is True:
                        batch_x_test, batch_y_test = get_next_batch(self.max_captcha, self.image_height, self.image_width)
                    else:
                        batch_x_test, batch_y_test = get_next_image(self.test_dir, self.max_captcha, self.image_height, self.image_width)
                    merge, acc = sess.run([merge_op, accuracy], feed_dict={self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1.})
                    summary_writer.add_summary(merge, step)
                    print("After %d training step(s), accuracy on test batch is %g." % (step, acc))

                    # save model
                    if acc > acc_record[index] and flag == index:
                        saver.save(sess, os.path.join(os.getcwd(), save_prefix + '_' + str(acc_record[index]) + '.model'), global_step=step)
                        flag += 1
                        if index < len(acc_record) - 1:
                            index += 1

            saver.save(sess, os.path.join(os.getcwd(), save_prefix + '_last.model'), global_step=step)

    def fine_tuning_model(self, max_step=12000, base_model=BASE_MODEL, max_restore_var=4, learning_rate_for_restore=0.0001,
                        learning_rate_for_train=0.001, acc_record=None, save_prefix='fine_tuning_model'):
        if acc_record is None:
            acc_record = [0.8]

        loss, accuracy, loss_scalar, merge_op, _ = self.convolutional_neural_networks()

        all_vars = tf.trainable_variables()
        var_to_restore = []
        var_to_train = []
        for v in all_vars[0:max_restore_var]:
            var_to_restore.append(v)
        for v in all_vars[max_restore_var:]:
            var_to_train.append(v)

        with tf.name_scope('Optimizer'):
            if len(var_to_restore) == 0:
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_for_train).minimize(loss, var_list=var_to_train)
            elif len(var_to_train) == 0:
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_for_restore).minimize(loss, var_list=var_to_restore)
            else:
                optimizer_1 = tf.train.AdamOptimizer(learning_rate=learning_rate_for_restore).minimize(loss, var_list=var_to_restore)
                optimizer_2 = tf.train.AdamOptimizer(learning_rate=learning_rate_for_train).minimize(loss, var_list=var_to_train)
                optimizer = tf.group(optimizer_1, optimizer_2)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(self.logdir, sess.graph)

            tf.train.Saver(all_vars).restore(sess, base_model)  # restore model
            base_step = int(base_model.split('/')[-1].split('-')[-1])
            print("Base model has been trained %d step(s)." % base_step)

            step, flag, index = 0, 0, 0
            for i in range(max_step):
                batch_x, batch_y = get_next_image(self.train_dir, self.max_captcha, self.image_height, self.image_width)
                loss_s, _, loss_ = sess.run([loss_scalar, optimizer, loss], feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 0.5})
                summary_writer.add_summary(loss_s, step)
                print("After %d training step(s), loss on training batch is %g." % (step, loss_))
                step += 1

                # 每100 step计算一次准确率
                if step % 100 == 0:
                    batch_x_test, batch_y_test = get_next_image(self.test_dir, self.max_captcha, self.image_height, self.image_width)
                    merge, acc = sess.run([merge_op, accuracy], feed_dict={self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1.})
                    summary_writer.add_summary(merge, step)
                    print("After %d training step(s), accuracy on test batch is %g." % (step, acc))

                    # save model
                    if acc > acc_record[index] and flag == index:
                        saver.save(sess, os.path.join(os.getcwd(), save_prefix + '_' + str(acc_record[index]) + '.model'), global_step=step)
                        flag += 1
                        if index < len(acc_record) - 1:
                            index += 1

            saver.save(sess, os.path.join(os.getcwd(), save_prefix + '_last.model'), global_step=step)

    def test_model(self, model=BASE_MODEL, test_num=None, result_dir=None):
        output = self.convolutional_neural_networks()[-1]

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model)
            step = int(model.split('/')[-1].split('-')[-1])
            print("Model has been trained %d step(s)." % step)

            true = []
            pred = []

            images = []
            image_list = os.listdir(self.test_dir)
            random.shuffle(image_list)
            if test_num is None:
                test_num = len(image_list)
            for files in image_list[0:test_num]:
                true.append(files)
                image_file = os.path.join(self.test_dir, files)
                image = array(Image.open(image_file).resize((self.image_width, self.image_height)))
                image_gray = convert2gray(image)
                images.append(image_gray.flatten() / 255)
            predict = tf.argmax(tf.reshape(output, [-1, self.max_captcha, CHAR_SET_LEN]), 2)
            text_list = sess.run(predict, feed_dict={self.X: images, self.keep_prob: 1})

            for text in text_list:
                vector = np.zeros(self.max_captcha * CHAR_SET_LEN)
                i = 0
                for n in text:
                    vector[i * CHAR_SET_LEN + n] = 1
                    i += 1
                pred.append(vec2text(vector))

            count = 0
            for i in range(len(true)):
                if true[i][0:self.max_captcha] == pred[i]:
                    count += 1
                    print(true[i], pred[i])
            print("Total Images: %d. Correct Images: %d.\nAccuracy is %g." % (test_num, count, count / test_num))

            if result_dir is None:
                result_dir = './result.csv'
            if os.path.exists(result_dir):
                os.remove(result_dir)
            with open(result_dir, 'a+', encoding='utf-8') as file:
                for i in range(len(true)):
                    file.write(true[i][0:self.max_captcha] + ',' + pred[i] + '\n')


if __name__ == '__main__':
    CrackCaptcha = CrackCaptcha(test_dir=r'./test_2')
    print("Image Heigth: %d\nImage Width: %d" % (CrackCaptcha.image_height, CrackCaptcha.image_width))
    print("Max length of Captcha: %d" % CrackCaptcha.max_captcha)
    CrackCaptcha.test_model()
    # train_model(gen_captcha=False, save_prefix='captcha2_5num_train')
    # fine_tuning_model(base_model='./base_model_gray_0.8.model-10000', max_step=20000, max_restore_var=4, save_prefix='2400_4restore')
    # fine_tuning_model(base_model='./base_model_gray_0.8.model-10000', max_step=20000, max_restore_var=6, save_prefix='2400_6restore')
    # fine_tuning_model(base_model='./base_model_gray_0.8.model-10000', max_step=20000, max_restore_var=8, save_prefix='2400_8restore')
    # test_model()
    # fine_tuning(12000)

    # text, image = gen_captcha_text_and_image()
    # predict_text = crack_captcha(convert2gray(image).flatten() / 255)
    # print("正确: {}  预测: {}".format(text, predict_text))
    # f = plt.figure()
    # ax = f.add_subplot(111)
    # ax.text(0.1, 0.9, predict_text, ha='center', va='center', transform=ax.transAxes)
    # plt.imshow(convert2gray(image))
    # plt.show()

    # img = array(Image.open(r'D:\CaptchaVariLength-master\data\images\four_digit\\' + '0j4n.png').resize((160, 60)))
    # plt.imshow(convert2gray(img))
    # plt.show()
    # predict_text = crack_captcha(convert2gray(img).flatten() / 255)
    # print(predict_text)

    # g = get_next_image(32)
    # for i in range(2):
    #     batch_x, batch_y = next(g)

    # train_more(15000)
