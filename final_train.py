# coding: utf-8

from utils import *
from setting import *
from PIL import Image
from pylab import array
import tensorflow as tf

with tf.name_scope('Input'):
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH], name='x_input')
    Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN], name='y_input')
keep_prob = tf.placeholder(tf.float32)  # dropout


# 定义CNN
def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    with tf.name_scope('Input_reshape'):
        x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        image_op = tf.summary.image('input', x, 9)

    # 3 conv layer
    with tf.name_scope('Conv1'):
        w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
        w_c1_his = tf.summary.histogram('w_c1', w_c1)
        b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, keep_prob)

    with tf.name_scope('Conv2'):
        w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
        w_c2_his = tf.summary.histogram('w_c2', w_c2)
        b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, keep_prob)

    with tf.name_scope('Conv3'):
        w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
        w_c3_his = tf.summary.histogram('w_c3', w_c3)
        b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    with tf.name_scope('FC'):
        w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024]))
        w_d_his = tf.summary.histogram('w_d', w_d)
        b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, keep_prob)

    with tf.name_scope('Out'):
        w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
        w_out_his = tf.summary.histogram('w_out', w_out)
        b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
        out = tf.add(tf.matmul(dense, w_out), b_out)
        # out = tf.nn.softmax(out)
    histogram_op = tf.summary.merge([w_c1_his, w_c2_his, w_c3_his, w_d_his, w_out_his, image_op])
    return out, histogram_op


# 训练
def train_model(gen_captcha=True, max_step=20000, learning_rate=0.001, acc_record=None, save_prefix='base_model'):
    """
    :param gen_captcha:
    :param max_step:
    :param learning_rate:
    :param acc_record: None or a list
    :param save_prefix:
    :return:
    """
    if acc_record is None:
        acc_record = [0.5, 0.8]

    output, histogram_op = crack_captcha_cnn()
    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
        loss_scalar = tf.summary.scalar('loss', loss)

    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    with tf.name_scope('Accuracy'):
        predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        accuracy_scalar = tf.summary.scalar('accuracy', accuracy)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)

        step, flag, index = 0, 0, 0
        for i in range(max_step):
            if gen_captcha is True:
                batch_x, batch_y = get_next_batch()
            else:
                batch_x, batch_y = get_next_image(train_dir)
            loss_s, _, loss_ = sess.run([loss_scalar, optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
            summary_writer.add_summary(loss_s, step)
            print("After %d training step(s), loss on training batch is %g." % (step, loss_))
            step += 1

            # 每100 step计算一次准确率
            if step % 100 == 0:
                if gen_captcha is True:
                    batch_x_test, batch_y_test = get_next_batch()
                else:
                    batch_x_test, batch_y_test = get_next_image(test_dir)
                acc_s, histogram, acc = sess.run([accuracy_scalar, histogram_op, accuracy],
                                                 feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                summary_writer.add_summary(acc_s, step)
                summary_writer.add_summary(histogram, step)
                print("After %d training step(s), accuracy on test batch is %g." % (step, acc))

                # save model
                if acc > acc_record[index] and flag == index:
                    saver.save(sess, os.path.join(os.getcwd(), save_prefix + '_' + str(acc_record[index]) + '.model'),
                               global_step=step)
                    flag += 1
                    if index < len(acc_record) - 1:
                        index += 1

        saver.save(sess, os.path.join(os.getcwd(), save_prefix + '_last.model'), global_step=step)


def fine_tuning_model(max_step=12000, base_model=base_model, max_restore_var=4, learning_rate_for_restore=0.0001,
                    learning_rate_for_train=0.001, acc_record=None, save_prefix='fine_tuning_model'):
    if acc_record is None:
        acc_record = [0.5, 0.8]

    output, histogram_op = crack_captcha_cnn()
    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
        loss_scalar = tf.summary.scalar('loss', loss)

    all_vars = tf.trainable_variables()
    var_to_restore = []
    var_to_train = []
    for v in all_vars[0:max_restore_var]:
        var_to_restore.append(v)
    for v in all_vars[max_restore_var:]:
        var_to_train.append(v)

    with tf.name_scope('Optimizer'):
        optimizer_1 = tf.train.AdamOptimizer(learning_rate=learning_rate_for_restore).minimize(loss, var_list=var_to_restore)
        optimizer_2 = tf.train.AdamOptimizer(learning_rate=learning_rate_for_train).minimize(loss, var_list=var_to_train)
        optimizer = tf.group(optimizer_1, optimizer_2)

    with tf.name_scope('Accuracy'):
        predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        accuracy_scalar = tf.summary.scalar('accuracy', accuracy)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(logdir, sess.graph)

        tf.train.Saver(var_to_restore).restore(sess, base_model)  # 载入保存模型
        base_step = int(base_model.split('/')[-1].split('-')[-1])
        print("基础模型已训练%d步" % base_step)

        step, flag, index = 0, 0, 0
        for i in range(max_step):
            batch_x, batch_y = get_next_image(train_dir)
            loss_s, _, loss_ = sess.run([loss_scalar, optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
            summary_writer.add_summary(loss_s, step)
            print("After %d training step(s), loss on training batch is %g." % (step, loss_))
            step += 1

            # 每50 step计算一次准确率
            if step % 50 == 0:
                batch_x_test, batch_y_test = get_next_image(test_dir)
                acc_s, histogram, acc = sess.run([accuracy_scalar, histogram_op, accuracy], feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                summary_writer.add_summary(acc_s, step)
                summary_writer.add_summary(histogram, step)
                print("After %d training step(s), accuracy on test batch is %g." % (step, acc))

                # save model
                if acc > acc_record[index] and flag == index:
                    saver.save(sess, os.path.join(os.getcwd(), save_prefix + '_' + str(acc_record[index]) + '.model'),
                               global_step=step)
                    flag += 1
                    if index < len(acc_record) - 1:
                        index += 1

        saver.save(sess, os.path.join(os.getcwd(), save_prefix + '_last.model'), global_step=step)


def test_model(model=base_model, test_num=None, result_dir=None):
    output, _ = crack_captcha_cnn()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model)
        step = int(model.split('/')[-1].split('-')[-1])
        print("模型已训练%d步" % step)

        true = []
        pred = []

        images = []
        image_list = os.listdir(test_dir)
        random.shuffle(image_list)
        if test_num is None:
            test_num = len(image_list)
        for files in image_list[0:test_num]:
            true.append(files)
            image_file = os.path.join(test_dir, files)
            image = array(Image.open(image_file).resize((IMAGE_WIDTH, IMAGE_HEIGHT)))
            image_gray = convert2gray(image)
            images.append(image_gray.flatten() / 255)
        predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: images, keep_prob: 1})

        for text in text_list:
            vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
            i = 0
            for n in text:
                vector[i * CHAR_SET_LEN + n] = 1
                i += 1
            pred.append(vec2text(vector))

        count = 0
        for i in range(len(true)):
            if true[i][0:MAX_CAPTCHA] == pred[i]:
                count += 1
                print(true[i], pred[i])
        print("%d张判断正确，准确率%g" % (count, count / test_num))

        if result_dir is None:
            result_dir = './result.csv'
        if os.path.exists(result_dir):
            os.remove(result_dir)
        with open(result_dir, 'a+', encoding='utf-8') as file:
            for i in range(len(true)):
                file.write(true[i][0:MAX_CAPTCHA] + ',' + pred[i] + '\n')


if __name__ == '__main__':
    print("验证码图像heigth: %d width: %d" % (IMAGE_HEIGHT, IMAGE_WIDTH))
    print("验证码最大长度: %d" % MAX_CAPTCHA)
    train_model(gen_captcha=False, save_prefix='captcha2_5num_train')
    # fine_tuning_model(max_step=20000, base_model='./base_model_gray_0.8.model-10000',
    #                   save_prefix='accontest_fine_tuning_model_20000step_1500train')
    # test_model(model='./fine_tuning_model_12000step_2400train_last.model-12000')

    # train_crack_captcha_cnn(20000)
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
