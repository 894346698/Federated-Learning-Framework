from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # 读取图片数据集
sess = tf.InteractiveSession()  # 创建session


# 一，函数声明部分

def weight_variable(shape):
    # 正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    # 池化卷积结果（conv2d）池化层采用kernel大小为2*2，步数也为2，周围补0，取最大值。数据量缩小了4倍
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 二，定义输入输出结构

# 声明一个占位符，None表示输入图片的数量不定，28*28图片分辨率
xs = tf.placeholder(tf.float32, [None, 28 * 28])
# 类别是0-9总共10个类别，对应输出分类结果
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
# x_image又把xs reshape成了28*28*1的形状，因为是灰色图片，所以通道是1.作为训练时的input，-1代表图片数量不定
x_image = tf.reshape(xs, [-1, 28, 28, 1])

# 三，搭建网络,定义算法公式，也就是forward时的计算

## 第一层卷积操作 ##
# 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征图像;
W_conv1 = weight_variable([5, 5, 1, 32])
# 对于每一个卷积核都有一个对应的偏置量。
b_conv1 = bias_variable([32])
# 图片乘以卷积核，并加上偏执量，卷积结果28x28x32
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 池化结果14x14x32 卷积结果乘以池化卷积核
h_pool1 = max_pool_2x2(h_conv1)

## 第二层卷积操作 ##
# 32通道卷积，卷积出64个特征
w_conv2 = weight_variable([5, 5, 32, 64])
# 64个偏执数据
b_conv2 = bias_variable([64])
# 注意h_pool1是上一层的池化结果，#卷积结果14x14x64
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
# 池化结果7x7x64
h_pool2 = max_pool_2x2(h_conv2)
# 原图像尺寸28*28，第一轮图像缩小为14*14，共有32张，第二轮后图像缩小为7*7，共有64张

## 第三层全连接操作 ##
# 二维张量，第一个参数7*7*64的patch，也可以认为是只有一行7*7*64个数据的卷积，第二个参数代表卷积个数共1024个
W_fc1 = weight_variable([7 * 7 * 64, 1024])
# 1024个偏执数据
b_fc1 = bias_variable([1024])
# 将第二层卷积池化结果reshape成只有一行7*7*64个数据# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# 卷积操作，结果是1*1*1024，单行乘以单列等于1*1矩阵，matmul实现最基本的矩阵相乘，不同于tf.nn.conv2d的遍历相乘，自动认为是前行向量后列向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout操作，减少过拟合，其实就是降低上一层某些输入的权重scale，甚至置为0，升高某些输入的权值，甚至置为2，防止评测曲线出现震荡，个人觉得样本较少时很必要
# 使用占位符，由dropout自动确定scale，也可以自定义，比如0.5，根据tensorflow文档可知，程序中真实使用的值为1/0.5=2，也就是某些输入乘以2，同时某些输入乘以0
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 对卷积结果执行dropout操作

## 第四层输出操作 ##
# 二维张量，1*1024矩阵卷积，共10个卷积，对应我们开始的ys长度为10
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# 最后的分类，结果为1*1*10 softmax和sigmoid都是基于logistic分类算法，一个是多分类一个是二分类
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 四，定义loss(最小误差概率)，选定优化优化loss，
cross_entropy = -tf.reduce_sum(ys * tf.log(y_conv))  # 定义交叉熵为loss函数
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)  # 调用优化器优化，其实就是通过喂数据争取cross_entropy最小化

# 五，开始数据训练以及评测
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={xs: batch[0], ys: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={xs: batch[0], ys: batch[1], keep_prob: 0.5})
print("test accuracy %g" % accuracy.eval(feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1.0}))
