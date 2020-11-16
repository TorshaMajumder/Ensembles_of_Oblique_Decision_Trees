#
#
# Implementation of Deep Neural Decision Trees by [Yongxin Yang et al.]
# ````Some part of the code has been taken from the original paper````
#
#
#......Importing all the packages...................
#
import numpy as np
import tensorflow as tf
from neural_network_decision_tree import nn_decision_tree
#
# Setting the random seed
#
np.random.seed(1943)
tf.set_random_seed(1943)

def random1(x):
    x1=np.random.choice(x,len(x))
    return x1


def ensemble(x):
    a = list(filter(None.__ne__, set(x)))
    l=[]
    tm=0
    for j in range(len(a)):
        tm=0
        for i in range(x.shape[0]):
            if(x[i]==a[j]):
                tm+=1
        l.append(tm)
    m = max(l)
    r = l.index(m)
    return a[r]

def dndt_predict(train_X, test_X, train_Y, num_class, num_cut, num_leaf, n_bag):
    #
    #
    d = train_X.shape[1]
    sess = tf.InteractiveSession()
    x_ph = tf.placeholder(tf.float32, [None, d])
    y_ph = tf.placeholder(tf.float32, [None, num_class])
    cut_points_list = [tf.Variable(tf.random_uniform([i])) for i in num_cut]
    leaf_score = tf.Variable(tf.random_uniform([num_leaf, num_class]))
    y_pred = nn_decision_tree(x_ph, cut_points_list, leaf_score, temperature=0.1)
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=y_pred, onehot_labels=y_ph))
    opt = tf.train.AdamOptimizer(0.1)
    train_step = opt.minimize(loss)
    sess.run(tf.global_variables_initializer())
    x_example = []
    y_label = []
    y_ens = []
    label_list = []
    indx = np.array(range(len(train_X)))

    for k in range(n_bag):
        examples = random1(indx)
        for i in examples:
            x = train_X[i]
            x_example.append(x)
            x_data = np.array(x_example)
            y = train_Y[i]
            y_label.append(y)
            y_data = np.array(y_label)
        #
        #
        for i in range(1000):
            _, loss_e = sess.run([train_step, loss], feed_dict={x_ph: train_X, y_ph: train_Y})
        #
        #
        sample_label = np.argmax(y_pred.eval(feed_dict={x_ph: test_X}), axis=1)
        #
        #
        label_list.append(sample_label)
        x_example.clear()
        y_label.clear()

    avg = np.array(label_list)
    avg = avg.reshape(n_bag, len(test_X))
    avg = np.transpose(avg)

    for i in range(len(test_X)):
        x = ensemble(avg[i,])
        y_ens.append(x)
    return y_ens


def dndt_fit(train_X, test_X, train_Y, d, num_class, n_bag):

    if d >= 12:
        new_features = np.random.choice(d, 10, replace=False)
        train_X = np.array(train_X[:,new_features])
        test_X = np.array(test_X[:, new_features])
        num_cut = np.ones((train_X.shape[1],), dtype=int)
        num_leaf = np.prod(np.array(num_cut) + 1)
        y_pred = dndt_predict(train_X, test_X, train_Y, num_class, num_cut, num_leaf, n_bag)
    else:
        num_cut = np.ones((d,), dtype=int)
        num_leaf = np.prod(np.array(num_cut) + 1)
        y_pred = dndt_predict(train_X, test_X, train_Y, num_class, num_cut, num_leaf, n_bag)

    return y_pred




