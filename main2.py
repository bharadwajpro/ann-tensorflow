import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np

tf.set_random_seed(0)
k_fold_num = 5
data = pd.read_csv('Bank_EXIT_Survey.csv')
kf = KFold(k_fold_num, shuffle=True, random_state=10)
le_city = LabelEncoder()
le_gender = LabelEncoder()
le_city.fit(['Pilani', 'Hyderabad', 'Goa'])
le_gender.fit(['Male', 'Female'])
data['City'] = le_city.transform(data['City'])
data['Gender'] = le_gender.transform(data['Gender'])
data.drop(['RowNumber', 'UID', 'Customer_name'], axis=1, inplace=True)
data = data.astype(np.float32)
# y = data['Status']
X = tf.placeholder(tf.float32, shape=[None, 10])
y = tf.placeholder(tf.float32, shape=[None, 2])
# X = data.loc[:, data.columns != 'Status']
n_nodes_hl1 = 20
n_nodes_hl2 = 10
n_nodes_hl3 = 10
n_nodes_hl4 = 5

n_classes = 2
batch_size = 100


def neural_network_model(td):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([10, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    hidden_4_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl4]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl4, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(td, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.relu(l4)

    output = tf.matmul(l4, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x, y):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 20
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(len(X_train) / batch_size)):
                # epoch_x, epoch_y = np.reshape(X_train[_*batch_size:_*batch_size+batch_size], (-1, 10)), np.reshape(y_train[_*batch_size:_*batch_size+batch_size], (-1, 1))
                epoch_x, epoch_y = X_train, y_train
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c/int(len(X_train) / batch_size)

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        #correct = print(y[0])
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: X_test, y: y_test}))


for train_index, test_index in kf.split(data):
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]
    X_train = train_data.loc[:, train_data.columns != 'Status']
    y_train = train_data['Status']
    X_test = test_data.loc[:, test_data.columns != 'Status']
    y_test = test_data['Status']
    y_test = np.reshape(y_test, (2000,1))
    y_train = np.reshape(y_train, (8000,1))
    
    # y_test = np.reshape(y_test, (1000,1))
    # y_train = np.reshape(y_train, (9000,1))
	
    from keras.utils.np_utils import to_categorical
    y_train= to_categorical(y_train)
    y_test = to_categorical(y_test)
    print(X_train.shape)
    print(y_test.shape)
    # exit(0)
    train_neural_network(X, y)
