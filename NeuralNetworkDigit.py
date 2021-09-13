import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
import glob2 as glob
from PIL import Image

def load_training_data():
    filelist_train_X = glob.glob('C:/Users/Aman Sheth/Pictures/TrainingData/*.png')
    train_X = np.expand_dims(1*np.invert(np.array([np.array(Image.open(fname).convert('1')) for fname in filelist_train_X])),axis = 3)
    filelist_test_X = glob.glob('C:/Users/Aman Sheth/Pictures/TestingData/*.png')
    test_X = np.expand_dims(1*np.invert(np.array([np.array(Image.open(fname).convert('1')) for fname in filelist_test_X])), axis = 3)
    train_Y = np.array([y_getter(xi) for xi in filelist_train_X])
    test_Y = np.array([y_getter(xi) for xi in filelist_test_X])
    
    return train_X, train_Y, test_X, test_Y
    

def y_getter(filename):
    Y = np.zeros(10,float)
    if("0_" in filename):
        Y[0] = 1.0
    elif("sprite_" in filename):
        Y[1] = 1.0
    elif("2_" in filename):
        Y[2] = 1.0
    elif("3_" in filename):
        Y[3] = 1.0
    elif("4_" in filename):
        Y[4] = 1.0
    elif("5_" in filename):
        Y[5] = 1.0
    elif("6_" in filename):
        Y[6] = 1.0
    elif("7_" in filename):
        Y[7] = 1.0
    elif("8_" in filename):
        Y[8] = 1.0
    elif("9_" in filename):
        Y[9] = 1.0
    return Y

    

def unison_shuffled_copies(a,b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def random_mini_batches(X_train, Y_train, minibatch_size, seed):
    minibatches = []
    X_train,Y_train = unison_shuffled_copies(X_train, Y_train)
    for i in range(0,X_train.shape[0],minibatch_size):
        X_train_mini = X_train[i:i+minibatch_size]
        Y_train_mini = Y_train[i:i+minibatch_size]
        minibatches.append((X_train_mini,Y_train_mini))
    return minibatches
    
def create_placeholders(n_H0,n_W0,n_C0,n_Y):
    '''Creates place holders for inputs and outputs, x and ys, for training 
    H0 = height of input image
    W0 = width of input image
    C0 = number of rgb channels
    Y = number of classes of output'''
    
    x = tf.placeholder(tf.float32,shape = (None,n_H0,n_W0,n_C0), name = "x")
    y = tf.placeholder(tf.float32,shape = (None,n_Y),name = "y")
    return x,y
    '''x = placeholder for data input 
       y = placeholder for input labels(answer)'''
       


def intialize_parameters():
    w1 = tf.get_variable("W1",[4,4,1,8],initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    w2 = tf.get_variable("W2",[2,2,8,16],initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    parameters = {"W1":w1,"W2":w2}
    return parameters



def forward_propagation(X,parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    """con2D -> Relu -> Maxpool -> Con2D -> Relu -> Maxpool -> Flatten -> FullyConnected"""
    
    #Con2D
    Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME')   
    #Relu
    A1 = tf.nn.relu(Z1)
    #Maxpool
    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')
    #Con2D
    Z2 = tf.nn.conv2d(P1, W2, strides = [1,1,1,1], padding = 'SAME')
    #Relu
    A2 = tf.nn.relu(Z2)
    #Maxpool
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    #Flatten
    P2 = tf.contrib.layers.flatten(P2)
    #FullyConnected
    Z3 = tf.contrib.layers.fully_connected(P2, 10, activation_fn = None)
    return Z3



def compute_cost(Z3,Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3,labels = Y))
    return cost



def model(X_train, Y_train, X_test, Y_test, learning_rate = .009, num_epoch = 100, minibatch_size = 64, print_cost = True):
    ops.reset_default_graph() #?
    tf.set_random_seed(1)
    seed = 3
    (m,n_H0,n_W0,n_C0) = X_train.shape
    n_Y = Y_train.shape[1]
    costs = []
    x,y = create_placeholders(n_H0,n_W0,n_C0,n_Y)
    parameters = intialize_parameters()
    Z3 = forward_propagation(x, parameters)
    cost = compute_cost(Z3, y)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epoch):
            minibatch_cost = 0
            num_minibatch = int(m/minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X,minibatch_Y) = minibatch
                _ , temp_cost = sess.run([optimizer,cost], feed_dict = {x:minibatch_X, y:minibatch_Y})
                minibatch_cost += temp_cost / num_minibatch
                
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i %f" % (epoch,minibatch_cost))
                
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
                
        plt.plot(np.squeeze(costs))
        plt.ylabel('Cost')
        plt.xlabel('Iteration per Tens')
        plt.title('Learning Rate Equal To ' + str(learning_rate))
        plt.show()
        
        predict_OP = tf.arg_max(Z3,1,name = "predict_OP")
        predict_OP = tf.Print(predict_OP,[predict_OP])
        correct_prediction = tf.equal(predict_OP,tf.arg_max(y,1))
        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({x:X_train,y:Y_train})
        test_accuracy = accuracy.eval({x:X_test,y:Y_test})
        print(train_accuracy, test_accuracy, parameters)
        tf.train.Saver().save(sess,"C:/Users/Aman Sheth/Pictures/Field/model.ckpt")
    return train_accuracy, test_accuracy, parameters

#%%
X_orig_train,Y_orig_train,X_orig_test,Y_orig_test = load_training_data()
print(Y_orig_train[720])
#%%
_,_,parameters = model(X_orig_train,Y_orig_train,X_orig_test,Y_orig_test)
        
#%%
sess = tf.Session()
saver = tf.train.import_meta_graph('C:/Users/Aman Sheth/Pictures/Field/model.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('C:/Users/Aman Sheth/Pictures/Field/'))
graph = tf.get_default_graph()
#print(tf.contrib.graph_editor.get_tensors(graph))
'''for op in graph.get_operations():
    print(str(op.name))'''
x = graph.get_tensor_by_name("x:0")
predict_OP = graph.get_tensor_by_name("predict_OP:0")


#%%
def load_field_image():
    file = Image.open("C:/Users/Aman Sheth/Pictures/Field/Untitled.png")
    field_X = np.expand_dims(1 * np.invert(np.array([np.array(file.convert('1'))])), axis = 3)
    return file,field_X



fieldFile, field_X = load_field_image()
#y = sess.run([predict_OP],feed_dict = {x:field_X})
plt.imshow(fieldFile)
#print(field_X)
print(sess.run(predict_OP,feed_dict = {x:field_X}))