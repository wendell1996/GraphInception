import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import random as rd
import time
import LSI
import GraphInterface
import networkx as nx

def next_batch(array_x,array_y,batch_Num):
    len = array_x.shape[0]
    batch_data_x = np.zeros((batch_Num,array_x.shape[1]))
    batch_data_y = np.zeros((batch_Num,array_y.shape[1]))
    for i in range(0,batch_Num,1):
        index = rd.randint(0,len-1)
        batch_data_x[i,:] = array_x[index,:]
        batch_data_y[i,:] = array_y[index,:]
    return batch_data_x,batch_data_y

def titleEmbedding(G,LSI,k=300):
    inputEmbedding = np.zeros([G.order(),k])
    for i,node in enumerate(G.nodes):
        titleWords = G.node[node]['paperTitle'].split(' ')
        for word in titleWords:
            inputEmbedding[i,:] = inputEmbedding[i,:] + LSI.word2vec(word,k).reshape([1,300])
    return inputEmbedding

data_size = 1000
path = '/Users/wendellcoma/Documents/data/dblp-ref/dblp-ref-3.json'
graphml_path = '/Users/wendellcoma/Documents/data/dblp-ref/DBLP3_2000.graphml'
subgraphml_path = '/Users/wendellcoma/Documents/data/dblp-ref/sub1DBLP3_2000.graphml'
DBLPFeature_path = '/Users/wendellcoma/Documents/data/dblp-ref/DBLP3Feature.txt'
GI = GraphInterface.GraphInterface()
G = GI.DBLPjson2networkxGraph(path,True,name='DBLP3',size=data_size)
#GI.graph2graphml(G,graphml_path)
H = GI.getSubgraph(G,[1,1,1],[4,3,2])
#得到P K-Order矩阵，可以保存
#PKOrder = GI.getPInKOrder(H,20)
#np.save('P20order.npy',PKOrder)
#GI.graph2graphml(H,subgraphml_path)
#GI.exportDBLPFeature(path,DBLPFeature_path,size=data_size)

#训练LSI-Embedding 训练好导出，使用时载入即可
#path_embeding = "/Users/wendellcoma/Documents/data/dblp-ref/DBLP3Feature.txt"
#LSI = LSI.LSI()
#LSI.trainModel(path_embeding,method='Frequency')
#wordList = LSI.getWordList()
#wordDic = LSI.getWordDic()
#x_train = titleEmbedding(H,LSI)
#np.save('inputDBLP3306_300.npy',x_train)

#模型参数
sample_len = 3306
samplet_len = 3306
sample_feature_len=300
#input_width = 57
hidden_len = 300
#PK20order
kernel_num = 3
classNum = 4
batch_size = 1

#导入数据
x_train = np.load('inputDBLP3306_300.npy').reshape([batch_size,sample_len,sample_feature_len,1])
y_train = np.load('DBLP_y_train3306.npy')
x_test = np.load('inputDBLPt.npy').reshape([batch_size,samplet_len,sample_feature_len,1])
y_test = np.load('DBLP_y_t.npy')
print('Training data loaded!')

PKOrder = np.load('P20order.npy')
#P = np.empty(shape=[PKOrder.shape[1],PKOrder.shape[2],1,kernel_num])
#for i in range(PKOrder.shape[0]):
#    P[:,:,0,i] = PKOrder[i,:,:]

#建立计算网
x = tf.placeholder(dtype=tf.float32,shape=[None,sample_len,sample_feature_len,1])
print('input:',x)
label = tf.placeholder(tf.float32,[None,classNum])
print('labels:',label)
filter_weight1 = tf.constant(PKOrder,dtype=tf.float32)
PMat = tf.constant(PKOrder,shape=PKOrder.shape,dtype=tf.float32)
for i in range(kernel_num):
    con_temp = tf.matmul(PMat[i,:,:],x[0,:,:,0])
    con_temp = tf.reshape(con_temp,[batch_size,con_temp.get_shape()[0],con_temp.get_shape()[1],1])
    #print('con_temp:',con_temp)
    if i == 0:
        con1 = con_temp
    else:
        con1 = tf.concat([con1,con_temp],axis=3)
        #print("con1:",con1)
pool1 = tf.nn.max_pool(con1,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME')
y1 = tf.reshape(pool1,[pool1.get_shape()[0],pool1.get_shape()[1]*pool1.get_shape()[2]*pool1.get_shape()[3]])
#print('y1:',y1)
nodes = y1.get_shape().as_list()
w1 = tf.Variable(tf.random_normal([nodes[1],classNum],stddev=0.1))
b1 = tf.Variable(tf.fill([classNum],0.1))
y2 = tf.nn.relu(tf.matmul(y1,w1))+b1

#loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y2,labels=label,name='cross_entropy_per_example')
cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy_mean)

#训练
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for i in range(1000):
    start_time = time.time()
    sess.run(train_step,feed_dict={x:x_train,label:y_train})
    _,loss_value = sess.run([train_step,cross_entropy_mean],feed_dict={x:x_train,label:y_train})
    duration = time.time()-start_time
    if i % 100 == 0:
        print('step%d,loss=%.2f(%.3fsec)'%(i,loss_value,duration))

#预测
correct_predict = tf.equal(tf.argmax(y2,1),tf.argmax(label,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict,tf.float32))
print('acuracy:%.6f'%accuracy.eval({x:x_test,label:y_test}))
