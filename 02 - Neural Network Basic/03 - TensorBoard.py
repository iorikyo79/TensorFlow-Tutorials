# -*- coding: utf-8 -*-
# 텐서보드를 이용하기 위해 각종 변수들을 설정하고 저장하는 방법을 익혀봅니다.

import tensorflow as tf
import numpy as np

#data = np.loadtxt('./data.csv', delimiter=',',
#                  unpack=True, dtype='float32')
data = np.genfromtxt('data.csv',dtype='float32',delimiter=',')  #unicode 지원을 위해 변경

#x_data = np.transpose(data[0:2])
#y_data = np.transpose(data[2:])
x_data = np.transpose(data[0:]) # 전체 Transpose 처리하고
y_data = x_data[2:]             # 3열부터 5열까까지 y_data로 할당
x_data = x_data[0:2]            # 0열부터 1열까지 x_data로 재할당
x_data = np.transpose(x_data[0:])   # Trnaspose 처리
y_data = np.transpose(y_data[0:])   #


#########
# 신경망 모델 구성
######
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# with tf.name_scope 으로 묶은 블럭은 텐서보드에서 한 레이어안에 표현해줍니다
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2, 30], -1., 1.), name="W1")
    L1 = tf.nn.relu(tf.matmul(X, W1))

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([30, 10], -1., 1.), name="W2")
    L2 = tf.nn.relu(tf.matmul(L1, W2))

with tf.name_scope('layer3'):
    W3 = tf.Variable(tf.random_uniform([10,30], -1., 1.), name="W3")
    L3 = tf.nn.relu(tf.matmul(L2, W3))

with tf.name_scope('output'):
    W4 = tf.Variable(tf.random_uniform([30, 3], -1., 1.), name="W4")
    model = tf.matmul(L3, W4)

with tf.name_scope('optimizer'):
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, Y))
    cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(Y, model))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost)

    # tf.summary.scalar 를 이용해 수집하고 싶은 값들을 지정할 수 있습니다.
    tf.summary.scalar('cost', cost)

#########
# 신경망 모델 학습
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 텐서보드에서 표시해주기 위한 값들을 수집합니다.
merged = tf.summary.merge_all()
# 저장할 그래프와 로그를 저장할 디렉토리를 설정합니다.
writer = tf.summary.FileWriter('c:/tmp/logs2', sess.graph)
# 이렇게 저장한 로그는, 학습 후 다음의 명령어를 이용해 웹서버를 실행시킨 뒤
# tensorboard --logdir=./logs
# 다음 주소와 웹브라우저를 이용해 텐서보드에서 확인할 수 있습니다.
# http://localhost:6006

for step in range(100):
    #sess.run(train_op, feed_dict={X: x_data, Y: y_data})
    ########################################################################################################
    # train_op : 실제 현재 모델의 학습을 실행시킬 op
    # cost : cost값을 중간에 확인해보기 위해 추가된 op.  cost를 run에 추가해서 출력된 값을 cost_value에 넣어둔다.
    # merged : summary에 기록할 내용을 생성하기 위한 op
    _, cost_value, summary = sess.run([train_op, cost, merged], feed_dict={X: x_data, Y:y_data})
    
    # 적절한 시점에 저장할 값들을 수집하고 저장합니다.
    #summary = sess.run(merged, feed_dict={X: x_data, Y:y_data})
    writer.add_summary(summary, step)

    # 확인을 위한 출력부 추가
    if step % 10 == 0:
        print( step, ':', cost_value )

#########
# 결과 확인
# 0: 기타 1: 포유류, 2: 조류
######
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print( '예측값:', sess.run(prediction, feed_dict={X: x_data}) )
print( '실제값:', sess.run(target, feed_dict={Y: y_data}) )

check_prediction = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
print( '정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}) )
