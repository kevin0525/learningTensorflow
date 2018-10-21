#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:36:54 2018

@author: kevin
"""

import tensorflow as tf
w1= tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2= tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
x = tf.constant([[0.7, 0.9]])  
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
sess = tf.Session()
#sess.run(w1.initializer)  
#sess.run(w2.initializer)  
#another way to initial variable
init_op = tf.global_variables_initializer()  
sess.run(init_op)
print(sess.run(y))  
sess.close()