#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 13:41:03 2024

@author: raminghorbani
"""




def input_gen(x, y, batch_size):
    size = (len(x)//batch_size)+1
    if len(x) == batch_size:
        size = 1
    if batch_size == 1:
        size = (len(x)//batch_size)
    if len(x) % batch_size == 0:
        size = (len(x)//batch_size)
    while True:
        
        for i in range(size):
            X = x[i*batch_size : (i+1)*batch_size]
            Y = y[i*batch_size : (i+1)*batch_size]
            yield X,Y
            
            
#steps per epochs           
def step_for_epoch(x, batch_size):
    size = (len(x)//batch_size)+1
    if len(x) == batch_size:
        size = 1
    if batch_size == 1:
        size = (len(x)//batch_size)  
    if len(x) % batch_size == 0:
        size = (len(x)//batch_size)
    return size
