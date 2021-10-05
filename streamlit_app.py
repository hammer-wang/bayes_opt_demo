from google.protobuf.symbol_database import Default
import streamlit as st
import numpy as np
import pandas as pd
import pickle as pkl
from PIL import Image
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from bayes_opt import UtilityFunction
import joblib
import os

np.random.seed(20)

def black_box_function(x1, x2):
    return -x1 ** 2 - (x2 - 1) ** 2 + 1

'''
# Bayesian Optimization for Active Experiment Design

_Author: Haozhu Wang_  
_Guo Group_

This web app is built for manufacturing optimization via Bayesian Optimization.

As a toy example, we look into maximizing a function $f(x_1, x_2) = -x_1^2 - (x_2 - 1) ^ 2 + 1.$ This function should be thought as a the groundtruth generation process for certain physical processes. 
'''

st.write('## Step I: Set feature value range.')
x1_min = st.number_input('x1_min', value=-1.0)
x1_max = st.number_input('x1_max', value=1.0)
x2_min = st.number_input('x2_min', value=-1.0)
x2_max = st.number_input('x2_max', value=2.0)

if 'opt' not in st.session_state:
    optimizer = BayesianOptimization(
        f=None,
        pbounds={'x1': (x1_min, x1_max), 'x2': (x2_min, x2_max)},
        verbose=2,
        random_state=42,
    )
    st.session_state['opt'] = optimizer

if 'params' not in st.session_state:
    st.session_state['params'] = None


utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)    
logs = {'x1':[], 'x2':[], 'target':[]}

st.write('## (Optional): Provide initial data points.')
'''
Initially provide 3 random guesses and their corresponding target values. 

(This feature is still under construction.)
'''
# with st.form(key='initial'):
# x1_1 = st.number_input(label='x1_1', value=2.0)
# x2_1 = st.number_input(label='x2_1', value=-1.0)
# y_1 = st.number_input(label='y_1', value=7.0)

# x1_2 = st.number_input(label='x1_2', value=0.0)
# x2_2 = st.number_input(label='x2_2', value=1.0)
# y_2 = st.number_input(label='y_2', value=0.0)

# x1_3 = st.number_input(label='x1_3', value=-1.0)
# x2_3 = st.number_input(label='x2_3', value=-1.0)
# y_3 = st.number_input(label='y_3', value=4.0)

# st.session_state['opt'].register(params = {'x1':x1_1, 'x2':x2_1}, target=y_1)
# st.session_state['opt'].register(params = {'x1':x1_2, 'x2':x2_2}, target=y_2)
# st.session_state['opt'].register(params = {'x1':x1_3, 'x2':x2_3}, target=y_3)

    # submit = st.form_submit_button('submit')



st.write('## II : Iterative optimization through experimentation')
'''
Each time, the Bayesian Optimization algorithm will propose a new point to explore given the current explored data. 

Users need to provide the target value obtained through experiment to the Bayesian Optimization algorithm before it can propose the next point.

step 1: click suggest to query the next data point.  
step 2: provide the measured target value by typing the value into the box.
step 3: click register to update the Bayesian Optimization model.
'''

def suggest_value():
    next_point = st.session_state.opt.suggest(utility)
    target_ = black_box_function(next_point['x1'], next_point['x2'])
    st.session_state['params'] = next_point
    st.write(f'hint {target_}')

def register(target):
    st.session_state['opt'].register(params = st.session_state.params, target=target)

st.button('Suggest', on_click=suggest_value)
value = st.number_input('target', value=0.0)
st.write(st.session_state.params)
st.write(value)
st.button('Resiger', on_click=register, args=(value,))

    
'''
## III : Visualize the experiment logs
'''
if st.button('Display log'):
    target_values = []
    for i, res in enumerate(st.session_state.opt.res):
        st.write("Iteration {}: \n\t{}".format(i, res))
        target_values.append(res['target'])

    st.line_chart(target_values)