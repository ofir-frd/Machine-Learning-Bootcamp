
'''
h(𝑥) = t0 + t1x +t2𝑥^2

X Y
0 1
1 3
2 7

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data1 = pd.DataFrame({'X': [0,1,2],
        'Y': [1,3,7]})

t0 = 20
t1 = 20
t2 = 0

def plot_result(x, y):    
    fig, axis = plt.subplots(figsize=(5,5))
    
    plt.plot(x,y)
    axis.set_title('Loss function')
    axis.set_xlabel("Step num.")
    axis.set_ylabel("loss funcition value")
  
    plt.show()


'''
A. Find the loss at the starting point and after 200 iterations, using the following
learning rates: 0.01, 0.1, 1
'''

def loss_function(y_calculated,y_target):
    return (((y_calculated-y_target)**2).mean())/2


def gradient_descent(theta, x_target, y_target, alpha, iterations):
    
    y_current = np.zeros(3).astype('float32')
    gradient_t = np.zeros([3,3]).astype('float32')
    power_values = np.array([0,1,2]).astype('float32')
    loss_result = []
    
    for i in range(iterations):
        
        y_current = theta[0] + theta[1]*x_target + theta[2]*x_target**2
        print('y_current:', y_current)
 
        loss_result.append(loss_function(y_current, y_target))

        gradient_t[0] = (y_current-y_target) * x_target**power_values[0]
        gradient_t[1] = (y_current-y_target) * x_target**power_values[1]
        gradient_t[2] = (y_current-y_target) * x_target**power_values[2]
        print('gradient_t:', gradient_t)
        
        theta[0] -= alpha*gradient_t[0].mean()
        theta[1] -= alpha*gradient_t[1].mean()
        theta[2] -= alpha*gradient_t[2].mean()
        print('theta:', theta)
        
        if i > 3:
            if loss_result[i] == loss_result[i-1] and loss_result[
                    i-1] == loss_result[i-2] or loss_result[i] > 1e+6 or loss_result[i] < 1e-6:
                print('last iteration:', i)
                actualSteps = i
                break
           
    print(theta, loss_result) 
       

gradient_descent(np.array([t0,t1,t2]), np.array(data1['X']), np.array(data1['Y']), 0.01, 200) 
gradient_descent(np.array([t0,t1,t2]), np.array(data1['X']), np.array(data1['Y']), 0.1, 200) 
gradient_descent(np.array([t0,t1,t2]), np.array(data1['X']), np.array(data1['Y']), 1, 100) 


'''
B. For each learning rate, explain why did the gradient descent succeed/fail?
'''

# in some cases the steps were to big and the theta diverged

'''
C. Repeat the process using LR=0.1, but this time with momentum γ = 0. 9.
'''

def moment_gradient_descent(theta, x_target, y_target, alpha, gama, iterations):

    y_current = np.zeros(3).astype('float32')
    velocity = np.zeros(3).astype('float32')
    gradient_t = np.zeros([3,3]).astype('float32') 
    power_values = np.array([0,1,2]).astype('float32')
    loss_result = []
    actualSteps = iterations
    
    for i in range(iterations):
        
        y_current = theta[0] + theta[1]*x_target + theta[2]*x_target**2
        print('y_current:', y_current)
 
        loss_result.append(loss_function(y_current, y_target))
        
        gradient_t[0] = (y_current-y_target) * x_target**power_values[0]
        gradient_t[1] = (y_current-y_target) * x_target**power_values[1]
        gradient_t[2] = (y_current-y_target) * x_target**power_values[2]
        
        gradient_t_means = np.nanmean(gradient_t, axis=1)
        
        velocity = gama*velocity + alpha*gradient_t_means
        
        theta -= velocity
      
        if i > 3:
            if loss_result[i] == loss_result[i-1] and loss_result[
                    i-1] == loss_result[i-2] or loss_result[i] > 1e+6 or loss_result[i] < 1e-6:
                print('last iteration:', i)
                actualSteps = i
                break
           
    print(theta, loss_result) 
       
    
    plot_result(np.arange(actualSteps),loss_result)
    
    
moment_gradient_descent(np.array([t0,t1,t2]).astype('float32').astype(
    'float32'), np.array(data1['X']).astype('float32'), np.array(data1[
        'Y']), 0.1,0.9, 100)


#y=m.T@x+b



