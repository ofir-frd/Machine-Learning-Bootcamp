import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Create the training function (fit function) for the logistic regression
# model, which starts from some random values for the parameters of theta,
# and performs gradient descent to minimize the loss, this should be done 
# using a for loop.

def fit_logistic_regression(x_train, y_train, epoch_length, alpha):
    
    theta = np.random.uniform(-0.5,0.5, 4)
    cost_value = []
    
    for epoch in range(epoch_length):
        
        y_current = calculate_sigmoid(x_train@theta)
         
        new_gradient = calculate_gradient(x_train, y_current, y_train)
 #       print('new_gradient:', new_gradient)
        theta -= alpha * new_gradient
 #       print('theta:', theta)
        cost_value.append(calculate_log_likelihood(y_current, y_train))
    
    return predict_sigmoid(y_current), cost_value
    

# train-test split by a percentage
def split_df(user_df, split_ratio):
    
    x_train = user_df.sample(frac = split_ratio)
    x_test = user_df.drop(x_train.index)
    y_train = x_train['class']
    y_test = x_test['class']
    
    return x_train.drop('class', axis=1), x_test.drop('class', axis=1), y_train, y_test


# Create a function which calculates the gradient
def calculate_gradient(x_vector, y_predict, y_start):
    
    gradient_values = (y_predict - y_start)  @ x_vector
    gradient_values = gradient_values/len(x_vector)

    return gradient_values


# calculates the negative log likelihood function (which is the
# cost/loss function )
def calculate_log_likelihood(y_current, y_train):
 
    y_current = np.array(y_current)
    y_train = np.array(y_train)[np.newaxis]

    log_likelihood = - np.sum(y_train * np.log(y_current) + (
        1 - y_train) * np.log(1 - y_current), axis=1)/len(y_current)
 
    return log_likelihood

    
# returns the prediction of the sigmoid function (result>0.5)
def predict_sigmoid(x):
    final_prediction = []
    for value in x:
        if value > 0.5:
            final_prediction.append(1)
        else:
            final_prediction.append(0)
    return final_prediction
  

# calculates the logit function in numpy
def calculate_sigmoid(x):
  return 1 / (1 + np.exp(-x))

# creat line plot based of given color conditions
def plot_scatter(x,y, color_condition,labelx,labely,legend_title):
    
    figure, axis = plt.subplots(1, 1)

    plt.scatter(x,y , color = color_condition, label=legend_title)
    axis.set_xlabel(labelx)
    axis.set_ylabel(labely)
    axis.legend(ncol=2, loc='best', frameon=True)
    
    plt.show()
    
# creat scatter plot based of given color conditions
def plot_line(x,y, color_condition,labelx,labely,legend_title):
    
    figure, axis = plt.subplots(1, 1)

    plt.plot(x,y , color = color_condition, label=legend_title)
    axis.set_xlabel(labelx)
    axis.set_ylabel(labely)
    axis.legend(ncol=2, loc='best', frameon=True)
    
    plt.show()

def main():

    iris_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
    iris_df.columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)', 'class']

    iris_df['class'] = [1 if (x == 'Iris-setosa') else 0 for x in iris_df['class'] ]
  
    x = iris_df[['sepal length (cm)', 'sepal width (cm)', 'class']].copy()

#    print(x.head())

    color_condition = ['blue' if (x == 1) else 'red' for x in x['class'] ]
    plot_scatter(x['sepal length (cm)'],x['sepal width (cm)'], color_condition,
                 'sepal length (cm)', 'sepal length (cm)', 'Iris-setosa')
   
    
    #############
    ### train ###
    #############
    x_train, x_test, y_train, y_test = split_df(iris_df, 0.7)

    predictions, cost_values = fit_logistic_regression(x_train, y_train, 170, 0.05)
  
    success_prediction_rate = np.equal(predictions,y_train)
    count_true = np.count_nonzero(success_prediction_rate)
    
    color_condition = ['blue' if (x == 1) else 'red' for x in predictions]
    plot_scatter(x_train['sepal length (cm)'],x_train['sepal width (cm)'], color_condition,
                 'sepal length (cm)', 'sepal length (cm)', 'Iris-setosa')

    print('logistic regression train output: correct {} times out of {}, which equals to a success rate of {}%' .format(count_true, len(y_train), round(100*count_true/len(y_train),2)))
    
    plot_scatter(np.arange(len(cost_values)), cost_values, 'green',
                 'epoch (num.)', 'loss function (arb.)', 'loss_function')    
    
    ############
    ### test ###
    ############   
    
    predictions, cost_values = fit_logistic_regression(x_test, y_test, 170, 0.05)

    success_prediction_rate = np.equal(predictions,y_test)
    count_true = np.count_nonzero(success_prediction_rate)

    color_condition = ['blue' if (x == 1) else 'red' for x in predictions]
    plot_scatter(x_test['sepal length (cm)'],x_test['sepal width (cm)'], color_condition,
                 'sepal length (cm)', 'sepal length (cm)', 'Iris-setosa')
    
    print('logistic regression test output: correct {} times out of {}, which equals to a success rate of {}%' .format(count_true, len(y_test), round(100*count_true/len(y_test),2)))
    
    plot_scatter(np.arange(len(cost_values)), cost_values, 'green',
                 'epoch (num.)', 'loss function (arb.)', 'loss_function')    
    
    
if __name__ == '__main__':
    main()
