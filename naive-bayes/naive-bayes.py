import numpy as np
import pandas as pd


## p (x,c) = (1/np.sqrt(2*np.pi*std_value)) * np.exp(-(x_value-mean_value)**2/2*std_value)
## receive: x_value - valued to be examined, mean_value and std_value of a feature
## returns: probability of x
def calculate_gaussian_distribution(x_value, mean_value, std_value):

    return (1/np.sqrt(2*np.pi*std_value)) * np.exp(-(x_value-mean_value)**2/
                                                   (2*std_value))

## receives: a data frame
## return: a DataFrame that contains feature title, its mean and standard deviation
def mean_and_std(data):
    
   x_train_summerized = []    
          
   for feature in data.columns:
       x_train_summerized.append([feature, data[feature].mean(), data[feature].std()])
        
   return pd.DataFrame(x_train_summerized)


def main():

    # 1. Handle the Data
    diabetes = pd.read_csv('diabetes.csv')
    
    x_train = diabetes.sample(frac = 0.7)
    x_test = diabetes.drop(x_train.index)
    
    x_train_0 = x_train[x_train['Outcome']==0]
    x_train_1 = x_train[x_train['Outcome']==1]
    
    y_test = x_test['Outcome']
    

    # 2. Summarize the Data (train)
    x_train_0_mean_and_std =  mean_and_std(x_train_0.drop('Outcome', axis=1))
    x_train_0_mean_and_std.columns = ['title','mean','str']
    x_train_0_mean_and_std = x_train_0_mean_and_std.set_index('title')
    print(x_train_0_mean_and_std)
    
    x_train_1_mean_and_std =  mean_and_std(x_train_1.drop('Outcome', axis=1))
    x_train_1_mean_and_std.columns = ['title','mean','str']
    x_train_1_mean_and_std = x_train_1_mean_and_std.set_index('title')
    print(x_train_1_mean_and_std)
    
    
    # 3. Write a prediction function
    predicte_label_0 = np.prod(calculate_gaussian_distribution(x_test.drop('Outcome', axis=1).iloc[0,:],
                                          x_train_0_mean_and_std.iloc[:,0],
                                          x_train_0_mean_and_std.iloc[:,1]))
    predicte_label_1 =  np.prod(calculate_gaussian_distribution(x_test.drop('Outcome', axis=1).iloc[0,:],
                                          x_train_1_mean_and_std.iloc[:,0],
                                          x_train_1_mean_and_std.iloc[:,1]))
    print('single prediction for label 1 is ', predicte_label_1>predicte_label_0)
    
    # 4. Make Predictions
    
    predicte_label_0 = []
    predicte_label_1 = []
    for index_value in range(0,len(x_test)):
        predicte_label_0.append(np.prod(calculate_gaussian_distribution(x_test.drop('Outcome', axis=1).iloc[index_value,:],
                                          x_train_0_mean_and_std.iloc[:,0],
                                          x_train_0_mean_and_std.iloc[:,1])))
        predicte_label_1.append(np.prod(calculate_gaussian_distribution(x_test.drop('Outcome', axis=1).iloc[index_value,:],
                                          x_train_1_mean_and_std.iloc[:,0],
                                          x_train_1_mean_and_std.iloc[:,1])))
 
    merged_predictions = np.greater(predicte_label_1, predicte_label_0)
    
    # 5. Evaluate Accuracy
    y_test = y_test.reset_index(drop=True)
    success_prediction_rate = np.equal(merged_predictions,y_test)
    count_true = np.count_nonzero(success_prediction_rate)

    print('Naise Bayse was correct {} times out of {}, which equals to a success rate of {}%' .format(count_true, len(y_test), round(100*count_true/len(y_test),2)))


if __name__ == "__main__":
    main()
