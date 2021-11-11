
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


def getting_and_knowing_your_data(chipo):

    #Step 4. See the first 10 entries
    print(chipo.head(10))
        
    #Step 5. What is the number of observations in the dataset?
    print(len(chipo.index))
    
    #Step 6. What is the number of columns in the dataset?
    print(len(chipo.columns))
    
    #Step 7. Print the name of all the columns.
    chipoFeatureList= []
        
    [chipoFeatureList.append(feature) for feature in chipo.columns]
              
     for feature in chipoFeatureList:
         print(feature)
    
    #Step 8. How is the dataset indexed?
    print(chipo.info)
    
    #Step 9. Which was the most ordered item?
    #Step 10. How many items were ordered?
    itemByOrderQuantity = chipo['item_name'].value_counts()
    print("{} with {} orders".format(itemByOrderQuantity.index[0],
                                     itemByOrderQuantity[0]))

    #Step 11. What was the most ordered item in the choice_description column?
    choiceDescriptionByQuantity = chipo['choice_description'].value_counts()
    print("{} with {} orders".format(choiceDescriptionByQuantity.index[0],
                                     choiceDescriptionByQuantity[0]))

    #Step 12. How many items were ordered in total? 
    print(chipo['quantity'].sum())
    
    #Step 13. Turn the item price into a float ( hint: you can create a
    #function and use the apply method to do this )  
    cleanItemPrice= []
    
    for i in range(0,len(chipo)):                                         
        newCell = chipo['item_price'][i]
        newCell = re.sub('\$', '', newCell)
        cleanItemPrice.append(float(newCell)) 
        
    chipo['item_price'].update(cleanItemPrice)
    chipo['item_price'] = chipo['item_price'].astype(float)    
    print(chipo['item_price'].dtypes)

    #Step 14. How much was the revenue for the period in the dataset?
    print(chipo['item_price'].sum(), '$')

    #Step 15. How many orders were made in the period?
    print(chipo['order_id'].max())

    #Step 16. What is the average amount per order?
    chipoByOrder = pd.DataFrame(chipo, columns=['order_id', 'item_price'])
    chipoByOrderMean = chipo.groupby(by='order_id').mean()
        
    print(chipoByOrderMean['item_price'].sum()/len(chipoByOrderMean.index))  
    
    #Step 17. How many different items are sold?
    print(len(itemByOrderQuantity.unique()))
    
    return chipo

def filtering_and_sorting(chipo)

    #Step 4. How many products cost more than $10.00?
    #Step 5. What is the price of each item?
    counterProducts = pd.DataFrame(chipo, columns=['item_name', 'item_price'])
    counterProductsByType = counterProducts.groupby(by='item_name').mean()
    
    totalBelowTenUSD = 0
    itemNamePrice = []
  
    for currentPrice in counterProductsByType['item_price']:
        if currentPrice>10:
            totalBelowTenUSD += 1
            itemNamePrice.append(currentPrice)
             
    print('Total items under $10:', totalBelowTenUSD)
    print('Prices list:', itemNamePrice)
    
    #Step 6. Print a data frame with only two columns item_name and item_price
    print(counterProducts)
    
    #Step 7. Sort by the name of the item
    counterProductsSortedByName = counterProducts.sort_values(by='item_name')
    print(counterProductsSortedByName)
    
    #Step 8. What was the quantity of the most expensive item ordered?
    print(counterProducts.sort_values(by='item_price').max())
    
    #Step 9. How many times were a Veggie Salad Bowl ordered?
    counterProductsSortedByName = pd.DataFrame(chipo, columns=['item_name'])
    counterProductsSortedByName = counterProductsSortedByName.assign(counterValue ='1')
    counterProductsSortedByName = counterProductsSortedByName.groupby(by='item_name').count()
    print(counterProductsSortedByName.loc['Veggie Salad Bowl']['counterValue'])
    
    #Step 10. How many times people ordered more than one Canned Soda?      
    itemByQuantity = pd.DataFrame(chipo, columns=['item_name', 'quantity'])
    quantitiesCounter = 0
    
    for i in range(0,len(itemByQuantity)):
        if itemByQuantity['item_name'][i] == 'Canned Soda' and itemByQuantity['quantity'][i] == 2:
            quantitiesCounter+=1

    print(quantitiesCounter)
    
    
def grouping(users):
    
    #Step 4. Discover what is the mean age per occupation
    usersMeanAgeByOccupation = pd.DataFrame(users, columns=['age','occupation'])
    usersMeanAgeByOccupation = usersMeanAgeByOccupation.groupby(by='occupation').mean()
    print(usersMeanAgeByOccupation)
    
    #Step 5. Discover the Male ratio per occupation and sort it from the most to the least
    usersRatioGenderByOccupation = pd.DataFrame(users, columns=['gender','occupation'])

    usersRatioGenderByOccupationMale = usersRatioGenderByOccupation.loc[
        usersRatioGenderByOccupation['gender'] == 'M']
    usersRatioGenderByOccupationFemale = usersRatioGenderByOccupation.loc[
        usersRatioGenderByOccupation['gender'] == 'F']
    
   usersRatioGenderByOccupationMale = usersRatioGenderByOccupationMale.groupby(
       by='occupation').count()
   usersRatioGenderByOccupationMale = usersRatioGenderByOccupationMale.rename(
       columns={'gender':'SumMale'})
   usersRatioGenderByOccupationFemale = usersRatioGenderByOccupationFemale.groupby(
       by='occupation').count()
   usersRatioGenderByOccupationFemale = usersRatioGenderByOccupationFemale.rename(
       columns={'gender':'SumFemale'})
   
   ratioList = pd.DataFrame(columns=['occupation','ratio'])
   for i in range(0,len(usersRatioGenderByOccupationMale)):
       if usersRatioGenderByOccupationMale.index[i] in usersRatioGenderByOccupationFemale.index:
           currentOccupation = usersRatioGenderByOccupationMale.index[i]
           ratioValue =  usersRatioGenderByOccupationMale['SumMale'][i]/usersRatioGenderByOccupationFemale[
               'SumFemale'][usersRatioGenderByOccupationMale.index[i]]
           
           ratioList.loc[len(ratioList.index)] = [currentOccupation,ratioValue]
           
       else:
          currentOccupation = usersRatioGenderByOccupationMale.index[i]
          ratioValue = 100
          ratioList.loc[len(ratioList.index)] = [currentOccupation,ratioValue]


   for i in range(0,len(usersRatioGenderByOccupationFemale)):
       if usersRatioGenderByOccupationFemale.index[i] not in usersRatioGenderByOccupationFemale.index:
           currentOccupation = usersRatioGenderByOccupationFemale.index[i]
           ratioValue = 0
           ratioList.loc[len(ratioList.index)] = [currentOccupation,ratioValue]

   ratioList = ratioList.sort_values(by = 'ratio', ascending = False)
   print(ratioList)    
    
   #Step 6. For each occupation, calculate the minimum and maximum ages
   usersOccupationAgeMinMax = pd.DataFrame(users, columns=['age','occupation'])
   occupationListMaxAge = usersOccupationAgeMinMax.groupby(by='occupation').max()
   occupationListMinAge = usersOccupationAgeMinMax.groupby(by='occupation').min()
   
   #Step 7. For each combination of occupation and gender, calculate the mean age
   usersOccupationGenderMeanAge = pd.DataFrame(users, columns=['occupation', 'gender' ,'age'])
   usersOccupationGenderMeanAge = usersOccupationGenderMeanAge.groupby(by=['occupation','gender']).mean()
   
   #Step 8. For each occupation present the percentage of women and men
   usersRatioGenderByOccupationSum = usersRatioGenderByOccupationMale[
       'SumMale'] + usersRatioGenderByOccupationFemale['SumFemale']
   usersRatioGenderByOccupationSum['doctor']=usersRatioGenderByOccupationMale.loc['doctor']
   
   usersRatioGenderByOccupationRatioMale = (usersRatioGenderByOccupationMale[
       'SumMale'] / usersRatioGenderByOccupationSum) * 100
   usersRatioGenderByOccupationRatioFeale = 100 - usersRatioGenderByOccupationRatioMale
   
   
def merge(data1, data2, data3):
    
    #Step 4. Join the two dataframes along rows and assign all_data 
    all_data = pd.concat([data1, data2], ignore_index=True, axis = 0)
    
    #Step 5. Join the two dataframes along columns and assign to all_data_col
    all_data_col = pd.concat([data1, data2], ignore_index=True, axis = 1)

    #Step 6. Print data3
    print(data3)    
    
    #Step 7. Merge all_data and data3 along the subject_id value
    all_data_and_data3 = all_data.merge(data3, on = 'subject_id', how = 'right')
    
    #Step 8. Merge only the data that has the same 'subject_id' on both data1 and data2
    all_data_same = data1.merge(data2, on = 'subject_id', how = 'inner')
    
    #Step 9. Merge all values in data1 and data2, with matching records
    #from both sides where available.
    all_data_same2 = data1.merge(data2, on = 'subject_id', how = 'outer')
   
    
   
def iris_function(iris):
    
    
    #Step 4. Create columns for the dataset
    iris.columns = ['SepalLength (cm)', 'SepalWidth (cm)', 'PetalLength (cm)', 'PetalWidth (cm)', 'class']
    
    
    #Step 5. Is there any missing value in the dataframe?
    irisFeatureList= []
    [irisFeatureList.append(feature) for feature in iris.columns]
              
    for feature in irisFeatureList:
        print("{} had {} % missing values".format(feature,np.round(iris[feature].isnull().sum()/len(iris)*100,2)))
        
    print ('\n')
    
    #Step 6. Let’s set the values of the rows 10 to 29 of the column 'petal_length' to NaN
    for i in range(10,29):
        iris.loc[i,'PetalLength (cm)'] = np.NaN
        
    #Step 7. Good, now lets substitute the NaN values to 1.0
    iris['PetalLength (cm)'] = iris['PetalLength (cm)'].replace(np.nan, 1.0)
    
    #Step 8. Now let's delete the column class
    iris = iris.drop(columns='class')
    
    #Step 9. Set the first 3 rows as NaN
    iris.iloc[0:3,:] = np.nan
    
    #Step 10. Delete the rows that have NaN
    iris = iris.dropna()
    
    #Step 11. Reset the index so it begins with 0 again
    iris = iris.reset_index()
    
    return iris
    
def main():
    
    chipo = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv', sep = '\t')

    chipo = getting_and_knowing_your_data(chipo)
    
       
    filtering_and_sorting(chipo)
  
    
    users = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user', sep = '|')
    
    users = grouping(users)
    
    
    data1 = pd.DataFrame({'subject_id': ['1', '2', '3', '4', '5'],
                         'first_name': ['Alex', 'Amy', 'Allen', 'Alice',
                                        'Ayoung'],
                         'last_name': ['Anderson', 'Ackerman', 'Ali',
                                       'Aoni', 'Atiches']})

    data2 = pd.DataFrame({'subject_id': ['4', '5', '6', '7', '8'],
                          'first_name': ['Billy', 'Brian', 'Bran', 'Bryce',
                                         'Betty'],
                          'last_name': ['Bonder', 'Black', 'Balwner', 'Brice',
                                        'Btisan']}) 
      
    data3 = pd.DataFrame({'subject_id': ['1', '2', '3', '4', '5', '7', '8',
                                         '9', '10', '11'],
                          'test_id': [51, 15, 15, 61, 16, 14, 15, 1, 61, 16]}) 
    
    merge(data1, data2, data3)
    

    iris = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
    
    iris = iris_function(iris)
    
if __name__ == "__main__":
    main()