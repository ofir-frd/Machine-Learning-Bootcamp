
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

   
def apply_function(crime):
     
    #Step 4. What is the data type of the columns? # all int64
    print(crime.dtypes)
    
    #Step 5. Convert the type of the column Year to datetime64
    crime['Year'] = pd.to_datetime(crime['Year'], format='%Y')
    
    #Step 6. Set the Year column as the index of the dataframe
    crime = crime.set_index('Year')
    
    #Step 7. Delete the Total column
    crime = crime.drop('Total', axis =1)
    
    #Step 8. Group the year by decades and sum the values
    #Pay attention to the Population column number, 
    #summing this column is a mistake.
    
    crime_by_year = pd.DataFrame(crime, columns=['Violent', 'Property', 'Murder',
       'Forcible_Rape', 'Robbery', 'Aggravated_assault', 'Burglary',
       'Larceny_Theft', 'Vehicle_Theft'])
    crime_by_year = crime_by_year.groupby(pd.Grouper(level='Year', freq='10Y')).sum()
    
    population_by_year = pd.DataFrame(crime, columns=['Population'])
    population_by_year = population_by_year.groupby(pd.Grouper(level='Year', freq='10Y')).min()
    crime_by_year = pd.concat([crime_by_year, population_by_year],  axis = 1)
    
    # Step 9. What is the most dangerous decade to live in the US?
    
    total_crime_per_year = pd.DataFrame(crime_by_year, columns=['Violent', 'Property', 'Murder',
       'Forcible_Rape', 'Robbery', 'Aggravated_assault', 'Burglary',
       'Larceny_Theft', 'Vehicle_Theft'])
    
    total_crime_per_year = total_crime_per_year.sum(axis=1)
    crime_ratio =  total_crime_per_year[:]/population_by_year['Population']
    print('The most dangerous decade is', crime_ratio[crime_ratio == crime_ratio.max()].index.tolist())

    return crime_by_year
    

def stats_function(baby_names):
    
    #Step 4. See the first 10 entries 0' and 'Id'
    print(baby_names.head(10))
    
    #Step 6. Is there more male or female names in the dataset?
    gender_data = baby_names.groupby(by='Gender')['Name'].count()
    print('There are more {} names' .format(gender_data[gender_data == gender_data.max()].index.tolist()))
    
    #Step 7. Group the dataset by name and assign to names
    names =  baby_names.groupby(by='Name')['Count'].sum()
    
    #Step 8. How many different names exist in the dataset?
    print('There are {} different names' .format(len(names.index)))
    
    #Step 9. What is the name with most occurrences?
    print('The most common name is:', names[names == names.max()].index.tolist())
    
    #Step 10. How many different names have the least occurrences?
    leastOccurrences = 0
    for singleValue in names:
        if singleValue == names.min():
            leastOccurrences +=1    
    print('There are {} name which have least occurrences of {}' .format(leastOccurrences, names.min()))
    
   #Step 11. What is the median name occurrence? 
   medianNames = [names == names.median()].count()
   print('Median name occurrance of {} can be found {} times' .format(int(names.median()),medianNames[0][:].sum())) 
   
   #Step 12. What is the standard deviation of names?
   print('The names standard deviantion is', int(names.std()))
   
   #Step 13. Get a summary with the mean, min, max, std and quartiles.
   print(names.describe())
   
    
def visualization_function(chipo):
    
    #Step 4. See the first 10 entries 0' and 'Id'

    
    #Step 5. Create a histogram of the top 5 items bought
    topProducts = pd.DataFrame(chipo, columns=['item_name', 'quantity'])
    topProducts = topProducts.groupby(by = 'item_name').count()
    topProducts = topProducts.sort_values(by = 'quantity', ascending=False).head(5)
    
    fig, axis = plt.subplots(figsize=(5,5))
    
    plt.bar(topProducts.index,topProducts['quantity'])
    plt.xticks(rotation=90)
    axis.set_title('top 5 items bought')
  
    plt.show()
    
    #Step 6. Create a scatterplot with the number of items ordered per order price,
    #Price should be in the X-axis and Items ordered in the Y-axis
    
    items_and_price =pd.DataFrame(chipo, columns=['item_name', 'item_price'])
    cleanItemPrice = []

    for i in range(0,len(items_and_price)):                                         
        newCell = items_and_price['item_price'][i]
        newCell = re.sub('\$', '', newCell)
        cleanItemPrice.append(float(newCell)) 
        
    items_and_price['item_price'].update(cleanItemPrice)
    items_and_price['item_price'] = items_and_price['item_price'].astype(float)  
    
    items_and_price = items_and_price.groupby(by = 'item_price').count()
    
    fig, axis = plt.subplots(figsize=(5,5))
    
    plt.scatter(items_and_price.index,items_and_price['item_name'])
    axis.set_title('Items per Price')
    axis.set_xlabel("Price ($)")
    axis.set_ylabel("Quantity")
  
    plt.show()
    
    
def creating_series_and_dataframes():
    
    #Step 2. Create a data dictionary that looks like the DataFrame below
    #Step 3. Assign it to a variable called
    data1 = pd.DataFrame({'evolution': ['Ivysaur', 'Charmeleon', 'Wartortle',
                                        'Metapod'],
                     'hp': ['45', '39', '44', '45',],
                     'name': ['Bulbasaur', 'Charmander', 'Squirtle',
                              'Caterpie'],
                     'pokedex': ['yes', 'no', 'yes', 'no'],
                     'type': ['grass', 'fire', 'water', 'bug']})
    print(data1)
    
    #Step 4. Oops...it seems the DataFrame columns are in alphabetical order. Place the
    #order of the columns as name, type, hp, evolution, pokedex
    data1 = data1[['name', 'type', 'hp', 'evolution', 'pokedex']]
    
    #Step 5. Add another column called place, and insert what you have in mind.
    place = ['here', 'there', 'faraway', 'closeby']
    data1['place'] = place
    print(data1)
    
    #Step 6. Present the data type of each column
    print(data1.dtypes)
    

def time_series(invest):    
    
   #Step 4. What is the frequency of the dataset? (i.e., what’s the time frequency?)
   invest['Date'] = pd.to_datetime(invest['Date'])
   print(invest['Date'][1]-invest['Date'][0]) # a week
   
   #Step 5. Set the column Date as the index.
   invest = invest.set_index('Date')
   
   #Step 6. What is the data type of the index?
   print(invest.index.dtype)
   
   #Step 7. Set the index to a DatetimeIndex type
   #Done on Step 4
   
   #Step 8. Change the frequency to monthly, sum the values and assign it to monthly.
   investMonthly = invest.groupby(pd.Grouper(level='Date', freq='1M')).sum()
   print(investMonthly)

   #Step 9. You will notice that it filled the dataFrame with months that don't have any data
   #with NaN. Let's drop these rows.
   investMonthly = investMonthly.loc[(investMonthly!=0).any(1)]
   print(investMonthly)
    
   #Step 10. Good, now we have the monthly data. Now change the frequency to year.
   investAnnually = investMonthly.groupby(pd.Grouper(level='Date', freq='1Y')).sum()
   print(investAnnually)
   
   
def main():

    crime = pd.read_csv('https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/04_Apply/US_Crime_Rates/US_Crime_Rates_1960_2014.csv')

    crime_by_year = appay_function(crime)

    
    baby_names = pd.read_csv('NationalNames.csv')

    baby_names = apply_function(baby_names)


    chipo = pd.read_csv('https://raw.githubusercontent.com/justmarkham/DAT8/master/data/chipotle.tsv', sep = '\t')

    visualization_function(chipo)


    creating_series_and_dataframes()


    invest = pd.read_csv('https://raw.githubusercontent.com/datasets/investor-flow-of-funds-us/master/data/weekly.csv')

    time_series(invest)

if __name__ == "__main__":
    main()