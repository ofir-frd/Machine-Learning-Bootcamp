
'''
### Question 1

Level 1

Question:
Write a program which will find all such numbers which are divisible by 7 but are not a multiple of 5, between 2000 and 3200 (both included).
The numbers obtained should be printed in a comma-separated sequence on a single line.

Hints: 
Consider use range(#begin, #end) method
'''

numbersList = []
for num in range(2000,3201):
    if num % 7 == 0 and num % 5 != 0:
            numbersList.append(num)
print(numbersList)


'''
### Question 2
Level 1

Question:
Write a program which can compute the factorial of a given numbers.
The results should be printed in a comma-separated sequence on a single line.
Suppose the following input is supplied to the program:
8
Then, the output should be:
40320

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.

'''

def factorical_calculator(inputNumber):

    if inputNumber < 2:
        return inputNumber
    
    return inputNumber * factorical_calculator(inputNumber-1)

print(factorical_calculator(8))


'''
### Question 3
Level 1

Question:
With a given integral number n, write a program to generate a dictionary that contains (i, i*i) such that is an integral number between 1 and n (both included). and then the program should print the dictionary.
Suppose the following input is supplied to the program:
8
Then, the output should be:
{1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64}

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
Consider use dict()
'''

def create_dictionary(inputNumber):

    newDictionary = {}

    for num in range(1,inputNumber+1):
        newDictionary[str(num)] = str(num*num)  
    
    return newDictionary

print(create_dictionary(8))


'''
### Question 4
Level 1

Question:
Write a program which accepts a sequence of comma-separated numbers from console and generate a list and a tuple which contains every number.
Suppose the following input is supplied to the program:
34,67,55,33,12,98
Then, the output should be:
['34', '67', '55', '33', '12', '98']
('34', '67', '55', '33', '12', '98')

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
tuple() method can convert list to tuple
'''



userInput = input('Please input a sequence of comma-separated numbers')
#userInput = '34,67,55,33,12,98'
splitedNumbersList = userInput.split(',')
splitedNumbersTuple = tuple(splitedNumbersList)
print(splitedNumbersList,splitedNumbersTuple)

'''
### Question 6
Level 2

Question:
Write a program that calculates and prints the value according to the given formula:
Q = Square root of [(2 * C * D)/H]
Following are the fixed values of C and H:
C is 50. H is 30.
D is the variable whose values should be input to your program in a comma-separated sequence.
Example
Let us assume the following comma separated input sequence is given to the program:
100,150,180
The output of the program should be:
18,22,24

Hints:
If the output received is in decimal form, it should be rounded off to its nearest value (for example, if the output received is 26.0, it should be printed as 26)
In case of input data being supplied to the question, it should be assumed to be a console input. 
'''
import math
userInput = input('Please input a sequence of comma-separated numbers')
#userInput = '100,150,180'
splitedNumbers = userInput.split(',')
calculatedValues = [int(math.sqrt((2 * 50 * int(num))/30)) for num in splitedNumbers]

print(calculatedValues)

'''
### Question 7
Level 2

Question:
Write a program which takes 2 digits, X,Y as input and generates a 2-dimensional array. The element value in the i-th row and j-th column of the array should be i*j.
Note: i=0,1.., X-1; j=0,1,¡­Y-1.
Example
Suppose the following inputs are given to the program:
3,5
Then, the output of the program should be:
[[0, 0, 0, 0, 0], [0, 1, 2, 3, 4], [0, 2, 4, 6, 8]] 

Hints:
Note: In case of input data being supplied to the question, it should be assumed to be a console input in a comma-separated form.
'''

import numpy
def create_array(x,y):
    
    newArray = numpy.zeros([x,y])
    
    for i in range(0,x):
        for j in range(0,y):
            newArray[i,j] = int(i*j)
    
    return newArray
    
print(create_array(3,5))

'''
### Question 8
Level 2

Question:
Write a program that accepts a comma separated sequence of words as input and prints the words in a comma-separated sequence after sorting them alphabetically.
Suppose the following input is supplied to the program:
without,hello,bag,world
Then, the output should be:
bag,hello,without,world

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
'''


def arrange_words(wordsList):
    
    seperatedWords = wordsList.split(',')
    return sorted(seperatedWords))

print(arrange_words(input('Please input a sequence of comma-separated numbers')))


'''
### Question 9
Level 2

Question£º
Write a program that accepts sequence of lines as input and prints the lines after making all characters in the sentence capitalized.
Suppose the following input is supplied to the program:
Hello world
Practice makes perfect
Then, the output should be:
HELLO WORLD
PRACTICE MAKES PERFECT

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
'''


while True:
    singleLine = input()
    print(singleLine.upper() + '\n')


'''
### Question 10
Level 2

Question:
Write a program that accepts a sequence of whitespace separated words as input and prints the words after removing all duplicate words and sorting them alphanumerically.
Suppose the following input is supplied to the program:
hello world and practice makes perfect and hello world again
Then, the output should be:
again and hello makes perfect practice world

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
We use set container to remove duplicated data automatically and then use sorted() to sort the data.
'''


def arrange_words_again(wordsList):
    
    seperatedWords = wordsList.split(' ')
    return sorted(set(seperatedWords))

print(arrange_words_again('hello world and practice makes perfect and hello world again'))




'''
### Question 11
Level 2

Question:
Write a program which accepts a sequence of comma separated 4 digit binary numbers as its input and then check whether they are divisible by 5 or not. The numbers that are divisible by 5 are to be printed in a comma separated sequence.
Example:
0100,0011,1010,1001
Then the output should be:
1010
Notes: Assume the data is input by console.

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
'''

def is_divisible_by_5(numbersList):
    
    divisibleNumbers = []
    
    for number in numbersList:
        if int(number)%5==0:
            divisibleNumbers.append(number)
    
    return divisibleNumbers


print(is_divisible_by_5(input('Please input a sequence of comma separated 4 digit binary numbers')))
#print(is_divisible_by_5(['0100','0011','1010','1001']))


'''
### Question 12
Level 2

Question:
Write a program, which will find all such numbers between 1000 and 3000 (both included) such that each digit of the number is an even number.
The numbers obtained should be printed in a comma-separated sequence on a single line.

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
'''
divisibleNumbers = []

for num in range(1000,3001):
    if num%2 == 0:
        divisibleNumbers.append(num)
        
print(divisibleNumbers)


'''
### Question 13
Level 2

Question:
Write a program that accepts a sentence and calculate the number of letters and digits.
Suppose the following input is supplied to the program:
hello world! 123
Then, the output should be:
LETTERS 10
DIGITS 3

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
'''
import string
def calculate_sum_of_letters_and_digits(userSentence):
    
    counterLetters=0
    counterDigits=0
      
    for char in userSentence:
        if char in string.ascii_letters:
            counterLetters+=1
        if char in string.digits:
            counterDigits+=1
            
    print('LETTERS ' + str(counterLetters) + '\n DIGITS ' + str(counterDigits))
  
calculate_sum_of_letters_and_digits(input('Please input a sentence containning letters and digits'))       
#calculate_sum_of_letters_and_digits('hello world! 123')        

'''
### Question 14
Level 2

Question:
Write a program that accepts a sentence and calculate the number of upper case letters and lower case letters.
Suppose the following input is supplied to the program:
Hello world!
Then, the output should be:
UPPER CASE 1
LOWER CASE 9

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
'''

import string
def calculate_sum_of_upper_and_lower_case_letters(userSentence):
    
    counterUpper=0
    counterLower=0
      
    for char in userSentence:
        if char in string.ascii_uppercase:
            counterUpper+=1
        if char in string.ascii_lowercase:
            counterLower+=1
            
    print('UPPER CASE ' + str(counterUpper) + '\nLOWER CASE ' + str(counterLower))
  
calculate_sum_of_upper_and_lower_case_letters(input('Please input a sentence containning letters only'))       
#calculate_sum_of_upper_and_lower_case_letters('Hello world!')  



'''
### Question 17
Level 2

Question:
Write a program that computes the net amount of a bank account based a transaction log from console input. The transaction log format is shown as following:
D 100
W 200

D means deposit while W means withdrawal.
Suppose the following input is supplied to the program:
D 300
D 300
W 200
D 100
Then, the output should be:
500

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
'''
import string
def calculate_bank_status(userSentence):
    
    finalSum = 0
    
    x = 'D 300\nD 300\nW 200\nD 100'
    y = x.splitlines()
    
    for singleLine in y:
        
        z = singleLine.split(' ')
        
        if z[0] == 'D':
            finalSum += int(z[1])
            
        if z[0] == 'W':
            finalSum -= int(z[1])
    
    return finalSum
  
calculate_bank_status(input('Please input bank actions'))       
#print(calculate_bank_status('D 300\nD 300\nW 200\nD 100'))


'''
### Question 18
Level 3

Question:
A website requires the users to input username and password to register. Write a program to check the validity of password input by users.
Following are the criteria for checking the password:
1. At least 1 letter between [a-z]
2. At least 1 number between [0-9]
1. At least 1 letter between [A-Z]
3. At least 1 character from [$#@]
4. Minimum length of transaction password: 6
5. Maximum length of transaction password: 12
Your program should accept a sequence of comma separated passwords and will check them according to the above criteria. Passwords that match the criteria are to be printed, each separated by a comma.
Example
If the following passwords are given as input to the program:
ABd1234@1,a F1#,2w3E*,2We3345
Then, the output of the program should be:
ABd1234@1

Hints:
In case of input data being supplied to the question, it should be assumed to be a console input.
'''

import re
def check_password(inputPassword)


    charConditions = re.compile(r'[^a-zA-Z0-9\$\#\@]')
    if len(inputPassword) > 12 or len(inputPassword) < 6 or not bool(charConditions.search(inputPassword))
        return 
    
    return inputPassword


userNameInput = input('Please input user name\n'))    
passwordInput = input('Please input password\n'))    
check_password(passwordInput)                                
#check_password('ABd1234@1')    


'''
### Question 20
Level 3

Question:
Define a class with a generator which can iterate the numbers, which are divisible by 7, between a given range 0 and n.

Hints:
Consider use yield
'''

class Iterator:
  def __init__(self, rangeValue):
      
      self.divisibleSeven = []
      
      for num in range (0,rangeValue+1):
          if num % 7 == 0 :
              self.divisibleSeven.append(num)
              
newIterator = Iterator(int(input('Please input number\n')))
print(newIterator.divisibleSeven)    

  
          
          
          