# -*- coding: utf-8 -*-



# Ofir Friedman



"""

1.
Define a function `max()` that takes two numbers as arguments and returns the largest of them. Use the if-then-else construct available in Python. (It is true thatPython has the `max()` function built in, but writing it yourself is nevertheless a goodexercise ).

"""

def max(numA, numB):
    
    if numA > numB:
        return numA
    
    return numB
    
    
    
"""

2. Define a function `max_of_three()` that takes three numbers as arguments and returns the largest of them.

"""

def max_of_three(numA, numB, numC):
    
    arrayofNumber= [numA, numB, numC]
    arrayofNumber.sort()
    return arrayofNumber[2]


"""

3.
Define a function that computes the length of a given list or string. ( It is true thatPython has the `len()` function built in, but writing it yourself is nevertheless a goodexercise ).

"""

def myLen(string):
    
    counterLen = 0
    
    for i in string:
        counterLen += 1
        
    return counterLen


"""

4.
Write a function that takes a character ( i.e. a string of length 1 ) and returns `True`if it is a vowel, `False` otherwise.

"""

def myLen(singleChar):
    
    vowels = ['a', 'e', 'i', 'o', 'u']
    return singleChar in vowels



"""
5.
Write a function `translate()` that will translate a text into "rövarspråket" (Swedishfor "robber's language"). That is, double every consonant and place an occurrence of"o" in between. For example, `translate("this is fun")` should return the string`"tothohisos isos fofunon".`
    
"""

def translate(string):  
    
    translatedString = []
    
    for i in string:
        if i != ' ':
            translatedString.append(i)
            translatedString.append('o')
        else:
            translatedString.append(' ')
            
    return translatedString
    
    
"""
6.
Define a function `sum()` and a function `multiply()` that sums and multiplies(respectively) all the numbers in a list of numbers. For example, `sum([1, 2, 3, 4])`should return `10`, and `multiply([1, 2, 3, 4])` should return `24`.
    
"""

def sum(listNumbers):
    
    result = 0
    
    for item in listNumbers:
        result += item
        
    return result
    
    
def multiply(listNumbers): 
    
     result = 1
    
    for item in listNumbers:
        result*=item
        
    return result   


"""
7.
Define a function `reverse()` that computes the reversal of a string. For example,`reverse( "I am testing" )` should return the string `"gnitset ma I"`.
    
"""

def reverse(myString):
    return myString[::-1]

    
"""
8.
Define a function `is_palindrome()` that recognizes palindromes (i.e. words thatlook the same written backwards). For example, `is_palindrome( "radar" )` shouldreturn `True`.
    
"""

def is_palindrome(myString):  
    
    mirorString = reverse(myString)
        
    return mirorString == myString
    

    
"""
9.
Write a function `is_member()` that takes a value ( i.e. a number, string, etc ) `x`and a list of values `a`, and returns True if `x` is a member of `a`, `False` otherwise.(Note that this is exactly what the in operator does, but for the sake of the exerciseyou should pretend Python did not have this operator).
    
"""

def is_member(inputValue, listValues):

    for i in listValues:
        if inputValue == i :
            return 'True'
        
    return 'False'
    
    
"""
10.
Define a function `overlapping()` that takes two lists and returns `True` if they have at least one member in common, `False` otherwise. You may use your`is_member()` function, or the in operator, but for the sake of the exercise, you should(also) write it using two nested for-loops.
    
"""

def overlapping(listA, listA):   
    
    
    #  write it using two nested for-loops (!)
    
    for i in listA:
       
        for j in listB: 
           
            if j == i:
                
                return 'True'
            
    return 'False'
        
  
"""
11.
Define a function `generate_n_chars()` that takes an integer `n` and a character`c` and returns a string, `n` characters long, consisting only of c:s. For example,`generate_n_chars( 5,"x" )` should return the string `"xxxxx"`. (Python is unusual inthat you can actually write an expression `5 * "x"` that will evaluate to `"xxxxx"`. Forthe sake of the exercise you should ignore that the problem can be solved in thismanner ).
    
"""

def generate_n_chars(n, c):    
    
    
    generatedChars = []
    
    for i in range (0,n):
        
        generatedChars.append(c)
        
    return generatedChars
    
    
"""
12.
Define a procedure `histogram()` that takes a list of integers and prints a histogram to the screen. For example, `histogram( [ 4, 9, 7 ] )` should print the following:
```
****
*********
*******
```
    
"""

def histogram(listIntegers):
    
    for i in range (0,len(listIntegers)):
        
        newLine = "*" * listIntegers[i]
        print(newLine)
        print("\n")



"""
13.
The function `max()` from exercise 1) and the function `max_of_three()` from exercise 2) will only work for two and three numbers, respectively. But suppose we have a much larger number of numbers, or suppose we cannot tell in advance how many they are? Write a function `max_in_list()` that takes a list of numbers andreturns the largest one.
"""

def max_in_list(listNumbers):

    listNumbers.sort(reverse=True)
    return listNumbers[0]
    
"""
14.
Write a program that maps a list of words into a list of integers representing the lengths of the corresponding words.
"""

def max_in_list(wordsList):
    
    lenList = []
    
    for i in range(0,len(wordsList)):
        
        lenList.append(len(wordsList[i]))
        
    return lenList    
    

    
"""
15.
Write a function `find_longest_word()` that takes a list of words and returns the
length of the longest one.
"""

def find_longest_word(wordsList):   
        
    longestLen = 0
    
    for i in range (0,len(wordsList)):
        
        if len(wordsList[i]) > longestLen:
            
            longestLen = len(wordsList[i])
        
    return longestLen


    
"""
16.
Write a function `filter_long_words()` that takes a list of words and an integer `n`
and returns the list of words that are longer than `n`.
"""

def filter_long_words(wordsList, n):  

    goodWords = []
    
    for word in wordsList:
        
        if len(word) > n:
            
            goodWords.append(word)
        
    return goodWords
    

    
"""
17.
Write a version of a palindrome recognizer that also accepts phrase palindromes such as "Go hang a salami I'm a lasagna hog.", "Was it a rat I saw?", "Step on no pets","Sit on a potato pan, Otis", "Lisa Bonet ate no basil", "Satan, oscillate my metallic
sonatas", "I roamed under it as a tired nude Maori", "Rise to vote sir", or the exclamation"Dammit, I'm mad!". Note that punctuation, capitalization, and spacing are usuallyignored.
"""
import re 

def inhanced_palindrome(myString):  
   
    
    clearString = re.sub('[^a-zA-Z]+', "", myString)
   
    return is_palindrome(clearString.lower())  
    

"""
18.
A pangram is a sentence that contains all the letters of the English alphabet at least once,
for example: "The quick brown fox jumps over the lazy dog". Your task here is to write a function
to check a sentence to see if it is a pangram or not.
"""
import string
def is_pangram(myString):
    
    clearString= re.sub('[^a-zA-Z]+', "", myString)

    for letter in string.ascii_lowercase:
        if letter not in clearString.lower():
            return False
    return True

"""
19.
"99 Bottles of Beer" is a traditional song in the United States and Canada. It is popular to
sing on long trips, as it has a very repetitive format which is easy to memorize, and can take a
long time to sing. The song's simple lyrics are as follows:
```99 bottles of beer on the wall, 99 bottles of beer.
Take one down, pass it around, 98 bottles of beer on the wall.```
The same verse is repeated, each time with one fewer bottle. The song is completed when the
singer or singers reach zero. Your task here is write a Python program capable of generating all
the verses of the song.
"""

def sing_99_bottle_of_beer(numOfBottles):

    bottleCounter = numOfBottles
    
    while bottleCounter >= 0:
        print(f'{bottleCounter} bottles of beer on the wall, {bottleCounter} bottles of beer.')
        bottleCounter-=1
        
"""
20.
Represent a small bilingual lexicon as a Python dictionary in the following fashion
```
{ "merry":"god", "christmas":"jul", "and":"och", "happy":gott", "new":"nytt", "year":"år" }
```
and use it to translate your Christmas cards from English into Swedish. That is, write a function
`translate()` that takes a list of English words and returns a list of Swedish words.
"""

def translate(englishWords):
    
    dicSwedish = {
        "merry":"god",
        "christmas":"jul",
        "and":"och",
        "happy":"gott",
        "new":"nytt",
        "year":"år" 
    }

    SwedishText = []

    for word in englishWords:
       SwedishText.append(dicSwedish[word])
        
    return SwedishText

"""
21.
Write a function `char_freq()` that takes a string and builds a frequency listing of the
characters contained in it. Represent the frequency listing as a Python dictionary. Try it with
something like `char_freq("abbabcbdbabdbdbabababcbcbab")`.
"""

def char_freq(userString):

    charDictionary = {}    

    sortedString = sorted(userString)
    
    for char in sortedString:
        
        if char in charDictionary:
            
            continue
        
        charCounter = 0
        
        for char2 in sortedString:
            if char2 == char:
                charCounter+=1
        
        charDictionary[char] = charCounter   
        
    
    return charDictionary

"""
22.
In cryptography, a Caesar cipher is a very simple encryption techniques in which each letter
in the plain text is replaced by a letter some fixed number of positions down the alphabet. For
example, with a shift of 3, A would be replaced by D, B would become E, and so on. The
method is named after Julius Caesar, who used it to communicate with his generals. `ROT-13
("rotate by 13 places")` is a widely used example of a Caesar cipher where the shift is 13. In
Python, the key for ROT-13 may be represented by means of the following dictionary:
```
key = {'a':'n', 'b':'o', 'c':'p', 'd':'q', 'e':'r', 'f':'s', 'g':'t', 'h':'u',
'i':'v', 'j':'w', 'k':'x', 'l':'y', 'm':'z', 'n':'a', 'o':'b', 'p':'c',
'q':'d', 'r':'e', 's':'f', 't':'g', 'u':'h', 'v':'i', 'w':'j', 'x':'k',
'y':'l', 'z':'m', 'A':'N', 'B':'O', 'C':'P', 'D':'Q', 'E':'R', 'F':'S',
'G':'T', 'H':'U', 'I':'V', 'J':'W', 'K':'X', 'L':'Y', 'M':'Z', 'N':'A',
'O':'B', 'P':'C', 'Q':'D', 'R':'E', 'S':'F', 'T':'G', 'U':'H', 'V':'I',
'W':'J', 'X':'K', 'Y':'L', 'Z':'M'}
```
Your task in this exercise is to implement an encoder/decoder of ROT-13. Once you're done,
you will be able to read the following secret message:
```
Pnrfne pvcure? V zhpu cersre Pnrfne fnynq!
```
Note that since English has 26 characters, your ROT-13 program will be able to both encode
and decode texts written in English.
"""
      
def rot_13(secretText):
    
    key = {'a':'n', 'b':'o', 'c':'p', 'd':'q', 'e':'r', 'f':'s', 'g':'t',
           'h':'u', 'i':'v', 'j':'w', 'k':'x', 'l':'y', 'm':'z', 'n':'a', 
           'o':'b', 'p':'c', 'q':'d', 'r':'e', 's':'f', 't':'g', 'u':'h', 
           'v':'i', 'w':'j', 'x':'k', 'y':'l', 'z':'m', 'A':'N', 'B':'O', 
           'C':'P', 'D':'Q', 'E':'R', 'F':'S', 'G':'T', 'H':'U', 'I':'V', 
           'J':'W', 'K':'X', 'L':'Y', 'M':'Z', 'N':'A', 'O':'B', 'P':'C', 
           'Q':'D', 'R':'E', 'S':'F', 'T':'G', 'U':'H', 'V':'I', 'W':'J', 
           'X':'K', 'Y':'L', 'Z':'M'}

    decipheredText = []
    
    for char in secretText:
        if char in key:
            decipheredText.append(key[char])
        else:
            decipheredText.append(char)
            
    return ''.join(decipheredText)
    
#Caesar cipher? I much prefer Caesar salad!


"""
23. 
Define a simple "spelling correction" function `correct()` that takes a string and sees to it that
1) two or more occurrences of the space character is compressed into one, and 2) inserts an
extra space after a period if the period is directly followed by a letter. E.g. `correct("This is very
funny and cool.Indeed!")` should return "This is very funny and cool. Indeed!" Tip: Use regular
expressions!
"""

import re
def correct(userString):

    noExtraSpacesString = re.sub(' +',' ',userString)
    addSpaceAfterDot =  re.sub(r'(?<=[.,])(?=[^\s])', r' ', noExtraSpacesString)   
    return addSpaceAfterDot


    

