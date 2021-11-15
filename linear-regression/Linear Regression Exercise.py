import numpy as np
import random
import matplotlib.pyplot as plt

'''
בנו מערך שמכיל עשרה מספרים שלמים רנדומליים. .a
'''


random_int_array = np.random.randint(1,50,10)


'''
מערך הזה יהיה .x שברים( ושמרו אותו במשתנה ( floats בנו מערך שמכיל עשרה .b
הבסיס לבניית שלושת הדאטהסטים בהמשך.
'''


random_fractions_float_array = np.random.uniform(1,20,10) / np.random.uniform(21,100,10)


'''
בנו וקטור של חמישה מספרים אקראיים שהם כפולה של שלוש. .c
'''


random_devidable_by_three_array = np.random.randint(1,20,5) * 3

'''
שאלת אתגר: בחרו באקראיות מספר ששייך לעשרת המספרים הראשונים של סדרת .d
פיבונאצ'י.
'''

def create_fibonacci_array(array_lenght):
    
    fibonacci_array = [1,1]
    i = 1
    
    while i < array_lenght-1:
        fibonacci_array.append(fibonacci_array[i]+fibonacci_array[i-1])
        i += 1                                           
              
    return fibonacci_array
                                                                       
random_fibonacci_value = random.choice(create_fibonacci_array(10))


'''
e. נבחר בשיפוע אקראי עבור הקו הישר, לדוגמא2 . ניקח10  נקודות אקראיות על ציר ה
x, )הנקודות מסעיף1b (, ונגדיר אותן כערכיx .
                     
'''


random_slope_power1 = np.random.randint(1,20,1)


'''
f. כדי לחשב את ערכיy , נציב במשוואה של קו ישר, כלומר, נכפיל בשיפוע שבחרנו.
התוצאה נשמור במשתנה בשם1 _y_line.
'''

y_line_1 = random_slope_power1 * random_int_array


'''
g. נוסיף רעש גאוסיאני לכל אחת מנקודות הy בעזרת הפונקציה )(random.normal ש
numpy שיודעת לייצר נקודות אקראיות מתוך ההתפלגות הגאוסיאנית:                                                         
'''


y_line_1 += np.int32(np.random.normal(loc=0.0, scale=10.0, size=10))


'''
𝑦 = 𝑎 · 𝑥 + 𝑏
h. נבחר קבוע אחר שהוא השיפוע של קו זה, ונכפיל בו את ערכיx  ונוסיף לו קבוע אחר
שהוא ה-b )סקאלר(. את התוצאה נשמור במשתנה בשם2 _y_line ונוסיף לו רעש גאוסיאני.
'''


random_free_value = np.random.randint(1,20,1)
y_line_2 = random_slope_power1 * random_int_array + random_free_value
y_line_2 += np.int32(np.random.normal(loc=0.0, scale=10.0, size=10))


'''
𝑦 = 𝑎 · 𝑥2 + 𝑏 · 𝑥 + 𝑐

i. עבור משוואת פרבולה יש לבחור שלושה קבועים אקראיים )a,b,c(, ולהציב את ערכי
ה-x ממקודם בנוסחה כדי לקבל את ערכי הy  התואמים ושימרו במשתנה3 _y_line
ונוסיף גם לו רעש גאוסיאני.
'''


random_slope_power2 = np.random.randint(1,20,1)
y_line_3 = random_slope_power2 * random_int_array**2 + random_slope_power1 * random_int_array + random_free_value
y_line_3 += np.int32(np.random.normal(loc=0.0, scale=100.0, size=10))


'''
linear regression calculation
h = (((x**t)*x)**-1)*((x**t)*y)
x: matrix of [n, features] x [m, amount of factors]
y: results values
h: outcome value for new coefficients
'''

'''
𝑦 = 𝑎 · 𝑥
'''


x_set1 = np.reshape(random_int_array, (len(random_int_array),1))

var1 = np.linalg.inv(x_set1.T @ x_set1)   # (((x**t)*x)**-1)
var2 = x_set1.T @ y_line_1   #((x**t)*y)
h_1 = var1 * var2
print('The coefficient of a linear equation passing through (0,0):', h_1)


'''
𝑦 = 𝑎 · 𝑥 + 𝑏

ונוסיף לו קבוע אחר x נבחר קבוע אחר שהוא השיפוע של קו זה, ונכפיל בו את ערכי .h
ונוסיף לו רעש y_line_2 סקאלר(. את התוצאה נשמור במשתנה בשם ( b- שהוא ה
גאוסיאני.
'''


x_once = np.ones((len(random_int_array),1))
x_set2 = np.column_stack((x_set1, x_once))

var1 = np.linalg.inv(x_set2.T @ x_set2)   # (((x**t)*x)**-1)
var2 = x_set2.T @ y_line_2   #((x**t)*y)
h_2 = var1 @ var2
print('The coefficients of a linear equation passing through (0,{}): {}' .format(random_free_value, h_2))


'''
𝑦 = 𝑎 · 𝑥2 + 𝑏 · 𝑥 + 𝑐

עשו התאמה של הנקודות בדאטה-סט זה לפרבולה. רמז: הדרך לעשות את זה היא
מחברים בין כל .ones ועמודה של 𝑥2 ערכי ,x לייצר שלוש עמודות של פיצ'רים: ערכי
.np.column_stack העמודות למטריצה בעזר
'''


x_set3 = np.column_stack((x_set1**2, x_set2))

var1 = np.linalg.inv(x_set3.T @ x_set3)   # (((x**t)*x)**-1)
var2 = x_set3.T @ y_line_3   #((x**t)*y)
h_3 = var1 @ var2
print('The coefficients of a parabola equation passing through (0,{}): {}' .format(random_free_value, h_3))


################
### Plotting ###
################

x_set_non_random = np.arange(0,40,1)
y_set_non_random = (h_1 * x_set_non_random).T

figure, axis = plt.subplots(1, 1)

plt.scatter(x_set1,y_line_1)
plt.plot(x_set_non_random, y_set_non_random, color = 'r')

plt.show()



figure, axis = plt.subplots(1, 1)

y_set_non_random = (h_2[0] * x_set_non_random + h_2[1]).T

figure, axis = plt.subplots(1, 1)

plt.scatter(x_set1,y_line_2)
plt.plot(x_set_non_random, y_set_non_random, color = 'r')

plt.show()


figure, axis = plt.subplots(1, 1)

y_set_non_random = (h_3[0] * x_set_non_random**2 + h_3[1] * x_set_non_random + h_3[2]).T

plt.scatter(x_set1,y_line_3)
plt.plot(x_set_non_random, y_set_non_random, color = 'r')

plt.show()



#################
### Challange ###
#################


'''
equation:
𝑦 = 𝑎 · 𝑒 (𝑏 𝑥**2 + 𝑐 𝑥)
  
ln(y) = ln(a) * 𝑏 𝑥**2 + 𝑐 𝑥

'''

x=np.array([0.08750722,0.01433097,0.30701415,0.35099786,0.80772547,0.16525226,
            0.46913072,0.69021229,0.84444625,0.2393042,0.37570761,0.28601187,0.26468939,
            0.54419358,0.89099501,0.9591165,0.9496439,0.82249202,0.99367066,0.50628823])
y=np.array([4.43317755,4.05940367,6.56546859,7.26952699,33.07774456,4.98365345,
            9.93031648,20.68259753,38.74181668,5.69809299,7.72386118,6.27084933,
            5.99607266,12.46321171,47.70487443,65.70793999,62.7767844,
            35.22558438,77.84563303,11.08106882])


x_lan_random = np.random.uniform(0.01,1,20)

x_lan_set_a = np.ones(20)
x_lan_set_ab = np.column_stack((x_lan_random**2, x_lan_set_a))
x_lan_set_abc = np.column_stack((x_lan_random, x_lan_set_ab))

y_lan = np.log(y)

var1 = np.linalg.inv(x_lan_set_abc.T @ x_lan_set_abc)   # (((x**t)*x)**-1)
var2 = x_lan_set_abc.T @ y_lan   #((x**t)*y)
h_lan = var1 @ var2


x_set_non_random = np.arange(0,1,0.1)
y_set_non_random = (h_lan[2] * np.exp((h_lan[1]*x_set_non_random**2)+h_lan[1]*x_set_non_random)).T


figure, axis = plt.subplots(1, 1)

plt.scatter(x,y)
plt.plot(x_set_non_random, y_set_non_random, color = 'r')

plt.show()