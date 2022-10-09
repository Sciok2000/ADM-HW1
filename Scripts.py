#PROBLEM 1:

#Introduction
#Say "Hello, World!" With Python:
if __name__ == '__main__':
    print("Hello, World!")

#Python If-Else:
#!/bin/python3
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())
if n % 2 ==0 and (n in range(2,5) or n > 20):
    print("Not Weird")
else:
    print("Weird")

#Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b) 
    print(a-b) 
    print(a*b)

#Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)

#Loops
if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i*i)

#Write a function
def is_leap(year):
    leap = False
    
    if year % 4 ==0:
        leap = True
    if year % 100 ==0:
        leap = False
    if year % 400 ==0:
        leap = True
    
    return leap

#Print Function
if __name__ == '__main__':
    n = int(input())
    for i in range(1,n+1):
        print(i,end="")

#Data types
#List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    
print([[i,j,k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if i+j+k != n])

#Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    myset = set(arr)
    l = list(myset)
    l.sort()
    print(l[-2])

#Nested Lists
if __name__ == '__main__':
    records = []
    grades = []
    for i in range(int(input())):
        name = input()
        grade = float(input())
        grades.append(grade)
        records.append([name,grade])
    grades.sort()
    no_duplicate_grades = list(set(grades))
    names = []
    for n in records:
        if n[1] == no_duplicate_grades[1]:
            names.append(n[0])
    names.sort()
    for output in names:
        print(output)

#Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    result = sum(student_marks[query_name])/len(student_marks[query_name])
    print(f"{result:.2f}")

#Lists
if __name__ == '__main__':
    N = int(input())
    l = []
    for i in range(N):
        inp = input().split()
        if inp[0] =="insert":
            l.insert(int(inp[1]),int(inp[2]))
        elif inp[0] =="remove":
            l.remove(int(inp[1]))
        elif inp[0] == "append":
            l.append(int(inp[1]))
        elif inp[0] == "sort":
            l.sort()
        elif inp[0] == "pop":
            l.pop(-1)
        elif inp[0] == "reverse":
            l.reverse()
        else:
            print(l)

#Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    tup = tuple(integer_list)
    print(hash(tup))
    
#Strings
#sWAP cASE
def swap_case(s):
    return "".join([char.lower() if char.isupper() else char.upper() for char in s])

#String Split and Join
def split_and_join(line):
    line = line.split(" ")
    line = "-".join(line)
    return line

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

#What's Your Name?
#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#

def print_full_name(first, last):
    # Write your code here
    return [print('Hello',first,last+"! You just delved into python.") if len(first) <=10 and len(last) <=10 else print('error')]

#Mutations
def mutate_string(string, position, character):
    l =list(string)
    l[position] = character
    string = ''.join(l)
    return string

#Find a string
def count_substring(string, sub_string):              
    return len([i for i in range(len(string)) if string[i:(i+len(sub_string))] == sub_string])

#String Validators
if __name__ == '__main__':
    s = input()
    alnum = False
    alpha = False
    digit = False
    lower = False
    upper = False
    for c in s:
        if c.isalnum():
            alnum = True
        if c.isalpha():
            alpha = True
        if c.isdigit():
            digit = True
        if c.islower():
            lower = True
        if c.isupper():
            upper = True
    print(alnum)
    print(alpha)
    print(digit)
    print(lower)
    print(upper)

#Text Alignment
# Enter your code here. Read input from STDIN. Print output to STDOUT
t = int(input())
h = 'H'
for i in range(t):
    print((h*i).rjust(t-1)+h+(h*i).ljust(t-1))

for i in range(t+1):
    print((h*t).center(t*2)+(h*t).center(t*6))

for i in range((t+1)//2):
    print((h*t*5).center(t*6))
    
for i in range(t+1):
    print((h*t).center(t*2)+(h*t).center(t*6))
    
for i in range(t):
     print(((h*(t-i-1)).rjust(t)+h+(h*(t-i-1)).ljust(t)).rjust(t*6))
    
#Text Wrap
def wrap(string, max_width):
    stri = ''
    wrapper = textwrap.TextWrapper(max_width)
    l = wrapper.wrap(text = string)
    for line in l:
        stri = stri+line+'\n'
    return stri

#Designer Door Mat
# Enter your code here. Read input from STDIN. Print output to STDOUT
N,M = list(map(int, input().split()))
num = 1
for i in range(N):
    if i< (int(N/2)):
        print((".|." *num).center(M,'-'))
        num +=2
    elif i == (int(N/2)):
        print(('WELCOME'.center(M, '-')))
    elif i > (int(N/2)):
        num -= 2
        print((('.|.' * num).center(M, '-')))
    
#String Formatting
def print_formatted(number):
    o = []
    h = []
    b = []
    x = len(bin(number)[2:])
    for i in range(1,number+1):
        o.append(oct(i))
        h.append(hex(i).upper())
        b.append(bin(i))
    for i in range(number):
        print(str(i+1).rjust(x),o[i][2:].rjust(x),h[i][2:].rjust(x),b[i][2:].rjust(x))

#Alphabet Rangoli
def print_rangoli(size):
    # your code goes here
    import string
    design = string.ascii_lowercase
    L = []
    for i in range(n):
        s = "-".join(design[i:n])
        L.append((s[::-1]+s[1:]).center(4*n-3, "-"))
        
    print('\n'.join(L[:0:-1]+L))

#Capitalize!
# Complete the solve function below.
def solve(s):
    for ph in s.split():
        phrase = ph
        ph=ph.capitalize()
        s=s.replace(phrase,ph)
    return s

#The Minion Game
def minion_game(string):
    # your code goes here
    player_1=0
    player_2=0
    vowels = ['A','E','I','O','U']
    for i in range(len(string)):
        if string[i] in vowels:
            player_1 +=len(string)-i
        else:
            player_2 +=len(string)-i
    if player_1 > player_2:
        print("Kevin", player_1)
    elif player_2 > player_1:
        print("Stuart", player_2)
    else:
        print("Draw")
        
#Merge the Tools!
def merge_the_tools(string, k):
    # your code goes here
    count = 0 
    for i in string: 
        count+=1 
        if count % k == 0: 
            print("".join(set(string[count-k:count])))

#Sets
#Introduction to Sets
def average(array):
    # your code goes here
    sett = set(array)
    tot = 0
    for i in sett:
        tot = tot+i
    return float(tot/len(sett))
        
#No Idea!
# Enter your code here. Read input from STDIN. Print output to STDOUT
n, m = map(int, input().split())
array = list(map(int, input().split()))
a = set(map(int, input().split()))
b = set(map(int, input().split()))
happiness = 0
for x in array:
    if x in a:
        happiness +=1
    if x in b:
        happiness -=1
print(happiness)
        
#Symmetric Difference
# Enter your code here. Read input from STDIN. Print output to STDOUT
M = int(input())
a = set([int(x) for x in input().split()])
N = int(input())
b = set([int(x) for x in input().split()])

result = sorted((a.difference(b)).union(b.difference(a)))
for i in result:
    print(i)

#Set .add()
# Enter your code here. Read input from STDIN. Print output to STDOUT
s = set()
for _ in range(int(input())):
    s.add(input())
print(len(s))
    
#Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))

N = int(input())
for _ in range(N):
    commands = input().split()
    if(len(commands)==1):
        s.pop()
    else:
        commands="s."+commands[0]+"("+commands[1]+")"
        eval(commands)
        
print(sum(list(s)))

#Set .union() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
A = set(input().split())
m = int(input())
B = set(input().split())
print(len(A.union(B)))

#Set .intersection() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
A = set(input().split())
m = int(input())
B = set(input().split())
print(len(A.intersection(B)))

#Set .difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
A = set(input().split())
m = int(input())
B = set(input().split())
print(len(A.difference(B)))

#Set .symmetric_difference() Operation
# Enter your code here. Read input from STDIN. Print output to STDOUT
n = int(input())
A = set(input().split())
m = int(input())
B = set(input().split())
print(len(A.symmetric_difference(B)))

#Set Mutations
# Enter your code here. Read input from STDIN. Print output to STDOUT
a=int(input()) 
a1=set(map(int,input().split())) 
n=int(input()) 
for _ in range(n): 
    cmd,t=input().split() 
    Set=set(map(int,input().split())) 
    if cmd=="update": 
        a1.update(Set) 
    elif cmd=="intersection_update": 
        a1.intersection_update(Set) 
    elif cmd=="difference_update": 
        a1.difference_update(Set) 
    elif cmd=="symmetric_difference_update":
        a1.symmetric_difference_update(Set) 
print(sum(a1))

#The Captain's Room
# Enter your code here. Read input from STDIN. Print output to STDOUT
K = int(input())
rooms_list = list(map(int,input().split()))
rooms_set = set(rooms_list)
for _ in list(rooms_set):
    rooms_list.remove(_)
captain_room = rooms_set.difference(set(rooms_list))
c_r=list(map(int,captain_room))
print(c_r[0])

#Check Subset
# Enter your code here. Read input from STDIN. Print output to STDOUT
T = int(input())
for i in range(T):
    a = int(input())
    A = set(list(map(int, input().split())))
    b = int(input())
    B = set(list(map(int, input().split())))
    print(A.issubset(B))

#Check Strict Superset
# Enter your code here. Read input from STDIN. Print output to STDOUT
set_A = set(map(int, input().split()))
N = int(input())
list_sets = []
for _ in range (N):
    list_sets.append(set(map(int,input().split())))
print(all(set_A.issuperset(s) for s in list_sets)) 

#Collections
#collections.Counter()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import Counter
X = int(input())
shoe_sizes = Counter(map(int,input().split()))
N = int(input())
total=0
for _ in range(N):
    size, value = tuple(map(int,input().split()))
    if shoe_sizes[size] > 0:
        shoe_sizes.subtract({size})
        total+=value

print(total)

#DefaultDict Tutorial
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import defaultdict 
A = defaultdict(list)

n, m = map(int, input().split())

for i in range(1, n+1):
    A[input()].append(i)
    
for i in range(m):
    b = input()
    
    if b in A.keys():
        print(*A[b])
    else:
        print(-1)

#Collections.namedtuple()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import namedtuple
N=int(input())
Student=namedtuple('Student',input().rsplit())
print(sum([int(Student(*input().rsplit()).MARKS) for _ in range(N)])/N)

#Collections.OrderedDict()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import OrderedDict
ordered_dict = {}
for _ in range(int(input())):
    my_list = input().split()
    price = int(my_list[-1])
    key = ' '.join(map(str, my_list[:-1]))
    if key in ordered_dict:
        ordered_dict[key] += price
    else:
        ordered_dict[key] = price
for key, value in ordered_dict.items(): 
    print(key, value)

#Word Order
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import OrderedDict
od = OrderedDict()
n = int(input())
for _ in range(n):
    arg = input()
    od[arg] = 1 + int(od.get(arg, 0))
print(len(od))
for item in od:
    print(od[item], end=' ')

#Collections.deque()
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque 
d = deque() 
for _ in range(int(input())):
    arg = input().split()
    if len(arg) == 1: 
        func = 'd.' + arg[0] + '(' + ')' 
        eval(func) 
    else:
        func = 'd.' + arg[0] + '(' + arg[1] + ')' 
        eval(func) 
print(*d, sep=' ')

#Company Logo
#!/bin/python3
import math
import os
import random
import re
import sys
from collections import Counter

if __name__ == '__main__':
    s = sorted(input())
    count_dict = Counter(s)
    for k, v in count_dict.most_common(3):
        print(k,v)

#Piling Up!
# Enter your code here. Read input from STDIN. Print output to STDOUT
from collections import deque
T = int(input())

for _ in range(T):
    input() #not used
    cubes = deque(map(int,input().split()))
    ans = "Yes"
   
    while len(cubes) >= 3:
        left, right = cubes.popleft(), cubes.pop()
        if not ((left >= cubes[0] or left >= cubes[-1]) or (right >= cubes[0] or right >= cubes[-1])):
            ans = "No"
            break
    
    print(ans)

#Date and Time
#Calendar Module
# Enter your code here. Read input from STDIN. Print output to STDOUT
import calendar
month, day, year = map(int,input().split())
print(calendar.day_name[calendar.weekday(year,month,day)].upper())

#Time Delta
#!/bin/python3

import math
import os
import random
import re
import sys
from datetime import datetime

# Complete the time_delta function below.
def time_delta(t1, t2):
    time_format = '%a %d %b %Y %H:%M:%S %z'
    t1 = datetime.strptime(t1, time_format)
    t2 = datetime.strptime(t2, time_format)
    return str(int(abs((t1-t2).total_seconds()))) 

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()

#Exceptions
# Enter your code here. Read input from STDIN. Print output to STDOUT
for i in range(int(input())):
    try:
        a,b=map(int,input().split())
        print(a//b)
    except Exception as e:
        print("Error Code:",e)

#Built-ins
#Zipped!
# Enter your code here. Read input from STDIN. Print output to STDOUT
N, X = map(int, input().split())
S = [list(map(float, input().split())) for _ in range(X)]
D = [sum(G)/X for G in zip(*S)]
print(*D, sep='\n')

#Athlete Sort
#!/bin/python3
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    sort_arr = sorted(arr, key=lambda x: x[k])
for i in sort_arr:
    for j in i:
        print(j, end=' ')
    print() 

#ginortS
# Enter your code here. Read input from STDIN. Print output to STDOUT
S=list(input())

numbers=[]
lowers=[]
uppers=[]
for i in S:
    if i.isdigit():
        numbers.append(i)
    else:
        if i.isupper():
            uppers.append(i)
        else:
            lowers.append(i)

a=sorted(list(map(int,numbers)))
b=list(map(str,sorted(a,key=lambda x:x%2==0)))

print(''.join(sorted(lowers))+''.join(sorted(uppers))+''.join(b))
        
#Python Functionals
#Map and Lambda Function
cube = lambda x: x**3 # complete the lambda function 

def fibonacci(n):
    # return a list of fibonacci numbers
    lst = [0,1]
    if n>2:
        for i in range(2, n):
            lst.append(lst[i-1]+lst[i-2])
    else:
        lst = lst[0:n]
    return lst 

#Regex and Parsing challenges
#Detect Floating Point Number
# Enter your code here. Read input from STDIN. Print output to STDOUT
for _ in range(int(input())):
    string = list(map(str,input().split()))
    for char in string:
        if '.' in char:
            try:
                float(char)
                print(True)
            except:
                print(False)
        else:
            print(False)
    
#Re.split()
regex_pattern = r"\D"	# Do not delete 'r'.

#Group(), Groups() & Groupdict()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
match = re.search(r"([a-zA-Z0-9])\1+", input())
print(match.group()[0] if match else -1)

#Re.findall() & Re.finditer()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
string = input()
pattern = re.finditer(r'(?<=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])([AEIOUaeiou]{2,})(?=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])', string)
match = [i for i in map(lambda x: x.group(), pattern)]
print(*match, sep='\n') if match != [] else print(-1)

#Re.start() & Re.end()
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
S, k = input(), input()
matches = re.finditer(r'(?=(' + k + '))', S)
anymatch = False
for match in matches:
    anymatch = True
    print((match.start(1), match.end(1) - 1))
if anymatch == False:
    print((-1, -1))

#Regex Substitution
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
for i in range(int(input())):
    s = input()
    s1 = re.sub(r"(?<= )(&&)(?= )", "and", s)
    print(re.sub(r"(?<= )(\|\|)(?= )", "or", s1))

#Validating Roman Numerals
regex_pattern = r""	
thousand = "M{0,3}" 
hundred = "(D?C{0,3}|C[DM])"
ten = "(L?X{0,3}|X[LC])"
digit = "(V?I{0,3}|I[VX])"
regex_pattern = r"%s%s%s%s$" % (thousand, hundred, ten, digit)

#Validating phone numbers
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re


pattern= r"^[789][0-9]{9}$"
for _ in range(int(input())):
    if re.search(pattern, input()):
        print('YES')
    else:
        print('NO')

#Validating and Parsing Email Addresses
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
N = int(input())
for i in range(N):
    name, email = input().split()
    pattern="<[a-z][a-zA-Z0-9\-\.\_]+@[a-zA-Z]+\.[a-zA-Z]{1,3}>"
    if bool(re.match(pattern, email)):
        print(name,email)

#Hex Color Code
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re
x = int(input())
t = [input().strip() for i in range(x)]
p = "\{[^\{\}]+\}"
s = "\#[0-9ABCDEF]{6}|\#[0-9ABCDEF]{3}"
m = re.findall(p,"'"+("".join(t))+"'")
for i in m:
    l = re.findall(s,i,flags=re.I)
    if len(l) >0:
        print(*l,sep='\n')
    else:
        continue

#HTML Parser - Part 1
# Enter your code here. Read input from STDIN. Print output to STDOUT
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(f"Start : {tag}")
        for name,value in attrs:
            print(f"-> {name} > {value}")

    def handle_startendtag(self, tag, attrs):
        print(f"Empty : {tag}")
        for name,value in attrs:
            print(f"-> {name} > {value}")

    def handle_endtag(self, tag):
        print(f"End   : {tag}")


parser = MyHTMLParser()
for _ in range(int(input())):
    parser.feed(input())
parser.close()

#HTML Parser - Part 2
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if data != '\n':
            if "\n" in data:
                print(">>> Multi-line Comment")
                print(data)
            else:
                print(">>> Single-line Comment")
                print(data)
    def handle_data(self, data):
        if not data == '\n':
            print(f">>> Data")
            print(data)
            
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

#Detect HTML Tags, Attributes and Attribute Values
# Enter your code here. Read input from STDIN. Print output to STDOUT
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_starttag(self,tag,attrs):
        print(tag)
        for ele in attrs:
            print('->',ele[0],'>',ele[1])
   
 
parser=MyHTMLParser()
for _ in range(int(input())):
    parser.feed(input())

#Validating UID
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re

T=int(input())

for _ in range(T):
    UID=input()
    if (UID.isalnum()) and (len(set(UID)) == 10) and \
        (re.match('(.*[A-Z].*){2}', UID) != None) and \
        (re.match('(.*[0-9].*){3}', UID) != None):
        print("Valid")
    else:
        print("Invalid")

#Validating Credit Card Numbers
# Enter your code here. Read input from STDIN. Print output to STDOUT
import re

n=int(input())
for _ in range(n):
    cc=input()
    if (re.fullmatch(r"^[456]\d{3}(-?\d{4}){3}$", cc) and \
         not re.search(r"([0-9])(-?\1){3}", cc)):
        print("Valid")
    else:
        print("Invalid")

#Validating Postal Codes
regex_integer_in_range = r"^[1-9]{1}\d{5}$"    # Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(\d)(?=(\d{1})\1)"    # Do not delete 'r'.

#Matrix Script
#!/bin/python3

import math
import os
import random
import re
import sys




first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)
    a = list(zip(*matrix))
string = ""
for i in range(len(a)):
    string += "".join(a[i])
pattern = re.compile(r"(?<=\w)[!@#$%& ]{1,}(?=\s*\w)")
new_string = re.sub(pattern," ",string)
print(new_string)

#XML
#XML 1 - Find the Score
def get_attr_number(node):
    return sum([len(i.keys()) for i in  node.iter() ])

#XML2 - Find the Maximum Depth


maxdepth = 0
def depth(elem, level):
    global maxdepth
    # your code goes here
    if (level == maxdepth):
        maxdepth += 1
    for child in elem:
        depth(child, level + 1)

#Clousers and Decorations
#Standardize Mobile Number Using Decorators
def wrapper(f):
    
    def fun(l):
        # complete the function
        f([f"+91 {i[-10:-5]} {i[-5:]}" for i in l])
    return fun

#Decorators 2 - Name Directory
def person_lister(f):
    def inner(people):
        # complete the function
        people.sort(key=lambda x: int(x[2]))
        return [f(person) for person in people]
    return inner

#Numpy
#Arrays
def arrays(arr):
    # complete this function
    # use numpy.array
    a = numpy.array(arr,float)
    reverse = a[::-1]
    return reverse

#Shape and Reshape
import numpy
l = list(map(int,input().split(' ')))
l = numpy.array(l)
print(numpy.reshape(l,(3,3)))

#Transpose and Flatten
import numpy

N, M = list(map(int, input().split()))
matrix = numpy.array([list(map(int, input().split())) for _ in range(N)])
print(numpy.transpose(matrix))
print(matrix.flatten())

#Concatenate
import numpy as np

M, N, P = map(int, input().split())   
arr1 = np.array([list(map(int, input().split())) for _ in range(M)])
arr2 = np.array([list(map(int, input().split())) for _ in range(N)])
print(np.concatenate((arr1, arr2)))

#Zeros and Ones
import numpy

shape = list(map(int,input().split()))
print(numpy.zeros((shape), dtype=numpy.int))
print(numpy.ones((shape), dtype = numpy.int))

#Eye and Identity
import numpy as np
np.set_printoptions(legacy='1.13')
N,M = map(int, input().split(' '))
print(np.identity(N)) if N==M else print(np.eye(N,M))

#Array Mathematics
import numpy as np
N,M = map(int,input().split())
Narray = np.array([input().split() for _ in range(N)],int)
Marray = np.array([input().split() for _ in range(N)],int)
print(np.add(Narray,Marray))
print(np.subtract(Narray,Marray))
print(np.multiply(Narray,Marray))
print(Narray//Marray)
print(np.mod(Narray,Marray))
print(np.power(Narray,Marray))

#Floor, Ceil and Rint
import numpy
numpy.set_printoptions(legacy='1.13')
l= numpy.array(list(map(float,input().split())))
print(numpy.floor(l), numpy.ceil(l),numpy.rint(l),sep="\n")

#Sum and Prod
import numpy as np

a=[]
N, M = map(int, input().split())
for _ in range(N):
    a.append(np.array(list(map(int, input().split()))))
print(np.prod(np.sum(a, axis=0)))

#Min and Max
import numpy as np
n, m = map(int, input().split())
a = np.array([list(map(int, input().split())) for _ in range (n)], np.int64)
print(np.max(np.min(a, axis=1)))

#Mean, Var, and Std
import numpy
N,M = map(int,input().split())
a = []
for _ in range(N):
    a.append(list(map(int, input().split())))
a = numpy.array(a)
print(numpy.mean(a, axis = 1))
print(numpy.var(a, axis = 0))
print(round(numpy.std(a), 11))

#Dot and Cross
import numpy
a=int(input())
arr1=numpy.array([list(map(int,input().split())) for _ in range(a)])
arr2=numpy.array([list(map(int,input().split())) for _ in range(a)])
print(numpy.dot(arr1,arr2))

#Inner and Outer
import numpy as np
A = np.array(list(map(int,input().split())))
B = np.array(list(map(int,input().split())))
print(np.inner(A,B))
print(np.outer(A,B))

#Polynomials
import numpy as np
P = list(map(float, input().split()))
x = float(input())
print(float(np.polyval(P, x)))

#Linear Algebra
import numpy
l=[]
j=int(input())
for i in range(j ):
    s=input().split()
    for d in range(len(s)):
        s[d]=float(s[d])
    l.append(s)        
print(round(float(numpy.linalg.det(l)),2))


#PROBLEM 2:

#Birthday Cake Candles
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    # Write your code here
    candles.sort()
    count = 0
    tallest = candles[-1]
    for i in candles:
        if i == tallest:
            count += 1
    return count
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

#Number Line Jumps
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2):
    # Write your code here
    boolean = 'NO'
    for _ in range(10000):
        if x1 == x2:
            boolean = 'YES'
        x1 +=v1
        x2 +=v2
    return boolean

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

#Viral Advertising
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    # Write your code here
    likes = [2]
    for i in range(n-1):
        likes.append(3*(likes[-1])//2)
    return sum(likes)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

#Recursive Digit Sum
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#

def superDigit(n, k):
    # Write your code here
    if(int(n)<=9):
        if((int(n)*k)<=9):
            return(int(n)*k)
            pass
        else:
            return superDigit(int(n)*k,1)
    temp=str(n)
    s=0
    for a in temp:
        s+=int(a)
    return superDigit(s,k)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

#Insertion Sort - Part 1
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort1(n, arr):
    # Write your code here
    j = n-1
    store = arr[j]
    for i in range(j, -1, -1):
        if store < arr[i-1] and i >= 1:
            arr[i] = arr[i-1]
            print(' '.join(str(x) for x in arr))
        else: 
            arr[i] = store
            print(' '.join(str(x) for x in arr))
            break

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

#Insertion Sort - Part 2
#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort2(n, arr):
    # Write your code here
    for i in range(1,len(arr)):
        temp=arr[i]
        j=i-1
        while j>=0 and temp<arr[j]:
            arr[j+1]=arr[j]
            j=j-1
        arr[j+1]=temp 
        print(*arr)

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)



