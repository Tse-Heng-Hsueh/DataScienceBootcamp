#1. Write a function  count_vowels(word) that takes a word as an argument and returns the number of vowels in the word
def count_vowels(word):
    vowels = "aeiouAEIOU"
    count = 0
    for i in word:
        if i in vowels:
            count += 1
    return count
#2. Iterate through the following list of animals and print each one in all caps
animals=['tiger', 'elephant', 'monkey', 'zebra', 'panther']
for i in animals:
    print(i.upper())

#3. Write a program that iterates from 1 to 20, printing each number and whether it's odd or even.
for i in range(1,21):
    if i%2==0:
        print(str(i)+"is an even number")
    else:
        print(str(i)+"is an odd number")
        

#4. Write a function sum_of_integers(a, b) that takes two integers as input from the user and returns their sum.
def sum_of_integers(a, b):
    return int(a)+int(b)
