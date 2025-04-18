name = input("please enter your name : ")
print("hello, " + name)

number = input("please enter a number : ")
number = int(number)

if number % 2 == 0:
    print("It's even")
else:
    print("It's odd")

text = "apple banana apple grape apple apple watermelon"
find = input("which word do you want to find ?")

print(text.count(find))