print("number condition")
age = 21

if age >= 20:
    print("can vote")
else:
    print("can't vote")

print("string condition")
names = ["bonny", "jack", "rose"]
for name in names:
    if name == "bonny":
        print(name.upper())
    else:
        print(name)

print("multi condition")
a = 5
b = 6
c = 4

if a > c and b > a:
    print("Both conditions are True")

if a > c or b < a:
    print("At least one of the conditions are True")

print("if-else condition")
card = "gold"
if card == "gold":
    print("30% off")
elif card == "silver":
    print("20% off")
else:
    print("10% off")

print("array condition")
scores = []
if scores:
    for score in scores:
        print(score)
    print("completed!!")
else:
    print("list is empty")

letters = ["a", "c", "d"]
letter = "b"

if letter not in letters:
    print(letter + "not in list")
