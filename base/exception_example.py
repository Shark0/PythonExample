try:
    print(1 / 0)
except ZeroDivisionError:
    print("You can't divide by zero")

while True:
    first = input("First number: ")
    second = input("Second number: ")
    try:
        ans = int(first) / int(second)
    except ZeroDivisionError:
        print("You can't divide by zero")
    else:
        print(ans)
        break

filename = "alice.txt"

try:
    with open(filename) as file:
        contents = file.read()
except FileNotFoundError:
    msg = "Sorry, the file " + filename + " does not exist"
    print(msg)
else:
    find = input("find the word: ")
    print(contents.count(find))
