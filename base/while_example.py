number = 1
while number < 6:
    print(number)
    number = number + 1

number = 0
while number < 10:
    number += 1
    if number % 3 != 0:
        continue
    print(number)

order = ["hamburger", "french fries", "cola"]
already_cooked = []

while order:
    cooking = order.pop()
    print("cooking: " + cooking)
    already_cooked.append(cooking)

print("left order: " + str(order))

print("Finished: ")
for cooked in already_cooked:
    print(cooked)

text = ["apple", "banana", "apple", "grape", "apple", "apple", "watermelon"]
while "apple" in text:
    text.remove("apple")
print(text)

response = {}  # 建立一個空字典

while True:
    name = input("What's your name?")
    place = input("Where would you want to go?")
    response[name] = place

    continue_prompt = input("Would you like to let someone else respond? (yes/no)")
    if continue_prompt == "no":
        break

for name, place in response.items():
    print(name + " would like to go " + place)
