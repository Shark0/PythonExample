## for-loop
print('for loop example')
students = ["bonny", "jack", "rose"]
for student in students:
    print(student)

## for range
print('for range example')
for number in range(1, 7):
    print(number)

odd_numbers = list(range(1, 10, 2))
print(odd_numbers)

print('for sub example')
colors = ["red", "orange", "yellow", "green", "blue"]
print(colors[0:3])
print(colors[1:4])
print(colors[:2])
print(colors[2:])
print(colors[-2:])

print('for slip example')
colors=["red","orange","yellow","green","blue"]
for color in colors[1:3] : # 印出colors串列中索引足標1和2的元素
    print(color)