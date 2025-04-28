def hello_world():
    print("hello world")


hello_world()


def greet(name):
    print("hello, " + name)


greet("Shark")


def pets(pet_name, pet_type):
    print("I have a " + pet_type + " and its name is " + pet_name)


pets("dog", "tony")  # 將dog引數放入pet_type參數中、tony引數放入pet_name參數中
pets("cat", "candy")

pets(pet_type="dog", pet_name="tony")
pets(pet_name="candy", pet_type="cat")


def get_location(city, area, zipcode):
    location = city + " " + area + " " + zipcode
    return location


return_value = get_location("新北市", "新莊區", "242")
print(return_value)


def get_name(first_name, last_name):
    name = {"first": first_name, "last": last_name}
    return name


return_value = get_name("bonny", "chang")
print(return_value)

while True:
    print("enter 'q' at any time to quit")
    first = input("enter your first name :")
    if first == "q":  # 當使用者輸入"q"則跳出迴圈
        break
    last = input("enter your last name :")
    if last == "q":
        break

    print(get_name(first, last))


def get_name(*names):  # *names參數中的星號會讓python建立一個名字為names的空多元組
    for name in names:
        print("hello, " + name)


get_name("bonny", "steven")
get_name("jack")
get_name("rose", "john", "jane")
