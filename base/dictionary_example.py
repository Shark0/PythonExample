car = {"color": "black", "brand": "Toyota"}
color = car["color"]
print("The car's color is " + color)

car["mileage"] = 10
print(car)

car["color"] = "white"
print(car)

del car["color"]
print(car)

for key, value in car.items():
    print("key : " + key + ", value : " + str(value))

for key in car.keys():
    print("key : " + key)

for value in car.values():
    print("value : " + str(value))

favorite_fruits = {"bonny": "apple", "jack": "banana", "rose": "grape", "steven": "apple"}
for key in sorted(favorite_fruits.keys()):
    print(key)

for value in set(favorite_fruits.values()):
    print(value)

pets = {
    "dog": {
        "name": "Tony",
        "age": 4,
        "gender": "male",
    },
    "cat": {
        "name": "Kate",
        "age": 3,
        "gender": "female",
    },
}

for pet, pet_info in pets.items():
    print("I have a " + pet + ", it's name is " + pet_info["name"] + ", " + str(
        pet_info["age"]) + " years old,and gender is " + pet_info["gender"])

colors = {"orange": ["red", "yellow"], "green": ["blue", "yellow"]}
for keys, values in colors.items():
    print("If you want to get color " + keys)
    print("you need to add :")
    for value in values:
        print(value)
