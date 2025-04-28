class Dog():
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def eat(self):
        print(self.name + " is eating")

    def sleep(self):
        print(self.name + " is sleeping")


dog1 = Dog("Tony", 3)
print("My dog's name is " + dog1.name + " and it's " + str(dog1.age) + " years old.")
dog1.eat()
dog1.sleep()


class Car():
    def __init__(self, year, brand, color):
        self.year = year
        self.brand = brand
        self.color = color
        self.miles = 0

    def get_name(self):
        print(str(self.year) + " " + self.brand)

    def get_mile(self):
        print("Your " + self.brand + " has " + str(self.miles) + " miles on it")

    def update_mile(self, mileage):
        self.miles = mileage

    def add_mile(self, kilometer):
        self.miles += kilometer


car1 = Car(10, "toyota", "black")
car1.miles = 87
car1.get_mile()
car1.update_mile(78)
car1.get_mile()
car1.add_mile(99)
car1.get_mile()

class ElectricCar(Car):
    def __init__(self, year, brand, color):
        super().__init__(year, brand, color)
        self.battery_size = 100  # 新增一個屬性初始值為100


    def get_battery(self):  # 新增一個get_battery方法
        print("Your " + self.brand + " has " + str(self.battery_size) + "-KWh batter")

electric_car = ElectricCar(2018, "Tesla", "White")
electric_car.get_name()
electric_car.get_battery()