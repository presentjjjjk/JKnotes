class Mammal:
    def eat(self):
        print('老祖宗在吃东西')

class Person(Mammal):
    def eat(self):
        print('人在吃东西')

person1=Person()
person1.eat()