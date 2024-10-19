# def person(name,age,gender):

#     def hit(person,dog):
#         dog['life']-=person['attack']
#         print(f"{person['name']}攻击了{dog['name']},{dog['name']}还剩下{dog['life']}点生命值")

#     person_dict={
#         'name':name,
#         'age':age,
#         'gender':gender,
#         'life':100,
#         'attack':10,
#         'hit':hit
#     }


#     return person_dict

# def dog(name,age,gender):

#     def bite(dog,person):
#         person['life']-=dog['attack']
#         print(f"{dog['name']}攻击了{person['name']},{person['name']}还剩下{person['life']}点生命值")
    
#     dog_dict={
#         'name':name,
#         'age':age,
#         'gender':gender,
#         'life':100,
#         'attack':10,
#         'bite':bite
#     }

    
#     dog_dict['bite']=bite
#     return dog_dict

# #创建人狗实例
# dog1=dog('旺财',3,'雄')
# person1=person('张三',20,'男')

# #人狗交互
# dog1['bite'](dog1,person1)
# person1['hit'](person1,dog1)    

# class Dog:
#     def attack_power(self):
#         return self.life * 0.1
#     def __init__(self, name, age, gender):
#         self.name = name
#         self.age = age
#         self.gender = gender
#         self.life = 100
#         self.attack = self.attack_power

# dog1=Dog('旺财',3,'雄')
# print(dog1.attack_power())

class Dog:
    breed='哈士奇'
    life=100
    attack=10
    def __init__(self, name, age, gender):
        self.name = name
        self.age = age
        self.gender = gender
    def bite(self,person):
        person.life-=self.attack
        print(f"{self.name}攻击了{person.name},{person.name}还剩下{person.life}点生命值")

class Person:
    breed='human'
    life=100
    attack=10
    def __init__(self, name, age, gender):
        self.name = name
        self.age = age
        self.gender = gender
    def hit(self,dog):
        dog.life-=self.attack
        print(f"{self.name}攻击了{dog.name},{dog.name}还剩下{dog.life}点生命值")

dog1=Dog('旺财',3,'雄')
person1=Person('张三',20,'男')
dog1.bite(person1)
person1.hit(dog1)