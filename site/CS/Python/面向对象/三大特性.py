# class Mammal:
#     def eat(self):
#         print('老祖宗在吃东西')

# class Person(Mammal):
#     def eat(self):
#         print('人在吃东西')

# person1=Person()
# person1.eat()

# class Example:
#     def __init__(self):
#         self.__private_attr = 10  # 私有属性
    
#     def __private_method(self):  # 私有方法
#         print("这是一个私有方法")
    
#     def public_method(self):
#         print(self.__private_attr)
#         self.__private_method()

# example = Example()
# example.public_method()

class Example:
    def __init__(self):
        self.__private_attr = 10  # 私有属性
    
    def __private_method(self):  # 私有方法
        return "这是一个私有方法的返回值"
    
    def public_method(self):
        private_attr_value = self.__private_attr
        private_method_result = self.__private_method()
        return private_attr_value, private_method_result

# 使用示例
e = Example()
attr_value, method_result = e.public_method()

print(f"私有属性的值: {attr_value}")
print(f"私有方法的返回值: {method_result}")