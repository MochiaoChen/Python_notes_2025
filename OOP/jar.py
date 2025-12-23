class Jar:
    def __init__(self, capacity=12):

    def __str__(self):
        
    def deposit(self, n):

    def withdraw(self, n):

    @property
    def capacity(self):

    @property
    def size(self):
    
def main():
    jar = Jar(10)
    print(jar)
    jar.deposit(5)
    print(jar)
    jar.withdraw(2)
    print(jar)
    print(f"Capacity: {jar.capacity}, Size: {jar.size}")
    
if __name__ == "__main__":
    main()