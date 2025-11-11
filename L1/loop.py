def main():
    number = get_number()
    meow(number)
    
def get_number():
    '''Prompt the user for a number and return it as an integer.'''
    n = int(input("Enter a number: "))
    return n
    
def meow(n):
    ''' Print "meow" n times.'''
    for i in range(n):
        print("meow")
    
main()