# %% [markdown]
# # ECE 60146 HW 1
# ### Akshita Kamsali

# %% [markdown]
# ### 1. Creating a class ```Sequence```
# 
# Creating ```__init__``` and overloading ```__gt__``` in Sequence class 

# %%
class Sequence(object):
    def __init__(self, array: list) -> None:
        self.array = array

    # overload the ">" operator using the __gt__ method
    def __gt__(self, second: 'Sequence') -> bool:
        # check if the two arrays are of equal length
        if len(self.array) != len(second.array):
            raise ValueError('Two arrays are not equal in length!')
        #initialise the count to zero
        num_gt = 0
        for i in range(len(self.array)):
            if self.array[i] > second.array[i]:
                num_gt += 1
        return num_gt

# %% [markdown]
# ### 2. Creating a class ```Fibonacci```
# Create class and overload ```__call__, __len__, __iter__```
# Outputs are shown below

# %%

class Fibonacci(Sequence):
    def __init__(self, first_value: int, second_value: int) -> None:
        super().__init__([])
        self.first_value = first_value
        self.second_value = second_value

    def __call__(self, length=5) -> list:
        #intialise the array with the first two values
        self.array = [self.first_value, self.second_value]
        for i in range(length - 2):
            # adding last two numbers in the array
            self.array.append(self.array[-1] + self.array[-2])
        return self.array

    def __len__(self):
        return len(self.array)
    def __iter__(self):
        return F_iterator(self)

# reference from Avi's slides
class F_iterator:
    def __init__(self, F):
        self.F = F.array
        self.index = 0

    def __next__(self):
        if self.index >= len(self.F):
            raise StopIteration
        else:
            self.index += 1
            return self.F[self.index - 1]

    def __iter__(self):
        return self


FS = Fibonacci(1, 2)
print("Function call: ", FS(length=5))
print("Length: ", len(FS))
print("Iterable: ", [n for n in FS])

# %% [markdown]
# ### 3. Creating a class ```Prime```
# Create class and overload ```__call__, __len__, __iter__```
# Outputs are shown below

# %%
class Prime(Sequence):
    def __init__(self):
        Sequence.__init__(self, [])
    
    def isPrime(self, n: int):
        for i in range(2, n//2 + 1):
            if n % i == 0:
                return 0
        return 1

    def __call__(self, length=5):
        self.array = []
        i = 2
        while len(self.array) < length:
            if self.isPrime(i):
                self.array.append(i)
            i += 1
        return self.array
        
    def __len__(self):
        return len(self.array)

    def __iter__(self):
        return F_iterator(self)

PS = Prime()
print("Function call: ", PS(length=8))
print("Length: ", len(PS))
print("Iterable: ", [n for n in PS])

# %% [markdown]
# ### 4. Comparing classes ```Prime``` and ```Fibonacci``` with ```>``` operator
# 
# Outputs are shown below for overloaded ``` > ``` operator

# %%
FS = Fibonacci(1, 2)
print("FS call to len 8: ", FS(length=8))
PS = Prime()
print("PS call to len 8: ", PS(length=8))
print("FS > PS call: ", FS > PS)
print("PS call to len 5: " )
PS(length=5)
print("FS > PS call: ", FS > PS)