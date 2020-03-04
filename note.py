"""
NESTED FUNCTION
    Use outer function be like a declare (x = 10),
    call inner function with any parameters (y=1, y=5)
    Example about nested function in python
        def num1(x):
           def num2(y):
              return x * y
           return num2
        res = num1(10)

        print(res(5)) # return 50

FUNCTION USE YIELD && RETURN
    #----------------------------------
    def square(lst):
        sq_lst = []
        for num in lst:
                sq_lst.append(num**2)
        return sq_lst

    #----------------------------------
    def square(lst):
        for num in lst:
            yield num**2

    #----------------------------------
    team = square([1, 2, 3])
    for value in team:
        print(value)
    # Output : 1, 4, 9 (both 2 type of square)

    Yield return a generator, can call each value step by step
    Return return a list of all values
"""

"""
Version2 re-implement VOCDataset by Tensorflow's TFRecords
Use TFRecords to batch dataset, the most important is padding all image to the same size
"""

"""
self.image_shape = (512, 512) => *self.image_shape = 512 512
[*self.image_shape, 3] = [512, 512, 3]
"""