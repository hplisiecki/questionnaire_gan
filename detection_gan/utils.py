import numpy as np

def box_numbers(numbers, response_scale):
    '''
    n: [0.1, 0.6, 0.9, 0.3, 0.09, 0.2, 0.1, 0,  0]
    s: [7,   5,   3,   10,  10,   10,  10,  10, 5]
    o: [1    3    3    3    1     2    1    0   0]
    '''
    return np.array(list(map(lambda x: np.digitize(x[0], np.linspace(0.0, 1, num=x[1]+1), right=True), zip(numbers, response_scale))))