import numpy as np

def box_numbers(numbers, response_scale):
    '''
    n: [0.1, 0.6, 0.9, 0.3, 0.09, 0.2, 0.1, 0,  0]
    s: [7,   5,   3,   10,  10,   10,  10,  10, 5]
    o: [1    3    3    3    1     2    1    0   0]
    '''
    return np.array(list(map(lambda x: np.digitize(x[0], np.linspace(0.0, 1, num=x[1]+1), right=True), zip(numbers, response_scale))))


def sampling_from_survey(three_d_array, scales):
    if three_d_array.shape[1] > 1:
        n_items = np.random.randint(1, three_d_array.shape[1], size=1)
        item_choice = np.random.choice(three_d_array.shape[1], n_items, replace=False)
        three_d_array = three_d_array[:, item_choice]
        scales = scales[item_choice]

    n_rows = np.random.randint(1, three_d_array.shape[0], size=1)
    three_d_array = three_d_array[np.random.choice(three_d_array.shape[0], n_rows, replace=False), :]
    return three_d_array, scales
