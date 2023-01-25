import numpy as np
import torch
def box_numbers(numbers, response_scale):
    '''
    numbers - a flat cuda tensor of numbers
    response_scale - a flat cuda tensor of the response scale
    n: [0.1, 0.6, 0.9, 0.3, 0.9, 0.2, 0.1, 0,  0]
    s: [7,   5,   3,   10,  10,   10,  10,  10, 5]
    o: [1    3    3    3    10     2    1    0   0]
    '''
    # bucketize
    scale_numbers = (numbers * response_scale)
    # change zeros to inf
    modify_numbers = torch.where(scale_numbers ==0, torch.inf, scale_numbers)
    rounded_numbers = torch.round(modify_numbers)
    scaled_buckets = (rounded_numbers / response_scale)
    bucketized_numbers = torch.where(torch.isinf(scaled_buckets), torch.zeros_like(scaled_buckets), scaled_buckets)

    return bucketized_numbers

def sampling_from_survey(three_d_array, scales):
    if three_d_array.shape[1] > 1:
        n_items = np.random.randint(1, three_d_array.shape[1], size=1)
        item_choice = np.random.choice(three_d_array.shape[1], n_items, replace=False)
        three_d_array = three_d_array[:, item_choice]
        scales = scales[item_choice]

    n_rows = np.random.randint(1, three_d_array.shape[0], size=1)
    three_d_array = three_d_array[np.random.choice(three_d_array.shape[0], n_rows, replace=False), :]
    return three_d_array, scales
