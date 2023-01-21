from generators import control_group, random_junk_group, flat_junk_group, ufo_junk_group
import multiprocessing
import pandas as pd


if __name__ == '__main__':

    dataset = pd.DataFrame()
    number_of_responses = 10000

    pool = multiprocessing.Pool(16)
    df = pd.DataFrame()
    data = pool.map(control_group, [500] * number_of_responses)
    pool.close()
    del pool
    list_df = [df]
    list_df.extend(data)
    df = pd.concat(list_df)

    print("Control group done")



    pool = multiprocessing.Pool(16)
    df = pd.DataFrame()
    data = pool.map(random_junk_group, [500] * number_of_responses)
    pool.close()
    del pool

    # concat

    list_df = [df]
    list_df.extend(data)
    df = pd.concat(list_df)

    print("Random junk group done")



    pool = multiprocessing.Pool(16)
    df = pd.DataFrame()
    data = pool.map(flat_junk_group, [500] * number_of_responses)
    pool.close()
    del pool
    list_df = [df]
    list_df.extend(data)
    df = pd.concat(list_df)


    print("Flat junk group done")

    pool = multiprocessing.Pool(16)
    df = pd.DataFrame()
    data = pool.map(ufo_junk_group, [500] * number_of_responses)
    pool.close()
    del pool
    list_df = [df]
    list_df.extend(data)
    df = pd.concat(list_df)


    print("UFO junk group done")



    df.to_csv(r"D:\data\data_hackaton\dataset.csv")

