# load
import pickle
import numpy as np
from detection_gan.utils import sampling_from_survey

with open(r'D:\data\surveys\survey_list.pickle', 'rb') as handle:
    survey_list = pickle.load(handle)

with open(r'D:\data\surveys\length_survey_list.pickle', 'rb') as handle:
    length_survey_list = pickle.load(handle)

with open(r'D:\data\surveys\length_answer_list.pickle', 'rb') as handle:
    length_answer_list = pickle.load(handle)

def real_dataloader(batch_size):
    # sample 10000 surveys
    # weighted sampling according to length_answer_list


    prob_distribution = np.array(length_answer_list) / np.sum(length_answer_list)
    prob_distribution = prob_distribution ** 0.25
    prob_distribution = prob_distribution / np.sum(prob_distribution)

    indexes = np.random.choice(len(survey_list), 1000, p=prob_distribution )

    survey_tuples = [survey_list[index] for index in indexes]

    # apply sampling_from_survey to each first element of the tuple
    # and concatenate the results
    # for i in survey_tuples:
    #     a = sampling_from_survey(i[0], i[1])

    survey_tuples = [(sampling_from_survey(survey_tuple[0], survey_tuple[1])) for survey_tuple in survey_tuples]

    # limit the length of the surveys to 100x40
    survey_tuples = [(survey[:100, :40], scale[:40]) for survey, scale in survey_tuples]
    survey_tuples = [(np.pad(survey, ((0, 100-survey.shape[0]), (0, 40-survey.shape[1])), 'constant'), np.pad(scale, (0, 40-scale.shape[0]), 'constant')) for survey, scale in survey_tuples]

    surveys = np.array([survey for survey, scale in survey_tuples])
    scales = np.array([scale for survey, scale in survey_tuples])

    batches = [ (surveys[i:i+batch_size], scales[i:i+batch_size]) for i in range(0, len(surveys), batch_size)]

    return batches



