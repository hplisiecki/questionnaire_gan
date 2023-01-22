# load
import pickle
from detection_gan.utils import sampling_from_survey

with open(r'D:\data\surveys\survey_list.pickle', 'rb') as handle:
    survey_list = pickle.load(handle)

with open(r'D:\data\surveys\length_survey_list.pickle', 'rb') as handle:
    length_survey_list = pickle.load(handle)

with open(r'D:\data\surveys\length_answer_list.pickle', 'rb') as handle:
    length_answer_list = pickle.load(handle)

def real_dataloader(batch_size, device):
    # sample 1000 surveys
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
    surveys = [survey_tuple[0] for survey_tuple in survey_tuples]
    scales = [survey_tuple[1] for survey_tuple in survey_tuples]
    # limit the length of the surveys to 100x40
    surveys = [survey[:100, :40] for survey in surveys]
    surveys = [np.pad(survey, ((0, 100-survey.shape[0]), (0, 40-survey.shape[1])), 'constant') for survey in surveys]

    return surveys, scales



