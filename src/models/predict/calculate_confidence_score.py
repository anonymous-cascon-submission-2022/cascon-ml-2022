"""
Confidence Score Legend

1 - No Merger / No Acquisition
2 - Potentially Merger or Acquisition
3 - Likely Merger or Acquisition
4 - Confident Merger or Acquisition
5 - High Confidence Merger or Acquisition


"""
import math


def calculate_sentence_confidence_score(is_merger_or_acquisition, is_merger, is_acquisition, involved_companies) -> int:
    """
    Calculate the confidence score of a processed sentence

    :param is_merger_or_acquisition: int, 1 if is merger or acquisition
    :param is_merger:  int, 1 if is merger
    :param is_acquisition:  int, 1 if is acquisition
    :param involved_companies: list, companies involved in sentence
    :return: int, confidence score
    """
    score = 1

    # is the sentence detected as a merger or an acquisition
    if is_merger_or_acquisition == 1:
        score += 1

    # is the sentence specified as a merger or specified as an acquisition
    if is_merger == 1 or is_acquisition == 1:
        score += 1

    # does the sentence have NER
    if involved_companies:
        if not involved_companies[0] == '{}':
            score += 1

    # does the sentence have multiple organizations mentioned
    if len(involved_companies) > 1:
        score += 1

    return score


def calculate_abstracted_confidence_score(instances) -> int:
    """
    Calculate the confidence score of either a webpage or a company

    :param instances: dataframe of sentences or webpages
    :return: int, confidence score
    """
    # get average of confidence scores of instances
    score = math.floor(sum(list(instances["confidenceScore"])) / len(instances["confidenceScore"]))

    # if more than 3 instances detected, add 1 (MAX score 5)
    # if score < 5 and len(instances["confidenceScore"]) > 3:
    #     score += 1

    return score
