import numpy as np


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def pk(actual, predicted, k=10):
    """
    Computes the precision at k.
    This function computes the precision at k between the query image and a list
    of database retrieved images.
    Parameters
    ----------
    actual : int
             The element that has to be predicted
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The precision at k over the input
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0
    for i in range(len(predicted)):
        if actual == predicted[i]:
            score += 1

    return score / len(predicted)


def mpk(actual, predicted, k=10):
    """
    Computes the precision at k.
    This function computes the mean precision at k between a list of query images and a list
    of database retrieved images.
    Parameters
    ----------
    actual : list
             The query elements that have to be predicted
    predicted : list
                A list of predicted elements (order does matter) for each query element
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The precision at k over the input
    """
    pk_list = []
    for i in range(len(actual)):
        score = pk(actual[i], predicted[i], k)
        pk_list.append(score)
    return np.mean(pk_list)


def AP(actual, predicted):
    """
    Computes the precision at k.
    This function computes the average precision precision between the query image and a list
    of database retrieved images.
    Parameters
    ----------
    actual : int
             The element that has to be predicted
    predicted : list
                A list of predicted elements (order does matter)
    Returns
    -------
    score : double
            The Average Precision over the input
    """
    gtp = 0
    ap = 0
    for i in range(len(predicted)):
        a = pk(actual, predicted, i + 1)
        if actual == predicted[i]:
            b = 1
            gtp += 1
        else:
            b = 0
        c = a * b
        ap += c
    if gtp == 0:
        return 0
    return ap / gtp


def mAP(actual, predicted):
    """
    Computes the precision at k.
    This function computes the mean Average Precision between the query image and a list
    of database retrieved images.
    Parameters
    ----------
    actual : list
             The query elements that have to be predicted
    predicted : list
                A list of predicted elements (order does matter) for each query element
    Returns
    -------
    score : double
            The mean Average Precision over the input
    """

    ap_list = []
    for i in range(len(actual)):
        ap = AP(actual[i], predicted[i])
        ap_list.append(ap)
    return np.mean(ap_list)


####### DATASET UTILS ############
def reduce_txt_embeds(embeds):
    aux1 = []
    for i in range(len(embeds)):
        aux2 = []
        for sent in embeds[i]:
            aux2.append(np.mean(sent, axis=0))
        aux1.append(aux2)
    return np.asarray(aux1)


def decay_learning_rate(init_lr, optimizer, epoch):
    """
    decay learning late every 4 epoch
    """
    lr = init_lr * (0.1 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr