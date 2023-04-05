import os

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.multiclass import OneVsRestClassifier
import seaborn as sns
import cv2


def AP(actual, predicted):
    """
    This function computes the average precision between the query image and a list
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
        a = pk(actual, predicted, i+1)
        if actual == predicted[i]:
            b = 1
            gtp += 1
        else:
            b = 0
        c = a*b
        ap += c
    if gtp == 0:
        return 0
    return ap/gtp


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


def rk(actual, predicted, k=10):
    """
    Computes the recall at k.
    This function computes the recall at k between the query image and a list
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
            The recall at k over the input
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0
    if actual in predicted:
        score = 1
    return score


def mrk(actual, predicted, k=10):
    """
    Computes the mean recall at k.
    This function computes the mean recall at k between a list of query images and a list
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
            The mean recall at k over the input
    """
    rk_list = []
    for i in range(len(actual)):
        score = rk(actual[i], predicted[i], k)
        rk_list.append(score)
    return np.mean(rk_list)


def plot_roc(train_features, test_features, train_labels, test_labels, classifier):
    idx_to_class = {0: 'Opencountry', 1: 'coast', 2: 'forest', 3: 'highway', 4: 'inside_city',
                    5: 'mountain', 6: 'street', 7: 'tallbuilding'}

    y_train = LabelBinarizer().fit_transform(train_labels)
    y_test = LabelBinarizer().fit_transform(test_labels)
    n_classes = y_train.shape[1]

    clf = OneVsRestClassifier(classifier)
    clf.fit(train_features, y_train)
    y_score = clf.predict_proba(test_features)

    fig, ax = plt.subplots(figsize=(6, 6))

    labels = np.unique(test_labels).tolist()
    palette = sns.color_palette("hls", 8)
    colors = cycle(palette)
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
          y_test[:, class_id],
          y_score[:, class_id],
          name=f"ROC curve for {idx_to_class[class_id]}",
          color=color,
          ax=ax,
        )

    plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic\nto One-vs-Rest multiclass")
    plt.legend()
    plt.savefig('roc.jpg')
    plt.show()


def plot_pr(train_features, test_features, train_labels, test_labels, classifier):
    idx_to_class = {0: 'Opencountry', 1: 'coast', 2: 'forest', 3: 'highway', 4: 'inside_city',
                    5: 'mountain', 6: 'street', 7: 'tallbuilding'}

    y_train = LabelBinarizer().fit_transform(train_labels)
    y_test = LabelBinarizer().fit_transform(test_labels)
    n_classes = y_train.shape[1]

    clf = OneVsRestClassifier(classifier)
    clf.fit(train_features, y_train)
    y_score = clf.predict_proba(test_features)

    fig, ax = plt.subplots(figsize=(6, 6))

    labels = np.unique(test_labels).tolist()
    palette = sns.color_palette("hls", 8)
    colors = cycle(palette)
    for class_id, color in zip(range(n_classes), colors):
        PrecisionRecallDisplay.from_predictions(
          y_test[:, class_id],
          y_score[:, class_id],
          name=f"PR curve for {idx_to_class[class_id]}",
          color=color,
          ax=ax,
        )

    plt.axis("square")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall\nto One-vs-Rest multiclass")
    plt.legend()
    plt.savefig('pr.jpg')
    plt.show()


def plot_imgs(neighbors, query_meta, catalogue_meta):
    idx_to_class = {0: 'Opencountry', 1: 'coast', 2: 'forest', 3: 'highway', 4: 'inside_city',
                    5: 'mountain', 6: 'street', 7: 'tallbuilding'}
    root_path = "../dataset/MIT_split/"

    for idx in [5, 260, 410, 690]:
        neighbour = neighbors[idx]
        query_path, query_label = query_meta[idx]
        query_img = cv2.imread(root_path+query_path)[:, :, ::-1]
        os.makedirs('results', exist_ok=True)
        plt.imshow(query_img)
        plt.title(f"Query Image: {idx_to_class[query_label]}")
        plt.axis('off')
        plt.savefig(f"results/query_{idx}.png", bbox_inches='tight')
        plt.show()

        for catalogue_idx in neighbour:
            catalogue_path, catalogue_label = catalogue_meta[catalogue_idx]
            catalogue_img = cv2.imread(root_path + catalogue_path)[:, :, ::-1]
            plt.imshow(catalogue_img)
            plt.title(f"Database Image: {idx_to_class[catalogue_label]}")
            plt.axis('off')
            plt.savefig(f"results/catalogue_{idx}_{catalogue_idx}.png", bbox_inches='tight')
            plt.show()
