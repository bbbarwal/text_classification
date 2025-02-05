"""Evaluation Metrics

Author: Kristina Striegnitz and <YOUR NAME HERE>

<HONOR CODE STATEMENT HERE>

Complete this file for part 1 of the project.
"""

def get_accuracy(y_pred, y_true):
    """Calculate the accuracy of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    correct_predictions = 0
    for p, t in zip(y_pred, y_true):
        if p == t: 
            correct_predictions += 1
    total_prediction = len(y_true)
    accuracy = correct_predictions / total_prediction
    return accuracy

def get_precision(y_pred, y_true, label=1):
    """Calculate the precision of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    label: label for which we are calculating precision
    """
    true_positive = 0
    predicted_positive = 0
    for p, t in zip(y_pred, y_true):
        if p == label:
            predicted_positive += 1
            if p == t:
                true_positive += 1
    if predicted_positive == 0:
        return 0
    precision = true_positive / predicted_positive
    #print(true_positive, predicted_positive)
    return precision


def get_recall(y_pred, y_true, label=1):
    """Calculate the recall of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    label: label for which we are calculating precision
    """
    true_positive = 0
    actual_positive = 0
    for p, t in zip(y_pred, y_true):
        if t == label: 
            actual_positive += 1
            if p == t:
                true_positive += 1
    if actual_positive == 0:
        return 0
    recall = true_positive / actual_positive
    return recall


def get_fscore(y_pred, y_true, label=1):
    """Calculate the f-score of the predicted labels.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    label: label for which we are calculating precision
    """
    precision = get_precision(y_pred, y_true, label)
    recall = get_recall(y_pred, y_true, label)

    if precision + recall == 0:
        return 0
    
    fscore = (2 * (precision * recall)) / (precision + recall)
    return fscore


def evaluate(y_pred, y_true, label=1):
    """Calculate precision, recall, and f-score of the predicted labels
    and print out the results.
    y_pred: list predicted labels
    y_true: list of corresponding true labels
    """
    print(f"Accuracy: {get_accuracy(y_pred, y_true):.2f}%")
    print(f"Precision: {get_precision(y_pred, y_true, label=1):.2f}%")
    print(f"Recall: {get_recall(y_pred, y_true, label=1):.2f}%")
    print(f"F-score: {get_fscore(y_pred, y_true, label=1):.2f}%")



y_pred = [0, 1, 0, 1, 0, 1, 1]
y_true = [0, 1, 0, 0, 0, 1, 1]
print(get_precision(y_pred, y_true, 1))


y_pred = [0, 0, 1, 0, 1, 1, 1, 0, 1, 1]
y_true = [0, 0, 1, 0, 1, 1, 1, 0, 1, 1]
print(get_accuracy(y_pred, y_true))