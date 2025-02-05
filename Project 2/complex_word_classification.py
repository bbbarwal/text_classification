"""Text classification for identifying complex words.

Author: Kristina Striegnitz and Baibhav Barwal

I abide by the honor code stated by Union College

Complete this file for parts 2-4 of the project.

"""

from collections import defaultdict
import gzip
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from evaluation import get_accuracy, get_fscore, get_precision, get_recall
from sklearn.ensemble import RandomForestClassifier

from syllables import count_syllables
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn


from evaluation import get_fscore, evaluate

def load_file(data_file):
    """Load in the words and labels from the given file."""
    words = []
    labels = []
    with open(data_file, 'rt', encoding="utf8") as f:
        i = 0
        for line in f:
            if i > 0:
                line_split = line[:-1].split("\t")
                words.append(line_split[0].lower())
                labels.append(int(line_split[1]))
            i += 1
    return words, labels


### 2.1: A very simple baseline

def all_complex(data_file):
    """Label every word as complex. Evaluate performance on given data set. Print out
    evaluation results."""
    words, y_true = load_file(data_file)
    y_pred = [1] * len(y_true)
    

    print("Evaluation on all_complex baseline: ")

    evaluate(y_pred, y_true)


### 2.2: Word length thresholding

def word_length_threshold(training_file, development_file):
    """Find the best length threshold by f-score and use this threshold to classify
    the training and development data. Print out evaluation results."""
    word_train, y_true = load_file(training_file)

    def evaluate_threshold(threshold, words, y_true):
        y_pred = [1 if len(word) >= threshold else 0 for word in words]
        accuracy = get_accuracy(y_true, y_pred)
        precision = get_precision(y_true, y_pred)
        recall = get_recall(y_true, y_pred)
        f1 = get_fscore(y_true, y_pred)
        return accuracy, precision, recall, f1

    word_train, y_true = load_file(training_file)
    
    best_threshold = 0
    best_f1 = 0
    best_metric = None
    for threshold in range(1, max(len(word) for word in word_train) + 1):
        accuracy, precision, recall, f1 = evaluate_threshold(threshold, word_train, y_true)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metric = (accuracy, precision, recall, f1)
    
    print(f"Best threshold on training data: {best_threshold}")
    print(f"Training data - Accuracy: {best_metric[0]}, Precision: {best_metric[1]}, Recall: {best_metric[2]}, F1: {best_metric[3]}")

    dev_words, y_dev = load_file(development_file)
    dev_metrics = evaluate_threshold(best_threshold, dev_words, y_dev)
    
    print(f"Development data - Accuracy: {dev_metrics[0]}, Precision: {dev_metrics[1]}, Recall: {dev_metrics[2]}, F1-Score: {dev_metrics[3]}")

### 2.3: Word frequency thresholding

def load_ngram_counts(ngram_counts_file):
    """Load Google NGram counts (i.e. frequency counts for words in a
    very large corpus). Return as a dictionary where the words are the
    keys and the counts are values.
    """
    counts = defaultdict(int)
    with gzip.open(ngram_counts_file, 'rt') as f:
        for line in f:
            token, count = line.strip().split('\t')
            if token[0].islower():
                counts[token] = int(count)
    return counts

def word_frequency_threshold(training_file, development_file, counts):
    """Find the best frequency threshold by f-score and use this
    threshold to classify the training and development data. Print out
    evaluation results.
    """
    max_freq = max(counts.values()) 
    min_freq = min(counts.values()) 
    print(f"\nMax word frequency: {max_freq}")
    print(f"Min word frequency: {min_freq}\n")

    train_words, y_train = load_file(training_file)

    frequency_thresholds = [10, 52640, 277261000, 1461742000, 7711493000, 20929107000, 28315430000, 34528974000, 41319061000, 47376829651]

    best_threshold = None
    best_f1 = 0
    best_metrics = None

    def evaluate_threshold(threshold, words, y_true):
        y_pred = [1 if counts[word] < threshold else 0 for word in words]
        accuracy = get_accuracy(y_true, y_pred)
        precision = get_precision(y_true, y_pred)
        recall = get_recall(y_true, y_pred)
        f1 = get_fscore(y_true, y_pred)
        return accuracy, precision, recall, f1
    
    for threshold in frequency_thresholds:
        accuracy, precision, recall, f1 = evaluate_threshold(threshold, train_words, y_train)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = (accuracy, precision, recall, f1)

    print(f"Best threshold on training data: {best_threshold}")
    print(f"Training data - Accuracy: {best_metrics[0]}, Precision: {best_metrics[1]}, Recall: {best_metrics[2]}, F1: {best_metrics[3]}")

    dev_words, y_dev = load_file(development_file)
    dev_metrics = evaluate_threshold(best_threshold, dev_words, y_dev)
    print(f"Development data - Accuracy: {dev_metrics[0]}, Precision: {dev_metrics[1]}, Recall: {dev_metrics[2]}, F1-Score: {dev_metrics[3]}")



### 3.1: Naive Bayes
def naive_bayes(training_file, development_file, counts):
    """Train a Naive Bayes classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """
    words_train, y_train = load_file(training_file)
    words_dev, y_dev = load_file(development_file)

    train_x, train_y = feature_extraction(words_train, y_train, counts)
    dev_x, dev_y = feature_extraction(words_dev, y_dev, counts)

    #normalization
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    train_x = (train_x - mean) / std
    dev_x = (dev_x - mean) / std

    clf = GaussianNB()
    clf.fit(train_x, train_y)

    #evaluation
    print("Performance on training data:")
    train_pred = clf.predict(train_x)
    dev_pred = clf.predict(dev_x)
    
    evaluate(train_pred, train_y)

    #evaluation on dev data
    print("Performance on development data:")
    evaluate(dev_pred, dev_y)

    correct_train = [word for word, true, pred in zip(words_train, train_y, train_pred) if true == pred]
    incorrect_train = [word for word, true, pred in zip(words_train, train_y, train_pred) if true != pred]

    correct_dev = [word for word, true, pred in zip(words_dev, dev_y, dev_pred) if true == pred]
    incorrect_dev = [word for word, true, pred in zip(words_dev, dev_y, dev_pred) if true != pred]

    # Print sample words
    #print("\nCorrectly Predicted Training Words:", correct_train[:20])
    #print("Incorrectly Predicted Training Words:", incorrect_train[:20])
    #print("\nCorrectly Predicted Development Words:", correct_dev[:20])
    #print("Incorrectly Predicted Development Words:", incorrect_dev[:20])

def feature_extraction(words, labels, counts):
    """Convert words into feature vectors: length and frequency."""
    x = np.array([[len(word), counts.get(word, 0)] for word in words])
    y = np.array(labels)
    return x, y
### 3.2: Logistic Regression

def logistic_regression(training_file, development_file, counts):
    """Train a Logistic Regression classifier using length and frequency
    features. Print out evaluation results on the training and
    development data.
    """
    words_train, y_train = load_file(training_file)
    words_dev, y_dev = load_file(development_file)

    train_x, train_y = feature_extraction(words_train, y_train, counts)
    dev_x, dev_y = feature_extraction(words_dev, y_dev, counts)

    #normalize
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    train_x = (train_x - mean) / std
    dev_x = (dev_x - mean) / std

    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_x, train_y)

    #evaluation
    print("Performance on training data:")
    train_pred = clf.predict(train_x)
    dev_pred = clf.predict(dev_x)

    evaluate(train_pred, train_y)

    #evaluation on dev data
    print("Performance on development data:")
    evaluate(dev_pred, dev_y)



### 3.3: Build your own classifier

def extract_features_RF(word, counts):
        word_length = len(word)
        word_freq = counts.get(word, 0)
        syllables = count_syllables(word)
        synonyms = len(wn.synsets(word))
        return [word_length, word_freq, syllables, synonyms]

def my_classifier(training_file, development_file, counts):
    ## YOUR CODE HERE
    word_train, y_train = load_file(training_file)
    word_dev, y_dev = load_file(development_file)
    
    #features
    train_x = np.array([extract_features_RF(word, counts) for word in word_train])
    train_y = np.array(y_train)

    dev_x = np.array([extract_features_RF(word, counts) for word in word_dev])
    dev_y = np.array(y_dev)

    #normalization
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    train_x = (train_x - mean) / std
    dev_x = (dev_x - mean) / std

    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
    clf.fit(train_x, train_y)

    #prediction
    train_pred = clf.predict(train_x)
    dev_pred = clf.predict(dev_x)

    print("Performance on training data:")
    evaluate(train_pred, train_y)

    print(" ")
    print("Performance on development data:")
    evaluate(dev_pred, dev_y)


def baselines(training_file, development_file, counts):
    print("========== Baselines ===========\n")

    print("Majority class baseline")
    print("-----------------------")
    print("Performance on training data")
    all_complex(training_file)
    print("\nPerformance on development data")
    all_complex(development_file)

    print("\nWord length baseline")
    print("--------------------")
    word_length_threshold(training_file, development_file)

    print("\nWord frequency baseline")
    #print("-------------------------")
    #print("max ngram counts:", max(counts.values()))
    #print("min ngram counts:", min(counts.values()))
    word_frequency_threshold(training_file, development_file, counts)

def classifiers(training_file, development_file, counts):
    print("\n========== Classifiers ===========\n")

    print("Naive Bayes")
    print("-----------")
    naive_bayes(training_file, development_file, counts)

    print("\nLogistic Regression")
    print("-----------")
    logistic_regression(training_file, development_file, counts)

    print("\nMy classifier")
    print("-----------")
    my_classifier(training_file, development_file, counts)

def final_model_trained(training_file, development_file, test_file, counts):
    word_train, y_train = load_file(training_file)
    word_dev, y_dev = load_file(development_file)

    combined_words = word_train + word_dev
    combined_labels = y_train + y_dev
     
    combined_x = np.array([extract_features_RF(word, counts) for word in combined_words])
    combined_y = np.array(combined_labels)

    #normalization
    mean = combined_x.mean(axis=0)
    std = combined_x.std(axis=0)
    combined_x = (combined_x - mean) / std

    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=42)
    clf.fit(combined_x, combined_y)

    test_words, _ = load_file(test_file)
    test_x = np.array([extract_features_RF(word, counts) for word in test_words])
    test_x = (test_x - mean) / std
    test_pred = clf.predict(test_x)

    with open("test_labels.txt", 'wt') as f:
        for label in test_pred:
            f.write(str(label) + "\n")
        




if __name__ == "__main__":
    training_file = "Project 2/data/complex_words_training.txt"
    development_file = "Project 2/data/complex_words_development.txt"
    test_file = "Project 2/data/complex_words_test_unlabeled.txt"
    print("Loading ngram counts ...")
    ngram_counts_file = "Project 2/data/ngram_counts.txt.gz"
    
    counts = load_ngram_counts(ngram_counts_file)

    baselines(training_file, development_file, counts)
    classifiers(training_file, development_file, counts)

    ## YOUR CODE HERE
    final_model_trained(training_file, development_file, test_file, counts)

