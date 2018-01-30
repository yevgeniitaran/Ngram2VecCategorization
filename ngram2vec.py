from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import re
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import logging
import matplotlib.pyplot as plt


def read_clusters(clusters, file_name):
    file = open(file_name, encoding="utf8")
    try:
        k = 0
        for line in file.read().splitlines():
            k = k + 1
            # print("String - line number: " + str(k) + " " + line)
            word_cluster = line.split("\t")
            clusters[word_cluster[0]] = word_cluster[1]
    except Exception:
        print("Wrong string - line number: " + str(k) + " " + line)
       
        
def process_text(data_samples, clusters):
    for i, _ in enumerate(data_samples):
        data_sample = data_samples[i]
        regex = r"\b\w+\b"
        m = re.findall(regex, data_sample)
        result = ""
        for word in m:
            category = clusters.get(word, None)
            if category != None:
                result += category + " "
        data_samples[i] = result


def prepare_dataset():
    selected_categories = ["alt.atheism", "soc.religion.christian",
                           "comp.graphics", "sci.med"]
    dataset = fetch_20newsgroups(shuffle=True, #categories=selected_categories,
                                 remove=("headers", "footers", "quotes"))
    return dataset


def split_test_data(dataset):
    return train_test_split(dataset.data, dataset.target, test_size=0.1, shuffle=False)


def custom_naiv_predict(docs_train, docs_test, y_train, y_test):
    text_clf = Pipeline([("vect", CountVectorizer(ngram_range=(1, 1), analyzer="word")),
                         ("tfidf", TfidfTransformer()),
                         ("clf", MultinomialNB()),
                         ])
    text_clf.fit(docs_train, y_train)

    y_predicted = text_clf.predict(docs_test)

    #print("ngram2vec prediction rate for " + str(clusters_num) + " clusters is " + str(np.mean(y_predicted == y_test)))
    return np.mean(y_predicted == y_test)


def usual_naiv_predict(docs_train, docs_test, y_train, y_test):
    text_clf = Pipeline([("vect", CountVectorizer(ngram_range=(1, 1), analyzer="word")),
                         ("tfidf", TfidfTransformer()),
                         ("clf", MultinomialNB()),
                         ])

    text_clf.fit(docs_train, y_train)

    predicted = text_clf.predict(docs_test)

    print("Naive bayes predict rate " + str(np.mean(predicted == y_test)))


def try_cluster_with_naiv2vec(file_name):
    clusters = {}
    read_clusters(clusters, file_name)

    dataset = prepare_dataset()
    data_samples = dataset.data
    process_text(data_samples, clusters)

    docs_train, docs_test, y_train, y_test = split_test_data(dataset)
    precision = custom_naiv_predict(docs_train, docs_test, y_train, y_test)
    return precision


def try_cluster_with_naiv2vec_with_logging(file_name):
    precision = try_cluster_with_naiv2vec(file_name)
    print("ngram2vec prediction rate for " + file_name + " clusters is " + str(precision))


def try_list_of_cluster_with_naiv2vec(clusters_num_array, precision_array):
    #for clusters_num in range(26, 100):
    for clusters_num in range(50, 501, 50):
        #file_name = "./clusters/clusters-" + str(clusters_num) + "-lower.txt"
        file_name = "./clusters-big-range-lower/clusters-" + str(clusters_num) + "-lower.txt"
        precision = try_cluster_with_naiv2vec(file_name)
        clusters_num_array.append(clusters_num)
        precision_array.append(precision)
        print("ngram2vec prediction rate for " + str(clusters_num) + " clusters is " + str(precision))


def try_default_naiv():
    dataset = prepare_dataset()
    docs_train, docs_test, y_train, y_test = split_test_data(dataset)
    usual_naiv_predict(docs_train, docs_test, y_train, y_test)


def main():
    logging.basicConfig(filename="myapp2.log", level=logging.INFO)
    clusters_num_array = []
    precision_array = []
    try_list_of_cluster_with_naiv2vec(clusters_num_array, precision_array)
    plt.plot(clusters_num_array, precision_array)
    try_cluster_with_naiv2vec_with_logging("clusterization-100.txt")
    try_default_naiv()

if __name__ == "__main__":
    main()


