import logging
import time
import gensim
from sklearn.cluster import KMeans
from time import gmtime, strftime


def info_log_with_time(str):
    logging.info(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " " + str)


def to_lower_case(w2v_model, most_frequent_lower, word_vectors_unique):
    """
    Collects only word vectors with most frequent case and then converts them to lowercase
    """
    i = 0
    word_vectors = w2v_model.wv.syn0
    info_log_with_time("Start lowercase transformation")
    w2vectors = list(zip(w2v_model.wv.index2word, word_vectors))
    for word_with_vec in w2vectors:
        if word_with_vec[0].lower() not in most_frequent_lower:
            most_frequent_lower.append(word_with_vec[0].lower())
            word_vectors_unique.append(word_with_vec[1])
        i = i + 1
        if i % 1000 == 0:
            info_log_with_time(str(i) + " " + str(word_with_vec[0]))
    info_log_with_time("Lowercase transformation is finished")


def clustering(most_frequent_lower, word_vectors_unique, file_name, clust_num):
    start = time.time()
    info_log_with_time("Starting clustering for " + str(clust_num) + " clusters")
    kmeans = KMeans(n_clusters=clust_num, n_jobs=-1, random_state=0)
    idx = kmeans.fit_predict(word_vectors_unique)

    w2vectors_list = list(zip(most_frequent_lower, idx))
    w2vectors_list_for_print = sorted(w2vectors_list, key=lambda val: val[1], reverse=False)
    file_out = open(file_name, "w", encoding="utf-8")
    for w2v in w2vectors_list_for_print:
        line = w2v[0] + '\t' + str(0) + str(w2v[1]) + '\n'
        file_out.write(line)
    file_out.close()
    end = time.time()
    info_log_with_time("Clustering took " + str(round((end - start) / 60, 1))  + " minutes")


def main():
    logging.basicConfig(filename='myapp-clust2.log', level=logging.INFO)
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300-SLIM.bin", binary=True)

    most_frequent_lower = []
    word_vectors_unique = []
    to_lower_case(w2v_model, most_frequent_lower, word_vectors_unique)

    for clust_num in range (50, 501, 50):
        file_name = "./clusters-big-range-lower/clusters-" + str(clust_num) + "-lower.txt"
        clustering(most_frequent_lower, word_vectors_unique, file_name, clust_num)


if __name__ == "__main__":
    main()


