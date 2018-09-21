# -*- coding: utf-8 -*-

import codecs
import os
import tensorflow as tf
from nltk.corpus import wordnet as wn

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('dir', '', 'dir to input and output .txt files')
tf.app.flags.DEFINE_string('file_name', '', 'input .txt file names')

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence
ENGLISH_TOKENS = ['.', '?', '!', ',', ':', '...', ';', '-', '–', '—', '(', ')', '[', ']', '{', '}']

max_sent_size = 50
max_enc_steps = 35
max_select_num = 3
max_window_size = 5

stop_words_set = set()


# get stop words set by wordnet english_stop_words.txt file
def get_stop_words_set():
    with codecs.open(os.path.join(FLAGS.dir, "english_stop_words.txt"), "r", "utf-8") as fr:
        for word in fr.readlines():
            stop_words_set.add(word.strip())


# remove the english punctuations in the string and return the corresponding list
def filter_tokens(sentence):
    for token in ENGLISH_TOKENS:
        sentence = sentence.replace(token, "")
    return sentence.strip().split()


# get sentence score by word net
def get_sent_avg_score(sentence):
    sent_lst = filter_tokens(sentence)
    # filter the stop, unambitious words and get the high meaning count words up to max_window_size
    dic_mean_cnt = {}
    for i in range(len(sent_lst)):
        if (sent_lst[i] not in stop_words_set) and (len(wn.synsets(sent_lst[i])) >= 2):
            dic_mean_cnt[sent_lst[i]] = len(wn.synsets(sent_lst[i]))
    del sent_lst
    # check if dic_mean_cnt size is less than one
    if len(dic_mean_cnt) <= 1:
        return 0.0
    # sort by word meaning count
    lst_mean_cnt = sorted(dic_mean_cnt.items(), key=lambda x: x[1], reverse=True)
    if len(lst_mean_cnt) > max_window_size:
        lst_mean_cnt = lst_mean_cnt[:max_window_size]
    sent_select_lst = [val[0] for val in lst_mean_cnt]
    # counter all the word pair counters
    dic_score = {} # dic_score[i][k] = c means word i and meaning k have common counters c
    for i in range(len(sent_select_lst)):
        for j in range(0, i):
            mean_i = wn.synsets(sent_select_lst[i])
            mean_j = wn.synsets(sent_select_lst[j])
            for k in range(len(mean_i)):
                for l in range(len(mean_j)):
                    counters = len(set(filter_tokens(mean_i[k].definition())).intersection(set(filter_tokens(mean_j[l].definition()))))
                    # update dic_score for i, j, k, counters
                    if i not in dic_score:
                        dic_score[i] = {k: counters}
                    else:
                        if k not in dic_score[i]:
                            dic_score[i].update({k: counters})
                        else:
                            dic_score[i][k] += counters
                    # update dic_score for j, i, l, counters
                    if j not in dic_score:
                        dic_score[j] = {l: counters}
                    else:
                        if l not in dic_score[j]:
                            dic_score[j].update({l: counters})
                        else:
                            dic_score[j][l] += counters
    # sort to get the sentence counters score
    counters_score = 0
    for key in dic_score.keys():
        dic_key_lst = sorted(dic_score[key].items(), key=lambda x: x[1], reverse=True)
        counters_score += dic_key_lst[0][1]
    avg_counters_score = counters_score / len(sent_select_lst)
    # return the average counters score
    return avg_counters_score


# get the selected texts from original texts
def get_data():
    # get stop word set
    get_stop_words_set()
    # read original texts and get high score texts
    with codecs.open(os.path.join(FLAGS.dir, "new_" + FLAGS.file_name), "w", "utf-8") as fw:
        with codecs.open(os.path.join(FLAGS.dir, FLAGS.file_name), "r", "utf-8") as fr:
            count = 0
            for line in fr:
                article, abstract = line.strip().split("\t")
                count += 1
                print("count: ", count)
                # check if it is an empty article or abstract
                if (len(article[8:]) == 0) or (len(abstract[9:]) == 0):
                    continue
                article_lst = article[8:].split()
                n = len(article_lst)
                start = 0
                article_sentences = []
                # get sentences which has the longest length subject to <= max_sent_size, and at most max_enc_steps sentences
                while start < n:
                    end = min(n - 1, start + max_sent_size - 1)
                    pos = end
                    while pos >= start and article_lst[pos] not in END_TOKENS:
                        pos -= 1
                    # find a end punctuation in [start, end]
                    if pos > start - 1:
                        article_sentences.append(article_lst[start: pos + 1])
                    # not find, continue to search a end punctuation in [end + 1, n - 1]
                    else:
                        pos = end + 1
                        while pos < n and article_lst[pos] not in END_TOKENS:
                            pos += 1
                        article_sentences.append(article_lst[start: pos + 1])
                    start = pos + 1
                    # meet the max_enc_steps, break out
                    if len(article_sentences) >= max_enc_steps:
                        break
                print("article senetnces length: ", len(article_sentences), "\n")
                # print("article senetnces: ", article_sentences)
                # alternative: based on WordNet, we evaluate the weights of all the sentences using the Simplified Lesk algorithm and arranges them in decreasing order according to their weights.
                # calculate the rouge score between sentences and reference summaries, and sort them in decreasing order according to their score
                score_dic = {}
                for sent in article_sentences:
                    score_dic[" ".join(sent)] = get_sent_avg_score(" ".join(sent))
                # sort
                score_lst = sorted(score_dic.items(), key=lambda x: x[1], reverse=True)
                if len(score_lst) > max_select_num:
                    score_lst = score_lst[:max_select_num]
                score_lst = [val[0] for val in score_lst]
                # set data format
                selected_article = article + " $SPLIT$ " + " $$$ ".join(score_lst)
                fw.write(selected_article + "\t" + abstract + "\n")


if __name__ == "__main__":
    assert FLAGS.dir and FLAGS.file_name
    get_data()
