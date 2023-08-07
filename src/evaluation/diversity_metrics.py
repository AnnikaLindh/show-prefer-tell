# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Show, Prefer and Tell: Incorporating User Preferences into Image Captioning (Lindh, Ross & Kelleher, 2023)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/show-prefer-tell
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Diversity metrics code adapted from Generating Diverse and Meaningful Captions (Lindh et al., 2018):
# https://github.com/AnnikaLindh/Diverse_and_Specific_Image_Captioning/

import sqlite3
import numpy as np


# Store captions in an sqlite table
def store_captions_in_table(db_path, table_name, example_captions):
    caption_data = list()
    for example_id in example_captions:
        for caption in example_captions[example_id]:
            caption_data.append((example_id, caption,))

    with sqlite3.connect(db_path) as conn:
        conn.execute('DROP TABLE IF EXISTS ' + table_name)
        conn.execute('CREATE TABLE ' + table_name + ' (example_id TEXT, caption TEXT)')
        conn.executemany('INSERT INTO ' + table_name + ' VALUES (?,?)', caption_data)
        conn.commit()


# Novel Sentences: percentage of generated captions not seen in the training set.
def calculate_novelty(db_path, table_name_gen, table_name_gts, verbose=False):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute('SELECT COUNT(*), caption FROM ' + table_name_gen + ' WHERE caption NOT IN (SELECT caption FROM ' + table_name_gts + ')')
        num_novel = cur.fetchone()[0]
        cur.execute('SELECT COUNT(*) FROM ' + table_name_gen)
        num_total = cur.fetchone()[0]
        cur.close()

    fraction_novel = float(num_novel)/float(num_total)

    if verbose:
        print("Total generated captions =", num_total)
        print("Number of novel =", num_novel)
        print("Fraction novel =", fraction_novel)
        print("Fraction seen in training data =", 1-fraction_novel)

    return fraction_novel * 100.0


# Diversity: Calculate Distinct caption stats
def calculate_distinct(db_path, table_name, verbose=False):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute('SELECT COUNT(DISTINCT caption) FROM ' + table_name)
        num_distinct = cur.fetchone()[0]
        cur.execute('SELECT COUNT(*) FROM ' + table_name)
        num_total = cur.fetchone()[0]
        cur.close()

    fraction_distinct = float(num_distinct) / float(num_total)
    if verbose:
        print("Total generated captions =", num_total)
        print("Number of distinct =", num_distinct)
        print("Fraction distinct =", fraction_distinct)

    return fraction_distinct * 100.0


# Vocabulary Size: number of unique words used in all generated captions
# Returns number of unique words used in the captions of this table
def calculate_vocabulary_usage(db_path, table_name, verbose=False):
    # Build a set of unique words used in the captions of this table+split
    vocab = set()

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute('SELECT caption FROM ' + table_name)
        for caption in cur:
            vocab.update(caption[0].split(' '))
        cur.close()

    if verbose:
        print("Total vocabulary used =", len(vocab))
        if 'UNK' in vocab:
            print('UNK is part of vocab.')
        else:
            print('UNK is NOT part of vocab.')

        # print("Vocab:", vocab)

    return len(vocab)


def calculate_caption_lengths(db_path, table_name, verbose=False):
    caption_lengths = list()

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute('SELECT caption FROM ' + table_name)
        for caption in cur:
            caption_lengths.append(len(caption[0].split(' ')))
        cur.close()

    avg_length = np.asarray(caption_lengths).mean()
    if verbose:
        print("Average caption length = ", avg_length)

    return avg_length
