# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Show, Prefer and Tell: Incorporating User Preferences into Image Captioning (Lindh, Ross & Kelleher, 2023)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/show-prefer-tell
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
from preprocess_entities import EntityRawPreprocessor
from preprocess_entities import create_gt_dicts, format_gt_as_candidates, create_unique_multi_splits
from preprocess_entities import map_vg_categories_to_entity_categories
from preprocess_entities import flickr30k_region_stats


RAW_DIR = '../data/cic/raw'
LABELS_DIR = '../data/cic/labels/'
UNIQUE_SPLITS_DIR = '../data/cic/splits'
RESULTS_DIR = '../results'
for current_dir in [RAW_DIR, LABELS_DIR, UNIQUE_SPLITS_DIR, RESULTS_DIR]:
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)

FLICKR30K_ENTITIES_DIR = '../dataset/Flickr30kEntities'


print("Mapping VG categories to Flickr30k Entities categories...")
map_vg_categories_to_entity_categories(vg_classes_path='../data/models/vg_flickr30ksplit/objects_vocab.txt',
                                       output_dir='../data/models/vg_flickr30ksplit/')


print("RAW ENTITIES DATA")
print("Preprocessing Raw Entities data")
ecp = EntityRawPreprocessor(datadir=FLICKR30K_ENTITIES_DIR,
                            output_dir=RAW_DIR)
ecp.preprocess_all('../data/splits_full.json')


print("SPLITS")
print("Creating splits for unique sequences and unique unsorted lists...")
create_unique_multi_splits(image_splits_path='../data/splits_full.json',
                           raw_dir=RAW_DIR,
                           output_dir=UNIQUE_SPLITS_DIR,
                           all_num_regions=range(0, 8))


print("GT CAPTIONS")
print("Creating gt caption dicts...")
create_gt_dicts(image_splits_path='../data/splits_full.json',
                raw_dir=RAW_DIR,
                out_dir=LABELS_DIR)

print("Formatting the GT captions as candidate CAPTIONS files...")
format_gt_as_candidates(gt_path=LABELS_DIR + '/gt_captions_test.json',
                        output_dir=RESULTS_DIR)


print("STATS")

print("Generating historgrams with number of entities per caption...")
flickr30k_region_stats(splits_path='../data/splits_full.json',
                       raw_dir=RAW_DIR,
                       outfile='../data_exploration/hist_region_per_caption')
