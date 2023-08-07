# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Show, Prefer and Tell: Incorporating User Preferences into Image Captioning (Lindh, Ross & Kelleher, 2023)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/show-prefer-tell
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from os import path as os_path
import json
import torch
from torch.utils.data import DataLoader
from model_wrapper import CaffeModel
from preprocessing.dataloaders import Flickr30kImageLoader, CICSinkhornPreprocessingDataset
from preprocessing.preprocess_cv_features import export_vocab_embeddings_matrix, SinkhornPreprocessor
from preprocessing.preprocess_cv_features import PrepareIoUSelectionTrainingData
from preprocessing.preprocess_cv_features import PreprocessGroupTraining


DEVICE = "cuda:0"

DATASET_DIR = '../dataset/'
IMAGE_DIR = os_path.join(DATASET_DIR, 'flickr30k_images')
FLICKR30K_SENTENCES_DIR = os_path.join(DATASET_DIR, 'Flickr30kEntities/Sentences')

RAW_DIR = '../data/cic/raw'
CIC_FEATURES_DIR = '../data/cic/image_bb_features'
MODEL_DIR = '../data/models/vg_flickr30ksplit'
SPLITS_PATH = '../data/splits_full.json'
REGION_RANKER_IOU_FEATURES_DIR = '../data/region_selector/features_iou'
SINKHORN_FEATURES_DIR = '../data/region_order/sinkhorn'
VG_TO_FLICKR30K_PATH = '../data/models/vg_flickr30ksplit/flickr30k_category_info.json'
REGION_ORDER_DIR = '../data/region_order'
GROUP_FEATURES_PATH = '../data/group_features_'


print("Loading model...")
torch.set_grad_enabled(False)
model = CaffeModel(image_dir=IMAGE_DIR, device=DEVICE)
model.load_model(prototxt_path=os_path.join(MODEL_DIR, 'test.prototxt'),
                 weights_path=os_path.join(MODEL_DIR, 'resnet101_faster_rcnn_final_iter_380000.caffemodel'),
                 caffe_proto_path=os_path.join(MODEL_DIR, 'caffe.proto')
                 )
print("...model loaded!")

print("Loading splits...")
with open(SPLITS_PATH, 'rt') as splits_file:
    splits = json.load(splits_file)['splits']


print("BB GROUP DATA")
dataloaders_grouping = dict()
for split in ['train', 'val', 'test']:
    dataset_grouping = Flickr30kImageLoader(example_ids=splits[split], image_dir=None, raw_dir=RAW_DIR,
                                            keep_entity_info=True)
    dataloaders_grouping[split] = DataLoader(dataset_grouping, batch_size=1, shuffle=False, num_workers=3,
                                             collate_fn=lambda batch: batch[0])  # return the first (and only) example

print("Preprocessing BB group training data...")
pgt = PreprocessGroupTraining(flickr30k_sentences_dir=FLICKR30K_SENTENCES_DIR)
for split in dataloaders_grouping:
    print(split)
    pgt.preprocess_all(dataloader=dataloaders_grouping[split], output_path=GROUP_FEATURES_PATH + split + ".npy")
print("...preprocessing complete!")


print("REGION SELECTION DATA")
all_example_ids = splits['train'] + splits['val'] + splits['test']
dataset_full = Flickr30kImageLoader(example_ids=all_example_ids, image_dir=IMAGE_DIR, raw_dir=RAW_DIR)
dataloader_full = DataLoader(dataset_full, batch_size=1, shuffle=False, num_workers=3,
                             collate_fn=lambda batch: batch[0])  # return the first (and only) example

print("Preparing region selection training data...")
regsel_prep = PrepareIoUSelectionTrainingData(dataloader=dataloader_full, model=model, filter_class_type="flickr30k",
                                              confidence_filter=0.3, class_nms=0.3, iou_min=None, iou_max=None,
                                              output_dir=REGION_RANKER_IOU_FEATURES_DIR,
                                              vg_to_flickr30k_path=VG_TO_FLICKR30K_PATH,
                                              device=DEVICE)
regsel_prep.preprocess_all()
print("...training data preprocessing complete!")


print("SINKHORN DATA")
print("Building the vocabulary embeddings matrix for Sinkhorn sorting...")
export_vocab_embeddings_matrix(embeddings_name='entities_glove_6B_300d',
                               output_dir=REGION_ORDER_DIR,
                               classes_path='../data/flickr30k_categories.txt',
                               embeddings_path='../data/glove/glove.6B.300d.txt')
print("...embeddings matrix saved!")

print("Preprocessing Sinkhorn network features for region ordering...")
with open("../data/cic/splits/splits_unique_sequences_0.json", "rt") as f:
    ordered_splits = json.load(f)['splits']
ordered_splits_example_ids = ordered_splits['train'] + ordered_splits['val'] + ordered_splits['test']
dataset_sinkhorn = CICSinkhornPreprocessingDataset(example_ids=ordered_splits_example_ids, data_dir=CIC_FEATURES_DIR)
dataloader_sinkhorn = DataLoader(dataset_sinkhorn, batch_size=1, shuffle=False, num_workers=7,
                                 collate_fn=lambda batch: batch[0])  # return the first (and only) example
prep_sh = SinkhornPreprocessor(dataloader=dataloader_sinkhorn,
                               visual_network=model,
                               output_dir=SINKHORN_FEATURES_DIR,
                               device=DEVICE)

prep_sh.preprocess()
print("...preprocessing finished!")

print("Exporting new splits for sinkhorn training...")
prep_sh.export_new_splits(output_path="../data/region_order/sinkhorn_splits.json", original_splits=ordered_splits)
print("...finished exporting new splits!")
