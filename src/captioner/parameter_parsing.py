# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Show, Prefer and Tell: Incorporating User Preferences into Image Captioning (Lindh, Ross & Kelleher, 2023)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/show-prefer-tell
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
import sys
import getopt
import os
import ast


_PARAMETER_DEFAULTS = dict(
    # General settings
    model_name="model_name",
    mode='pipeline',  # pipeline / evaluate_gt / evaluate_model / evaluate_model_multi
    gt_regions='none',  # none / candidates / grouping / selection / all / matched
                        # (upto which point should we use GT region data, matched for RPN-to-GT box matching)
    device='cuda:0',
    num_dataloader_workers=3,
    seed=333,

    # Inference settings
    use_all_regions='enforced',  # 'optional' / 'enforced', the latter treats EOC as NEXTTOKEN until out of regions
    split='test',  # 'test' / 'val'
    bb_grouping_type='nn',  # 'none' / 'nn'
    grouping_confidence_threshold=0.5,
    gather_grouping_stats='False',
    region_ranker_type='nn_iou',  # 'nn_iou' / 'object_probs'
    num_regions_min=1,  # generate or evaluate on num_regions_min to num_regions_max (0 meaning all)
    num_regions_max=5,  # set to the same as num_regions_min if only one setting is desired
    allow_fewer_regions='False',  # False will skip prediction if too few regions were detected
    region_ordering_type='rulebased',  # 'sinkhorn' / 'rulebased' / 'identity'
    limit_test_examples=0,
    max_seq_length=50,

    # CIC settings
    cic_batch_size=20,
    load_type='CIDEr',  # use the pre-trained CIC-model's best CIDEr checkpoint
    load_path_cg='../data/cic/model/',  # '' will translate to None, otherwise load_type will be joined to the path

    # Evaluation settings
    eval_mode='all',  # all | same (evaluate on the same number of regions or compared to all length captions)
    metrics="['METEOR', 'CIDEr', 'SPICE']",  # metrics="['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'ROUGE-L', 'METEOR', 'CIDEr', 'SPICE']"

    # Directories and paths
    splits_path='../data/splits_full.json',
    multi_splits_dir='../data/cic/splits',
    shared_examples_path='',
    vocabulary_path='../data/cic/vocabulary.json',
    cic_dir='../data/cic/image_bb_features',
    raw_dir='../data/cic/raw',
    image_dir='../dataset/flickr30k_images',
    results_dir='../results',
    cnn_model_dir='../data/models/vg_flickr30ksplit',
    vg_to_flickr30k_path='../data/models/vg_flickr30ksplit/flickr30k_category_info.json',
    class_embeddings_path='../data/region_order/entities_glove_6B_300d_vocab_embeddings.pt',
    grouping_model_path='../checkpoints/bb_group_model/total_loss',
    region_ranker_model_path='../checkpoints/region_ranking/total_loss',
    region_ordering_model_path='../checkpoints/sinkhorn_model/val',
    gt_captions_path="../data/cic/labels/gt_captions",
    diversity_db_path="../results/diversity_db.sql",
)


# Parse commandline options and return as a dictionary with default values for those not specified
def parse_parameters(parameters, verbose=True):
    # Format the parameter key names for use in getopts
    parameter_keys = [param_key + '=' for param_key in _PARAMETER_DEFAULTS.keys()]

    # Convert --option=value to a list of tuples
    parsed_parameters, _ = getopt.getopt(parameters, '', parameter_keys)

    # Remove leading -- from the key names, convert values to their correct types (from string) and store them in a dict
    parsed_parameters = {pair[0][2:]: type(_PARAMETER_DEFAULTS[pair[0][2:]])(pair[1]) for pair in parsed_parameters}

    # Start with the default parameter dict and override any specified parameter values to get our updated dict
    _PARAMETER_DEFAULTS.update(parsed_parameters)

    if verbose:
        print("To replicate, run with the following parameters:")
        print(' '.join(['--' + param_key + '=' + str(_PARAMETER_DEFAULTS[param_key]) for param_key in _PARAMETER_DEFAULTS.keys()]))

    # Fill in derived parameters
    _PARAMETER_DEFAULTS['device'] = torch.device(_PARAMETER_DEFAULTS['device'])

    if _PARAMETER_DEFAULTS['load_path_cg'] == '':
        _PARAMETER_DEFAULTS['load_path_cg'] = None
    else:
        _PARAMETER_DEFAULTS['load_path_cg'] = os.path.join(_PARAMETER_DEFAULTS['load_path_cg'],
                                                           _PARAMETER_DEFAULTS['load_type'])

    _PARAMETER_DEFAULTS['results_dir'] = os.path.join(os.getcwd(), _PARAMETER_DEFAULTS['results_dir'])

    # Convert the metrics strings into lists
    _PARAMETER_DEFAULTS['metrics'] = ast.literal_eval(_PARAMETER_DEFAULTS['metrics'])

    # Convert gather_grouping_stats and allow_fewer_regions into actual booleans
    _PARAMETER_DEFAULTS['gather_grouping_stats'] = (_PARAMETER_DEFAULTS['gather_grouping_stats'] == 'True')
    _PARAMETER_DEFAULTS['allow_fewer_regions'] = (_PARAMETER_DEFAULTS['allow_fewer_regions'] == 'True')

    return _PARAMETER_DEFAULTS


if __name__ == '__main__':
    # Run this script directly to test the parameter parsing
    print(parse_parameters(sys.argv[1:]))
