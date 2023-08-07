# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Show, Prefer and Tell: Incorporating User Preferences into Image Captioning (Lindh, Ross & Kelleher, 2023)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/show-prefer-tell
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import sys
import os
sys.path.append(os.getcwd())

import json
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from caption_generator import CaptionGenerator
from dataloaders import RawDataset, PreprocessedDataset, CollatePreprocessedData
from evaluation.text_metrics import EvaluateCaptions
from parameter_parsing import parse_parameters
from region_ranker.model import RegionRankerObjectProbs, RegionRankerNNIoU
from vision_network import VisionNetwork
from bb_grouping.bb_group_models import GroupingNN
from ordering.rulebased_ordering import order_rulebased, IdentityOrder
from ordering.sinkhorn import SinkhornOrdering
from evaluation.diversity_metrics import store_captions_in_table, calculate_distinct, calculate_novelty
from evaluation.diversity_metrics import calculate_vocabulary_usage, calculate_caption_lengths
import matplotlib


# Prevent errors due to lack of a connected display
matplotlib.use('Agg')
from matplotlib import pyplot as plt


_CONFIG = None


# Run the pipeline on raw image data, optionally with some ground-truth info for ablation tests
def run_pipeline(dataloader, cnn, caption_generator, num_regions, allow_fewer_regions, display_boxes_dir=None,
                 print_results=False):
    if display_boxes_dir is None:
        export_params = None
    else:
        export_params = {"output_dir": display_boxes_dir}

    # Ignore the allow_fewer_regions setting if we are not limiting the number of regions
    if num_regions < 1:
        allow_fewer_regions = True

    num_skipped_images = 0
    all_example_ids = list()
    all_predictions = list()
    all_next_positions = list()
    for batch in dataloader:
        if export_params is not None:
            export_params["image_id"] = batch["example_id"]

        vision_features = cnn.extract_features(batch["image"], num_regions=num_regions,
                                               class_nms=0.3, override_boxes=batch["boxes"],
                                               export_params=export_params)

        if (vision_features is None) or ((not allow_fewer_regions) and
                                         (len(vision_features['region_features'])-1 < num_regions)):
            num_skipped_images += 1
        else:
            predictions, _, _, _, next_positions = caption_generator.inference(
                full_image_features=vision_features['full_image_features'],
                region_features=vision_features['region_features'],
                region_start_indices=vision_features['region_start_indices'],
                region_end_indices=vision_features['region_end_indices'],
                max_seq_length=_CONFIG['max_seq_length'],
                device=_CONFIG['device']
                )

            decoded_prediction = caption_generator.decode_sequences(predictions)

            all_example_ids.append(batch["example_id"])
            all_predictions.append(decoded_prediction[0])
            all_next_positions.append(next_positions)

            if print_results:
                print(batch["example_id"], decoded_prediction)

    if allow_fewer_regions:
        print("Number of skipped images due to zero accepted boxes:", num_skipped_images)
    else:
        print("Number of skipped images due to too few boxes:", num_skipped_images)

    return all_example_ids, all_predictions, all_next_positions


# Run the pipeline with pre-processed region features from Lindh et al. (2020) for ablation tests
def run_cic_gt_pipeline(dataloader, cnn, caption_generator, num_regions, allow_fewer_regions, print_results=False,
                        export_params=None):
    all_example_ids = list()
    all_predictions = list()
    all_next_positions = list()
    num_skipped_images = 0

    # Ignore the allow_fewer_regions setting if we are not limiting the number of regions
    if num_regions < 1:
        allow_fewer_regions = True

    for batch in dataloader:
        if batch["boxes"] is None:
            vision_features = batch
        else:
            if export_params is not None:
                export_params["image_id"] = batch["example_ids"][0]

            vision_features = cnn.order_cic_regions(cic_features=batch, num_regions=num_regions,
                                                    class_nms=0.3, export_params=export_params)

        if (not allow_fewer_regions) and (len(vision_features['region_features']) - 1 < num_regions):
            num_skipped_images += 1
        else:
            predictions, _, _, _, next_positions = caption_generator.inference(
                full_image_features=vision_features['full_image_features'].to(_CONFIG['device']),
                region_features=vision_features['region_features'].to(_CONFIG['device']),
                region_start_indices=vision_features['region_start_indices'],
                region_end_indices=vision_features['region_end_indices'],
                max_seq_length=_CONFIG['max_seq_length'],
                device=_CONFIG['device']
            )

            decoded_predictions = caption_generator.decode_sequences(predictions)

            all_example_ids.extend(batch["example_ids"])
            all_predictions.extend(decoded_predictions)
            # Convert into the format expected by the store_captions function
            all_next_positions.extend([np.expand_dims(nexts, 0) for nexts in next_positions])

            if print_results:
                print(batch["example_ids"], decoded_predictions)

    if not allow_fewer_regions:
        print("Number of skipped images due to too few boxes:", num_skipped_images)

    return all_example_ids, all_predictions, all_next_positions


# Run the pipeline on raw image data but matching detections to ground-truth boxes for ablation tests
def run_box_matched_pipeline(dataloader, cnn, caption_generator, num_regions, allow_fewer_regions,
                             display_boxes_dir=None, print_results=False):
    if display_boxes_dir is None:
        export_params = None
    else:
        export_params = {"output_dir": display_boxes_dir}

    # Ignore the allow_fewer_regions setting if we are not limiting the number of regions
    if num_regions < 1:
        allow_fewer_regions = True

    num_skipped_images = 0
    all_example_ids = list()
    all_predictions = list()
    all_next_positions = list()
    percent_matched_entities = list()
    for batch in dataloader:
        if num_regions > 0:
            batch["gt_entity_order"] = batch["gt_entity_order"][:num_regions]

        if export_params is not None:
            export_params["image_id"] = batch["example_id"]

        vision_features = cnn.extract_closest_box_features(image=batch["image"],
                                                           gt_boxes=batch["boxes"],
                                                           gt_entities=batch["gt_entities"],
                                                           gt_entity_order=batch["gt_entity_order"],
                                                           class_nms=0.3, min_confidence=0.3,
                                                           export_params=export_params)

        if (vision_features is None) or ((not allow_fewer_regions) and
                                         (len(vision_features['region_features'])-1 < num_regions)):
            num_skipped_images += 1
        else:
            if allow_fewer_regions:
                # print("Matching", (len(vision_features['region_features'])-1), "out of", len(batch["gt_entity_order"]))
                percent_matched_entities.append((len(vision_features['region_features'])-1) / len(batch["gt_entity_order"]))

            predictions, _, _, _, next_positions = caption_generator.inference(
                full_image_features=vision_features['full_image_features'],
                region_features=vision_features['region_features'],
                region_start_indices=vision_features['region_start_indices'],
                region_end_indices=vision_features['region_end_indices'],
                max_seq_length=_CONFIG['max_seq_length'],
                device=_CONFIG['device']
                )

            decoded_prediction = caption_generator.decode_sequences(predictions)

            all_example_ids.append(batch["example_id"])
            all_predictions.append(decoded_prediction[0])
            all_next_positions.append(next_positions)

            if print_results:
                print(batch["example_id"], decoded_prediction)

    if allow_fewer_regions:
        print("Number of skipped images due to zero accepted boxes:", num_skipped_images)
    else:
        print("Number of skipped images due to too few boxes:", num_skipped_images)

    if allow_fewer_regions:
        # Export histogram of percent of GT boxes matched by RPN detections per example
        output_path = os.path.join(_CONFIG["results_dir"], 'percent_matched_hist_' + _CONFIG["model_name"] + "_"
                                   + num_regions)
        _export_matched_boxes_hist(percent_matched_entities, output_path + '.png')

    return all_example_ids, all_predictions, all_next_positions


def evaluate_captions_from_file(generated_captions_path, output_dir, gt_captions_path, metrics, db_path, model_name,
                                num_regions, eval_mode='all', shared_ids=None):
    with open(generated_captions_path, 'rt') as captions_file:
        captions_data = json.load(captions_file)
        captions = captions_data['generated_captions']
        next_positions = captions_data['next_positions']

    # Only evaluate the shared examples
    if shared_ids is not None:
        captions = {example_id: captions[example_id] for example_id in shared_ids}

    # Setup evaluation
    eval = EvaluateCaptions()
    eval.setup(metrics=metrics,
               gt_path=gt_captions_path,
               example_ids=None)

    # Run standard metrics calculations
    num_eval_captions = num_regions if (eval_mode == 'same') else 0
    scores, _, num_captions = eval.evaluate_candidate_captions(captions, num_regions=num_eval_captions)
    if scores is None:
        print(f"WARNING: No gt to compare to for model={model_name}, num_regions={num_regions}, eval_mode={eval_mode}")
        return

    print(f"Evaluated {num_captions} out of {len(captions)} captions.")

    # Calculate diversity metrics
    store_captions_in_table(db_path=db_path, table_name="generated_captions", example_captions=captions)
    novel = calculate_novelty(db_path=db_path, table_name_gen="generated_captions", table_name_gts="gt_train")
    diversity = calculate_distinct(db_path=db_path, table_name="generated_captions")
    vocab = calculate_vocabulary_usage(db_path=db_path, table_name="generated_captions")
    cap_length = calculate_caption_lengths(db_path=db_path, table_name="generated_captions")

    num_chunks = average_num_chunks(captions=captions, next_positions=next_positions)

    # Save results to csv file
    csv_row = list()
    csv_row.append(model_name)  # model_name
    csv_row.append(num_regions)  # num_regions
    csv_row.extend([scores[metric] for metric in metrics])  # <metric>
    csv_row.append(novel)  # novel
    csv_row.append(diversity)  # distinct
    csv_row.append(vocab)  # vocab_size
    csv_row.append(cap_length)  # length
    csv_row.append(num_chunks)  # num_chunks
    csv_row.append(len(captions))  # num_generated_captions
    csv_row.append(num_captions)  # num_evaluated_captions
    csv_row.append(eval_mode)  # eval_mode

    output_path = os.path.join(output_dir, f"METRICS_{model_name}_{num_regions}.csv")
    if os.path.exists(output_path):
        file_mode = 'at'
    else:
        file_mode = 'wt'

    with open(output_path, file_mode) as f:
        writer = csv.writer(f, delimiter=';')
        if file_mode == 'wt':
            header = ['model_name', 'num_regions'] + metrics + ['novel', 'distinct', 'vocab_size', 'length',
                                                                'num_chunks', 'num_generated_captions',
                                                                'num_evaluated_captions', 'eval_mode']
            writer.writerow(header)
        writer.writerow(csv_row)


def _initialize_details_dict(original_dict):
    return {metric: dict() for metric in original_dict}


def _add_individual_scores(main_dict, individual_scores):
    for metric in individual_scores:
        for image_id in individual_scores[metric]:
            try:
                main_dict[metric][image_id].append(
                    individual_scores[metric][image_id])
            except KeyError:
                main_dict[metric][image_id] = [
                    individual_scores[metric][image_id]]


# Caption files that contain multiple captions per image like in CIC
def evaluate_multi_captions_from_file(generated_captions_path, output_dir, gt_captions_path, metrics, db_path,
                                      model_name, num_regions, eval_mode='all', evaluate_gt=False, shared_ids=None):
    with open(generated_captions_path, 'rt') as captions_file:
        captions_data = json.load(captions_file)
        all_captions = captions_data['generated_captions']
        if 'next_positions' in captions_data:
            next_positions = captions_data['next_positions']
        else:
            next_positions = None

    # Only evaluate the shared examples
    if shared_ids is not None:
        all_captions = {example_id: all_captions[example_id] for example_id in shared_ids}

    # Setup evaluation
    eval = EvaluateCaptions()
    eval.setup(metrics=metrics,
               gt_path=gt_captions_path,
               example_ids=None)

    num_eval_captions = num_regions if (eval_mode == 'same') else 0

    # Run standard metrics calculations for all versions of region sequences
    details_all = None
    total_num_captions = 0
    for i_annotation in range(5):
        current_captions = {example_id[:-2]: all_captions[example_id] for example_id in all_captions
                            if example_id[-1] == str(i_annotation)}
        if len(current_captions) > 0:
            _, current_details, num_captions = eval.evaluate_candidate_captions(current_captions,
                                                                                evaluate_gt=evaluate_gt,
                                                                                num_regions=num_eval_captions)
            if current_details is None:
                continue

            total_num_captions += num_captions

            # Initialize the dict
            if details_all is None:
                details_all = _initialize_details_dict(current_details)

            # Transfer the individual scores into the lists in details_all
            _add_individual_scores(details_all, current_details)

    if details_all is None:
        print(f"WARNING: No gt to compare to for model={model_name}, num_regions={num_regions}, eval_mode={eval_mode}")
        return

    print(f"Evaluated {total_num_captions} out of {len(all_captions)} captions.")

    # Calculate the average score for each individual image
    details_all = {metric:
                   {image_id: np.nanmean(np.asarray(details_all[metric][image_id]))
                    for image_id in details_all[metric]}
                   for metric in details_all}

    # Calculate the average scores for each metric
    scores_all = {metric: np.nanmean(np.asarray(list(details_all[metric].values())))
                  for metric in details_all}

    # Calculate diversity metrics
    store_captions_in_table(db_path=db_path, table_name="generated_captions", example_captions=all_captions)
    novel = calculate_novelty(db_path=db_path, table_name_gen="generated_captions", table_name_gts="gt_train")
    diversity = calculate_distinct(db_path=db_path, table_name="generated_captions")
    vocab = calculate_vocabulary_usage(db_path=db_path, table_name="generated_captions")
    cap_length = calculate_caption_lengths(db_path=db_path, table_name="generated_captions")

    if next_positions is None:
        num_chunks = '-'
    else:
        num_chunks = average_num_chunks(captions=all_captions, next_positions=next_positions)

    # Save results to csv file
    csv_row = list()
    csv_row.append(model_name)  # model_name
    csv_row.append(num_regions)  # num_regions
    csv_row.extend([scores_all[metric] for metric in metrics])  # <metric>
    csv_row.append(novel)  # novel
    csv_row.append(diversity)  # distinct
    csv_row.append(vocab)  # vocab_size
    csv_row.append(cap_length)  # length
    csv_row.append(num_chunks)  # num_chunks
    csv_row.append(len(all_captions))  # num_generated_captions
    csv_row.append(total_num_captions)  # num_evaluated_captions
    csv_row.append(eval_mode)  # eval_mode

    output_path = os.path.join(output_dir, f"METRICS_{model_name}_{num_regions}.csv")
    if os.path.exists(output_path):
        file_mode = 'at'
    else:
        file_mode = 'wt'

    with open(output_path, file_mode) as f:
        writer = csv.writer(f, delimiter=';')
        if file_mode == 'wt':
            header = ['model_name', 'num_regions'] + metrics + ['novel', 'distinct', 'vocab_size', 'length',
                                                                'num_chunks', 'num_generated_captions',
                                                                'num_evaluated_captions', 'eval_mode']
            writer.writerow(header)
        writer.writerow(csv_row)


def average_num_chunks(captions, next_positions):
    all_num_chunks = list()
    for example_id in captions:
        # Every chunk ends with NEXTTOKEN, so we have at least as many chunks as NEXTTOKENs
        num_chunks = len(next_positions[example_id])
        # Add the number of NEXTTOKENs to the number of word tokens in this caption to get the total length
        caption_full_length = len(captions[example_id]) + num_chunks
        # If there is no final region-less chunk, then the last NEXTTOKEN position should be at the last index
        if ((num_chunks == 0) and (caption_full_length > 0)) or ((caption_full_length-1) > next_positions[example_id][-1]):
            num_chunks += 1

        all_num_chunks.append(num_chunks)

    return np.asarray(all_num_chunks).mean()


def _create_dataloader(example_ids, num_regions, use_preprocessed_cic_data=False):
    if use_preprocessed_cic_data:
        if _CONFIG['gt_regions'] == "all":
            image_dir = None
            batch_size = _CONFIG['cic_batch_size']
        else:
            image_dir = _CONFIG['image_dir']
            batch_size = 1

        dataset = PreprocessedDataset(example_ids=example_ids,
                                      data_dir=_CONFIG['cic_dir'],
                                      image_dir=image_dir,
                                      num_regions=num_regions)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=_CONFIG['num_dataloader_workers'],
                                collate_fn=CollatePreprocessedData(
                                    region_feature_size=2053))
    else:
        if _CONFIG['gt_regions'] == "candidates" or _CONFIG['gt_regions'] == "matched":
            raw_dir = _CONFIG['raw_dir']
        else:
            raw_dir = None

        dataset = RawDataset(example_ids=example_ids, image_dir=_CONFIG['image_dir'], raw_dir=raw_dir)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                                num_workers=_CONFIG['num_dataloader_workers'],
                                collate_fn=lambda batch: batch[0])  # return the first (and only) example

    return dataloader


def _create_caption_generator():
    cg = CaptionGenerator(model_type='region_attention',
                          vocabulary_path=_CONFIG['vocabulary_path'],
                          word_embedding_size=1024,
                          visual_feature_size=2048,
                          spatial_feature_size=5,
                          hidden_size=1024,
                          use_all_regions=_CONFIG['use_all_regions'] == 'enforced',
                          inference_only=True,
                          num_layers=2,
                          learning_rate=0,
                          dropout_lstm=0.7,
                          dropout_word_embedding=0.7,
                          l2_weight=0,
                          block_unnecessary_tokens=True,
                          device=_CONFIG['device'])

    if _CONFIG['load_path_cg'] is not None:
        print("Starting from PATH", _CONFIG['load_path_cg'])
        cg.load(checkpoint_path=_CONFIG['load_path_cg'], load_optimizer=False)

    cg.set_mode('test')
    return cg


def _load_region_ranker():
    if _CONFIG["gt_regions"] == "matched":
        return None

    ranking_model = None

    ranking_type = _CONFIG["region_ranker_type"]
    if ranking_type == "nn_iou":
        model_path = _CONFIG["region_ranker_model_path"]

        last_index = model_path.rfind('/')
        outpath = model_path[:last_index + 1] + "settings.json"
        with open(outpath, 'rt') as infile:
            settings = json.load(infile)

        ranking_model = RegionRankerNNIoU(**settings, enable_training=False, device=_CONFIG["device"])
        ranking_model.load(model_path, load_optimizer=False)
        ranking_model.set_mode("eval")
    elif ranking_type == "object_probs":
        ranking_model = RegionRankerObjectProbs()

    assert ranking_model is not None, "Unknown Region Selector type: {}".format(ranking_type)

    return ranking_model


def _get_ordering_function():
    if _CONFIG["gt_regions"] == "matched":
        return None

    ordering_function = None

    region_ordering_type = _CONFIG["region_ordering_type"]

    if region_ordering_type == 'sinkhorn':
        model_path = _CONFIG["region_ordering_model_path"]
        ordering_function = SinkhornOrdering(device=_CONFIG["device"],
                                             class_embeddings_path=_CONFIG["class_embeddings_path"])
        ordering_function.load(model_path, load_optimizer=False)
    elif region_ordering_type == 'rulebased':
        ordering_function = order_rulebased
    elif region_ordering_type == 'identity':
        ordering_function = IdentityOrder()
    else:
        assert False, "Unknown region_ordering_type: " + region_ordering_type

    return ordering_function


def _get_grouping_model():
    if _CONFIG["gt_regions"] == "matched":
        return None

    bb_grouping_type = _CONFIG["bb_grouping_type"]

    if bb_grouping_type == 'none':
        grouping_model = None
    elif bb_grouping_type == 'nn':
        model_path = _CONFIG["grouping_model_path"]
        last_index = model_path.rfind('/')
        outpath = model_path[:last_index + 1] + "settings.json"
        with open(outpath, 'rt') as infile:
            settings = json.load(infile)

        grouping_model = GroupingNN(num_features=settings["num_features"],
                                    num_hidden=settings["num_hidden"],
                                    confidence_threshold=_CONFIG["grouping_confidence_threshold"],
                                    device=_CONFIG["device"],
                                    gather_stats=_CONFIG["gather_grouping_stats"])
        grouping_model.load(model_path, load_optimizer=False)
        grouping_model.set_mode("eval")
    else:
        assert False, "Unknown bb_grouping_type: " + bb_grouping_type

    return grouping_model


def _export_grouping_hist(pair_scores, filepath):
    plt.hist(pair_scores, bins=100)
    plt.title("Pair scores across full dataset")
    plt.savefig(
        fname=filepath,
        dpi='figure',
        bbox_inches='tight'
    )
    plt.close()

    stats_file_path = filepath[:-3] + "json"
    with open(stats_file_path, "wt") as outfile:
        json.dump({'pair_scores': pair_scores}, outfile)


def _export_matched_boxes_hist(percent_matched, filepath):
    bins = np.linspace(-0.05, 1.05, 12)
    plt.hist(percent_matched, bins=bins, rwidth=0.9, range=[-0.05, 1.05])
    plt.xticks([current_bin+0.05 for current_bin in bins if current_bin < 1.0])
    plt.title("Percent of GT entities matched by RPN detections")
    plt.savefig(
        fname=filepath,
        dpi='figure',
        bbox_inches='tight'
    )
    plt.close()

    stats_file_path = filepath[:-3] + "json"
    with open(stats_file_path, "wt") as outfile:
        json.dump({'percent_matched': percent_matched}, outfile)


def store_captions(example_ids, predictions, next_positions, output_path):
    generated_captions = {k: [v] for k, v in zip(example_ids, predictions)}
    nexts = {k: v[0].tolist() for k, v in zip(example_ids, next_positions)}
    with open(output_path, 'wt') as outfile:
        json.dump({'generated_captions': generated_captions, 'next_positions': nexts}, outfile)


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    _CONFIG = parse_parameters(sys.argv[1:])

    torch.random.manual_seed(_CONFIG['seed'])

    # Pipeline with pre-processed region data
    if _CONFIG["mode"] == "pipeline" and _CONFIG["gt_regions"] != "none" and _CONFIG["gt_regions"] != "candidates" and _CONFIG["gt_regions"] != "matched":
        # The dataloader may be created later to take into account the current num_regions
        dataloader = None

        # Load the dataset splits
        if _CONFIG["gt_regions"] == "grouping":
            splits_type = "all"

            with open(_CONFIG['splits_path'], 'rt') as splits_file:
                example_ids = json.load(splits_file)['splits'][_CONFIG['split']]

            # Only keep the example ids that we have raw files for
            examples_files = os.listdir(_CONFIG["raw_dir"])
            example_ids = [example_id for example_id in example_ids if example_id + '_raw.json' in examples_files]

            # Restrict the number of examples in the dataset if requested
            if (_CONFIG['limit_test_examples'] > 0) and (_CONFIG['limit_test_examples'] < len(example_ids)):
                example_ids = example_ids[0:_CONFIG['limit_test_examples']]

            dataloader = _create_dataloader(example_ids=example_ids, num_regions=0, use_preprocessed_cic_data=True)
        else:
            if _CONFIG["gt_regions"] == "selection":
                splits_type = "unsorted"
            else:
                splits_type = "sequences"

        cg = _create_caption_generator()

        cnn = None
        if _CONFIG["gt_regions"] != "all":
            model_load_paths = [os.path.join(_CONFIG['cnn_model_dir'], 'test.prototxt'),
                                os.path.join(_CONFIG['cnn_model_dir'],
                                             'resnet101_faster_rcnn_final_iter_380000.caffemodel'),
                                os.path.join(_CONFIG['cnn_model_dir'], 'caffe.proto')]

            region_ordering = _get_ordering_function()
            region_ranker = None
            if _CONFIG["gt_regions"] != "selection":
                region_ranker = _load_region_ranker()

            cnn = VisionNetwork(model_load_paths=model_load_paths,
                                bb_grouping=None, region_ranker=region_ranker,
                                region_ordering=region_ordering, vg_to_flickr30k_path=_CONFIG["vg_to_flickr30k_path"],
                                device=_CONFIG['device'])

        if (_CONFIG["num_regions_min"] == 0) and (_CONFIG["num_regions_max"] == 0):
            all_num_regions = [0]
            print("Generating captions using all available regions.")
        else:
            all_num_regions = range(_CONFIG["num_regions_min"], _CONFIG["num_regions_max"]+1)
            print(f"Generating captions using num_regions from {_CONFIG['num_regions_min']} to {_CONFIG['num_regions_max']}.")

        for num_regions in all_num_regions:
            if splits_type != "all":
                # We need to recreate the dataloader with the unique split for each number of regions
                with open(os.path.join(_CONFIG['multi_splits_dir'], f"splits_unique_{splits_type}_{num_regions}.json"), 'rt') as f:
                    example_ids = json.load(f)["splits"][_CONFIG["split"]]

                # Restrict the number of examples in the dataset if requested
                if (_CONFIG['limit_test_examples'] > 0) and (_CONFIG['limit_test_examples'] < len(example_ids)):
                    example_ids = example_ids[0:_CONFIG['limit_test_examples']]

                dataloader = _create_dataloader(example_ids=example_ids, num_regions=num_regions,
                                                use_preprocessed_cic_data=True)

            # display_boxes_dir = os.path.join(_CONFIG["results_dir"], "boxes",
            #                                 _CONFIG["model_name"] + '_' + str(num_regions))
            # os.makedirs(display_boxes_dir, exist_ok=True)
            # export_params = {"output_dir": display_boxes_dir, "image_dir": _CONFIG["image_dir"]}
            export_params = None

            example_ids, predictions, next_positions = run_cic_gt_pipeline(dataloader, cnn, cg,
                                                                           num_regions=num_regions,
                                                                           allow_fewer_regions=_CONFIG["allow_fewer_regions"],
                                                                           print_results=True,
                                                                           export_params=export_params)

            # Store the generated captions
            output_path = os.path.join(_CONFIG["results_dir"], 'CAPTIONS_' + _CONFIG["model_name"] + "_"
                                       + str(num_regions) + '.json')
            store_captions(example_ids, predictions, next_positions, output_path)

    # Pipeline starting from raw image pixel data
    elif _CONFIG["mode"] == "pipeline":
        # The dataloader may be created later to take into account the current num_regions
        dataloader = None

        model_load_paths = [os.path.join(_CONFIG['cnn_model_dir'], 'test.prototxt'),
                            os.path.join(_CONFIG['cnn_model_dir'], 'resnet101_faster_rcnn_final_iter_380000.caffemodel'),
                            os.path.join(_CONFIG['cnn_model_dir'], 'caffe.proto')]

        region_ranker = _load_region_ranker()
        region_ordering = _get_ordering_function()

        grouping_model = _get_grouping_model()
        cnn = VisionNetwork(model_load_paths=model_load_paths,
                            bb_grouping=grouping_model, region_ranker=region_ranker,
                            region_ordering=region_ordering, vg_to_flickr30k_path=_CONFIG["vg_to_flickr30k_path"],
                            device=_CONFIG['device'])

        # Load the dataset splits
        if _CONFIG['gt_regions'] != "matched":
            with open(_CONFIG['splits_path'], 'rt') as splits_file:
                example_ids = json.load(splits_file)['splits'][_CONFIG['split']]

            # Only keep the example ids that we have raw files for
            examples_files = os.listdir(_CONFIG["raw_dir"])
            example_ids = [example_id for example_id in example_ids if example_id + '_raw.json' in examples_files]

            # Restrict the number of examples in the dataset if requested
            if (_CONFIG['limit_test_examples'] > 0) and (_CONFIG['limit_test_examples'] < len(example_ids)):
                example_ids = example_ids[0:_CONFIG['limit_test_examples']]

            dataloader = _create_dataloader(example_ids=example_ids, num_regions=0)

        cg = _create_caption_generator()

        if (_CONFIG["num_regions_min"] == 0) and (_CONFIG["num_regions_max"] == 0):
            all_num_regions = [0]
            print("Generating captions using all available regions.")
        else:
            all_num_regions = range(_CONFIG["num_regions_min"], _CONFIG["num_regions_max"]+1)
            print(f"Generating captions using num_regions from {_CONFIG['num_regions_min']} to {_CONFIG['num_regions_max']}.")

        for num_regions in all_num_regions:
            if _CONFIG['gt_regions'] == "matched":
                with open(os.path.join(_CONFIG['multi_splits_dir'], f"splits_unique_sequences_{num_regions}.json"), 'rt') as f:
                    example_ids = json.load(f)["splits"][_CONFIG["split"]]

                # Restrict the number of examples in the dataset if requested
                if (_CONFIG['limit_test_examples'] > 0) and (_CONFIG['limit_test_examples'] < len(example_ids)):
                    example_ids = example_ids[0:_CONFIG['limit_test_examples']]

                dataloader = _create_dataloader(example_ids=example_ids, num_regions=num_regions)

            display_boxes_dir = None
            # display_boxes_dir = os.path.join(_CONFIG["results_dir"], "boxes",
            #                                 _CONFIG["model_name"] + '_' + str(num_regions))
            # os.makedirs(display_boxes_dir, exist_ok=True)

            if _CONFIG["gt_regions"] == "matched":
                example_ids, predictions, next_positions = run_box_matched_pipeline(dataloader, cnn, cg,
                                                                                    num_regions=num_regions,
                                                                                    allow_fewer_regions=_CONFIG["allow_fewer_regions"],
                                                                                    display_boxes_dir=display_boxes_dir,
                                                                                    print_results=True)
            else:
                example_ids, predictions, next_positions = run_pipeline(dataloader, cnn, cg, num_regions=num_regions,
                                                                        allow_fewer_regions=_CONFIG["allow_fewer_regions"],
                                                                        display_boxes_dir=display_boxes_dir,
                                                                        print_results=True)

            if _CONFIG["gather_grouping_stats"] and grouping_model is not None:
                hist_dir = os.path.join(_CONFIG["results_dir"], "group_hist")
                os.makedirs(hist_dir, exist_ok=True)
                hist_path = os.path.join(hist_dir, f"{_CONFIG['model_name']}_{str(num_regions)}_{_CONFIG['split']}.png")
                _export_grouping_hist(pair_scores=grouping_model.all_scores, filepath=hist_path)

            # Store the generated captions
            output_path = os.path.join(_CONFIG["results_dir"], 'CAPTIONS_' + _CONFIG["model_name"] + "_"
                                       + str(num_regions) + '.json')
            store_captions(example_ids, predictions, next_positions, output_path)

    elif _CONFIG["mode"].startswith("evaluate"):
        if _CONFIG["eval_mode"] == 'all':
            print("Evaluating captions on all available gt captions.")
        elif _CONFIG["eval_mode"] == 'same':
            print(f"Evaluating captions on gt with num_regions from {_CONFIG['num_regions_min']} to {_CONFIG['num_regions_max']}.")
        else:
            assert False, f"ERROR: Unknown eval_mode: {_CONFIG['eval_mode']}"

        db_path = _CONFIG["diversity_db_path"]
        if not os.path.exists(_CONFIG["diversity_db_path"]):
            # Create the sql db and store the GT captions for the train split (used for diversity metrics)
            with open(_CONFIG["gt_captions_path"] + "_train.json") as gt_file:
                gt_caption_data = json.load(gt_file)
            store_captions_in_table(db_path=db_path, table_name="gt_train",
                                    example_captions=gt_caption_data["all"])

        if _CONFIG["mode"] == "evaluate_gt":
            model_name = 'gt'
        else:
            model_name = _CONFIG["model_name"]

        if (_CONFIG["num_regions_min"] == 0) and (_CONFIG["num_regions_max"] == 0):
            all_num_regions = [0]
        else:
            all_num_regions = range(_CONFIG["num_regions_min"], _CONFIG["num_regions_max"]+1)

        if len(_CONFIG["shared_examples_path"]) > 0:
            print("Evaluating on shared ids from", _CONFIG["shared_examples_path"])
            with open(_CONFIG["shared_examples_path"]) as f:
                all_shared_ids = json.load(f)
        else:
            all_shared_ids = None

        for num_regions in all_num_regions:
            captions_path = os.path.join(_CONFIG["results_dir"], f"CAPTIONS_{model_name}_{num_regions}.json")
            output_dir = os.path.dirname(captions_path)
            shared_ids = None
            if all_shared_ids is not None:
                shared_id_models = os.path.basename(_CONFIG["shared_examples_path"][len("SHARED_IDS_"):].split('.')[0])
                output_dir = os.path.join(output_dir, shared_id_models)
                os.makedirs(output_dir, exist_ok=True)

            if _CONFIG["mode"] == "evaluate_model":
                if all_shared_ids is not None:
                    try:
                        shared_ids = all_shared_ids["image_ids"][str(num_regions)]
                    except KeyError:
                        continue

                evaluate_captions_from_file(generated_captions_path=captions_path,
                                            output_dir=output_dir,
                                            gt_captions_path=_CONFIG["gt_captions_path"] + "_" + _CONFIG["split"] + ".json",
                                            metrics=_CONFIG["metrics"],
                                            db_path=db_path,
                                            model_name=model_name,
                                            num_regions=num_regions,
                                            eval_mode=_CONFIG["eval_mode"],
                                            shared_ids=shared_ids)
            elif (_CONFIG["mode"] == "evaluate_model_multi") or (_CONFIG["mode"] == "evaluate_gt"):
                if all_shared_ids is not None:
                    try:
                        shared_ids = all_shared_ids["example_ids"][str(num_regions)]
                    except KeyError:
                        continue

                evaluate_multi_captions_from_file(generated_captions_path=captions_path,
                                                  output_dir=output_dir,
                                                  gt_captions_path=_CONFIG["gt_captions_path"] + "_" + _CONFIG["split"] + ".json",
                                                  metrics=_CONFIG["metrics"],
                                                  db_path=db_path,
                                                  model_name=model_name,
                                                  num_regions=num_regions,
                                                  eval_mode=_CONFIG["eval_mode"],
                                                  evaluate_gt=(_CONFIG["mode"] == "evaluate_gt"),
                                                  shared_ids=shared_ids)
