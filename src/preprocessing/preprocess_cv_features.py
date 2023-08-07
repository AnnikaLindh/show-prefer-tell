# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Show, Prefer and Tell: Incorporating User Preferences into Image Captioning (Lindh, Ross & Kelleher, 2023)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/show-prefer-tell
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from __future__ import division

import os
from os import path as os_path
import re
import numpy as np
import json
import torch
from captioner.vision_network import VisionData, draw_rectangles, box_areas, calculate_iou
import matplotlib


# Prevent errors due to lack of display on the server nodes
matplotlib.use('Agg')
from matplotlib import pyplot as plt


_FLICKR30K_CATEGORIES = {'people': 0,
                         'animals': 1,
                         'instruments': 2,
                         'vehicles': 3,
                         'other': 4,
                         'clothing': 5,
                         'bodyparts': 6}


def _export_hist(data, filepath, title, bins=100):
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.savefig(
        fname=filepath,
        dpi='figure',
        bbox_inches='tight'
    )
    plt.close()


# (x_max - x_min) / 2 + x_min = x center   (y_max - y_min) / 2 + y_min = y center
def box_relative_centers(boxes, image_half_width, image_half_height):
    x_center = (boxes[:, 2] + boxes[:, 0]) / 2
    x_relative_center = np.abs(x_center - image_half_width) / image_half_width

    y_center = (boxes[:, 3] + boxes[:, 1]) / 2
    y_relative_center = np.abs(y_center - image_half_height) / image_half_height

    return x_relative_center, y_relative_center


class PrepareIoUSelectionTrainingData:
    def __init__(self, dataloader, model, filter_class_type, confidence_filter, class_nms, iou_min, iou_max,
                 output_dir, vg_to_flickr30k_path, device):
        self.dataloader = dataloader
        self.model = model
        self.filter_class_type = filter_class_type
        self.confidence_filter = confidence_filter
        self.class_nms = class_nms
        self.iou_min = iou_min
        self.iou_max = iou_max
        self.output_dir = output_dir
        self.device = device

        # Load the data to translate the vision network's categories into Flickr30k categories
        with open(vg_to_flickr30k_path) as infile:
            self.vg_idx_to_flickr30k_idx = json.load(infile)["vg_idx_to_flickr30k_idx"]

        if not os_path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def preprocess_all(self):
        for batch in self.dataloader:
            if batch is None:
                continue

            image_id = batch["example_id"]
            img = batch["image"]
            all_bbs = batch["all_bbs"]
            gt_entities = batch["gt_entities"]

            num_accepted = len(gt_entities)

            # Skip any examples without GT entity boxes since this would leave us with zero positive examples
            if num_accepted == 0:
                print("No GT entity boxes for image:", image_id)
                continue

            # Automatically detect boxes with the RPN and extract features from the c2p model
            vision_data_rpn = self._extract_rpn_features(img)

            if vision_data_rpn is None:
                print("No DETECTIONS for image:", image_id)
                continue

            rpn_boxes = vision_data_rpn.rois
            categories = vision_data_rpn.object_classes_30k
            probs = vision_data_rpn.object_probs_30k
            feature_maps = vision_data_rpn.feature_maps

            # Merge gt bbs into entity boxes, and keep track of partial boxes
            entity_boxes, partial_boxes, partial_bb_entities = self._derive_entity_boxes(gt_entities, all_bbs)

            # Calculate the iou between rpn_boxes and full GT entities [num_rpn, num_gt]
            entity_iou = calculate_iou(rpn_boxes, entity_boxes)

            # Get the max IoU for each RPN to any GT box
            entity_iou, _ = entity_iou.max(dim=1)

            # Find any additional matches between grouped rpn-boxes and entity boxes
            extra_matches = None
            if partial_boxes is not None:
                extra_matches = self._calculate_grouped_iou(
                    rpn_boxes=rpn_boxes,
                    rpn_categories=categories,
                    rpn_feature_maps=feature_maps,
                    rpn_probs=probs,
                    gt_partial_boxes=partial_boxes,
                    partial_bb_entities=partial_bb_entities,
                    gt_entity_boxes=entity_boxes)

            # Combine the entity info with the extra info
            if extra_matches is not None:
                # Unpack the returned values
                extra_boxes, extra_iou, extra_categories, extra_feature_maps, extra_probs = extra_matches

                # Add the extra values onto the entity values
                rpn_boxes = torch.cat([rpn_boxes] + extra_boxes, dim=0)
                entity_iou = torch.cat([entity_iou] + extra_iou, dim=0)
                categories = torch.cat([categories] + extra_categories, dim=0)
                feature_maps = torch.cat([feature_maps] + extra_feature_maps, dim=0)
                probs = torch.cat([probs] + extra_probs, dim=0)

            # Turn the iou matrix into labels with 0.0 and 1.0 values
            if self.iou_min is not None:
                entity_iou[entity_iou < self.iou_min] = 0.0
            if self.iou_max is not None:
                entity_iou[entity_iou > self.iou_max] = 1.0

            # Convert to numpy
            rpn_boxes = rpn_boxes.cpu().numpy()
            entity_iou = entity_iou.cpu().numpy()
            categories = categories.cpu().numpy()
            feature_maps = feature_maps.cpu().numpy()
            probs = probs.cpu().numpy()

            # Box centers relative to image height and width
            relative_centers_x, relative_centers_y = box_relative_centers(rpn_boxes,
                                                                          image_half_width=img.shape[1]/2,
                                                                          image_half_height=img.shape[0]/2)

            # Relative box area compared to full image area
            image_area = img.shape[0] * img.shape[1]
            relative_areas = box_areas(rpn_boxes) / image_area

            # Flickr30k category for each entity as one hot encodings
            num_boxes = len(categories)
            one_hot_categories = np.zeros([num_boxes, 7], dtype=np.float)
            one_hot_categories[list(range(num_boxes)), categories] = 1.0

            # Combine all features into a single matrix with all features
            features = np.concatenate([np.expand_dims(probs, axis=1),
                                       np.expand_dims(relative_centers_x, axis=1),
                                       np.expand_dims(relative_centers_y, axis=1),
                                       np.expand_dims(relative_areas, axis=1),
                                       one_hot_categories,
                                       feature_maps],
                                      axis=1)

            output_path = os_path.join(self.output_dir, image_id + '_entity_selection.npz')
            np.savez(output_path, features=features, labels=entity_iou)

    def _extract_rpn_features(self, image):
        img_tensor, im_info, _ = self.model.prepare_input(image=image, bboxes=None)
        img_tensor = img_tensor.to(device=self.device)
        im_info = im_info.to(device=self.device)

        clear_except = {"rpn_conv/3x3": ["im_info", "res4b22"],
                        "res5a_branch1": ["roipool5", "rois"],
                        "res5c": ["rois"],
                        "res5c_branch2b_relu": ["rois", "res5b"]}

        _ = self.model.forward(data=img_tensor, clear_except=clear_except, im_info=im_info)
        blobs = self.model.model.blobs

        # Drop the levels column and scale rois to the original scale
        rois = blobs["rois"].data_[:, 1:] / im_info[2]
        if len(rois) == 0:
            return None

        # Get the object probabilities
        object_probs_all = blobs["cls_prob"].data_
        object_probs_all = torch.reshape(object_probs_all, shape=(len(object_probs_all), -1,))

        # Get the predicted classes and then drop the background probs
        object_probs_max, object_classes = torch.max(object_probs_all, dim=1)
        object_probs_all = object_probs_all[:, 1:]

        # Gather all features into an object
        vision_data = VisionData(im_info=im_info, full_image_features=None, rois=rois,
                                 feature_maps=blobs["pool5_flat"].data_, object_probs_max=object_probs_max,
                                 object_probs_all=object_probs_all, object_classes=object_classes)

        # Only keep boxes with a higher probability in one of the non-background classes
        fg_idx = torch.nonzero(object_classes != 0).squeeze(-1)
        if len(fg_idx) == 0:
            return None
        vision_data.filter(keep_idx=fg_idx)

        # Compensate for including the background class in torch.max
        vision_data.object_classes += 1

        if self.filter_class_type == "flickr30k":
            # Derive the flickr30k classes from the vg classes
            vision_data.derive_flickr30k_object_classes(vg_idx_to_flickr30k_idx=self.vg_idx_to_flickr30k_idx,
                                                        device=self.device)

            probs = vision_data.object_probs_30k
        else:
            probs = vision_data.object_probs_max

        # Only allow boxes with a minimum confidence score
        keep_idx = torch.nonzero(probs > self.confidence_filter).squeeze(-1)
        if len(keep_idx) == 0:
            return None
        vision_data.filter(keep_idx=keep_idx)

        # Perform class-sensitive NMS on the boxes with predicted classes
        vision_data.class_nms(iou_threshold=self.class_nms, class_type=self.filter_class_type)

        return vision_data

    def _derive_entity_boxes(self, gt_entities, gt_bbs):
        entity_boxes = list()
        partial_boxes = list()
        partial_bb_entities = list()
        for i_entity in range(len(gt_entities)):
            bb_ids = gt_entities[i_entity]
            if len(bb_ids) == 1:
                entity_box = torch.tensor(gt_bbs[bb_ids[0]], device=self.device, dtype=torch.float32)
            else:
                # Combine the group entity boxes into one large box that encompasses all of them
                entity_box = [np.min(gt_bbs[bb_ids, 0]), np.max(gt_bbs[bb_ids, 1]),
                              np.min(gt_bbs[bb_ids, 2]), np.max(gt_bbs[bb_ids, 3])]
                entity_box = torch.tensor(entity_box, device=self.device, dtype=torch.float32)

                # Also add the individual boxes to the partial boxes to simulate automatic grouping
                partial_bb_entities.extend([i_entity for _ in bb_ids])
                partial_boxes.append(torch.tensor(gt_bbs[bb_ids], device=self.device, dtype=torch.float32))

            entity_boxes.append(entity_box)

        entity_boxes = torch.stack(entity_boxes, dim=0)
        if len(partial_boxes) > 0:
            partial_boxes = torch.cat(partial_boxes, dim=0)
            partial_bb_entities = torch.tensor(partial_bb_entities, device=self.device)
        else:
            partial_boxes = None
            partial_bb_entities = None

        return entity_boxes, partial_boxes, partial_bb_entities

    def _calculate_grouped_iou(self, rpn_boxes, rpn_categories, rpn_feature_maps, rpn_probs,
                               gt_partial_boxes, partial_bb_entities, gt_entity_boxes):
        # Get the iou of each pair of rpn-box and partial gt-box [num_rpn, num_partial_gt]
        partial_iou = calculate_iou(rpn_boxes, gt_partial_boxes)

        merged_rpn_boxes = list()
        merged_feature_maps = list()
        merged_categories = list()
        merged_probs = list()
        matched_rpn_index_sets = list()
        double_matched_sets = list()
        matched_entity_boxes = list()

        # Only allow grouping of the same category for the RPN boxes
        unique_cats, unique_counts = torch.unique(rpn_categories, return_counts=True)
        for (cat, count) in zip(unique_cats, unique_counts):
            if count < 2:
                continue

            current_rpn_idx = (rpn_categories == cat).nonzero(as_tuple=True)[0]

            # Find if any of the gt entities can be matched to a merged rpn box that overlap with the gt partial boxes
            for gt_entity in partial_bb_entities:
                current_gt_idx = torch.nonzero((partial_bb_entities == gt_entity), as_tuple=True)[0]

                # Find matches between current rpn boxes and each of the current gt boxes
                rpn_rows = current_rpn_idx.repeat_interleave(len(current_gt_idx))
                gt_cols = current_gt_idx.repeat(len(current_rpn_idx))
                rpn_matched_idx = self._match_boxes_by_iou(partial_iou, rpn_rows, gt_cols, min_iou=self.class_nms)

                # Only count it as a successful match if at least 2 of the partial boxes have been matched
                if rpn_matched_idx is None or len(rpn_matched_idx) < 2:
                    continue

                # Merge these rpn boxes into a grouped region
                merged_rpn_box = torch.tensor([[torch.min(rpn_boxes[rpn_matched_idx, 0]),
                                                torch.max(rpn_boxes[rpn_matched_idx, 1]),
                                                torch.min(rpn_boxes[rpn_matched_idx, 2]),
                                                torch.max(rpn_boxes[rpn_matched_idx, 3])]],
                                              device=self.device)

                # Take the average of the individual box features
                merged_feature_map = torch.mean(rpn_feature_maps[rpn_matched_idx, :], dim=0, keepdim=True)
                merged_feature_maps.append(merged_feature_map)

                # Take the max of the individual probs for this grouped entity
                current_merged_prob, _ = torch.max(rpn_probs[rpn_matched_idx], dim=0, keepdim=True)

                # Make sure we don't end up with two different IoUs for the same exact merged box
                current_match = set(rpn_matched_idx)
                if (current_match in matched_rpn_index_sets) and (current_match not in double_matched_sets):
                    double_matched_sets.append(current_match)

                # Add all the info about this current match
                matched_rpn_index_sets.append(current_match)
                merged_rpn_boxes.append(merged_rpn_box)
                merged_categories.append(cat.unsqueeze(0))
                merged_probs.append(current_merged_prob)
                matched_entity_boxes.append(gt_entity_boxes[gt_entity:gt_entity+1, :])

        if len(merged_rpn_boxes) == 0:
            return None

        # Calculate the IoU for each matched pair (these will be stored along the diagonal of the returned matrix)
        iou = calculate_iou(torch.cat(merged_rpn_boxes), torch.cat(matched_entity_boxes))

        # Add each matched merged box along with its best iou value
        matched_boxes = list()
        matched_ious = list()
        matched_categories = list()
        matched_feature_maps = list()
        matched_probs = list()
        skip_indices = list()
        for i_match in range(len(matched_rpn_index_sets)):
            if i_match in skip_indices:
                continue

            # Get the max IoU for this merged rpn box if it was matched against multiple gt boxes
            matched_iou = iou[i_match, i_match]
            matched_idx = matched_rpn_index_sets[i_match]
            if matched_idx in double_matched_sets:
                for i_other_match in range(i_match+1, len(matched_rpn_index_sets)):
                    if matched_rpn_index_sets[i_other_match] == matched_idx:
                        matched_iou = torch.maximum(matched_iou, iou[i_other_match, i_other_match])
                        skip_indices.append(i_other_match)

            # Add the best matched IoU and the other matched info
            matched_boxes.append(merged_rpn_boxes[i_match])
            matched_ious.append(matched_iou.unsqueeze(0))
            matched_categories.append(merged_categories[i_match])
            matched_feature_maps.append(merged_feature_maps[i_match])
            matched_probs.append(merged_probs[i_match])

        return matched_boxes, matched_ious, matched_categories, matched_feature_maps, matched_probs

    @staticmethod
    def _match_boxes_by_iou(iou, rpn_rows, gt_cols, min_iou):
        rpn_matches = list()
        gt_matches = list()

        iou_values, sorted_idx = torch.sort(iou[rpn_rows, gt_cols], descending=True)

        for current_iou, current_idx in zip(iou_values, sorted_idx):
            if current_iou < min_iou:
                break

            rpn_idx = rpn_rows[current_idx]
            if rpn_idx in rpn_matches:
                continue

            gt_idx = gt_cols[current_idx]
            if gt_idx in gt_matches:
                continue

            # Mark this RPN and GT as already matched
            rpn_matches.append(rpn_idx)
            gt_matches.append(gt_idx)

        if len(rpn_matches) == 0:
            rpn_matches = None
        else:
            rpn_matches = torch.stack(rpn_matches)

        return rpn_matches


class SinkhornPreprocessor:
    def __init__(self, dataloader, visual_network, output_dir, device):
        self.dataloader = dataloader
        self.visual_network = visual_network
        self.output_dir = output_dir
        self.device = device

        # Keep track of which examples are valid for sinkhorn training so we can create new splits for this
        self.valid_examples = list()

    def preprocess(self):
        for batch in self.dataloader:
            example_id = batch["example_id"]
            region_features = batch["region_features"]
            spatial_features = batch["spatial_features"]

            # Extract the class probabilities from the region_features

            # Skip any examples with less than 2 GT boxes (indicated by None by the dataloader)
            if region_features is None:
                print("Need at least 2 GT boxes to train ordering. Skipping example:", example_id)
                continue

            # print("Extracting features...")
            # Extract feature maps and predicted classes from the visual network
            object_probs = self._extract_features(region_features)
            # print("...features extracted!")

            # Save the top 3 probs and classes for each region and convert to numpy
            object_probs, object_classes = torch.sort(object_probs, dim=1, descending=True)
            object_probs = object_probs[:, :3].cpu().numpy()
            object_classes = object_classes[:, :3].cpu().numpy()

            # Save this example's features
            self.valid_examples.append(example_id)
            np.savez(os_path.join(self.output_dir, str(example_id) + '.npz'),
                     region_features=region_features, spatial_features=spatial_features, object_probs=object_probs,
                     object_classes=object_classes)

    def _extract_features(self, region_features):
        region_features_tensor = torch.tensor(region_features)
        region_features_tensor = region_features_tensor.reshape(len(region_features), -1, 1, 1)
        override_variables = {"pool5": region_features_tensor.to(device=self.device)}

        start_from = "pool5_flat"

        _ = self.visual_network.forward(data=None, override_variables=override_variables, start_from=start_from)
        blobs = self.visual_network.model.blobs

        object_probs = blobs["cls_prob"].data_
        # Sort out the dimensions and drop the background class
        object_probs = object_probs.reshape(len(object_probs), -1)[:, 1:]

        return object_probs

    def export_new_splits(self, output_path, original_splits):
        sinkhorn_splits = dict()
        for split in original_splits:
            sinkhorn_splits[split] = [example_id for example_id in original_splits[split]
                                      if example_id in self.valid_examples]

        with open(output_path, 'wt') as f:
            json.dump({'splits': sinkhorn_splits}, f)


def export_vocab_embeddings_matrix(embeddings_name, output_dir, classes_path, embeddings_path):
    with open(classes_path, 'rt') as classes_file:
        # Get a list of lists where the outer list is the full vocab while the inner lists contain variations
        vocab = [line.strip().split(',') for line in classes_file.readlines()]

    with open(embeddings_path, 'rt') as embeddings_file:
        # Get a list of lists where each inner list has the word followed by each column's float value as a string
        embeddings = [line.strip().split(' ') for line in embeddings_file.readlines()]

    # Turn the list of lists with string embeddings into a dict with torch float embeddings
    embeddings_lookup = {embedding_line[0]: torch.tensor([float(s_float) for s_float in embedding_line[1:]],
                                                         dtype=torch.float32)
                         for embedding_line in embeddings}

    # Store the ordered embeddings for our class vocab (and later stack into a torch matrix)
    embeddings_matrix = list()

    # Report missing words from the vocab
    missing_words = list()

    for words in vocab:
        current_embedding = None
        remaining_words = list()
        for word in words:
            # The VG vocab includes words such as "tennis racket" but GLoVe only includes single words
            sub_words = word.split(' ')
            word = sub_words[-1]
            if len(sub_words) > 1:
                remaining_words.append(sub_words[0])

            try:
                current_embedding = embeddings_lookup[word]
                break
            except KeyError:
                continue

        if current_embedding is None:
            # Try the first parts of the current words (if any)
            for word in remaining_words:
                try:
                    current_embedding = embeddings_lookup[word]
                    break
                except KeyError:
                    continue

            # If there is still no match, use the embedding for "unk" and mark this word as missing
            if current_embedding is None:
                missing_words.append(words)
                current_embedding = embeddings_lookup["unk"]

        embeddings_matrix.append(current_embedding)

    # Convert the list into a torch matix
    embeddings_matrix = torch.stack(embeddings_matrix)

    # Export the matrix
    torch.save(embeddings_matrix, os_path.join(output_dir, embeddings_name + "_vocab_embeddings.pt"))

    # Export the list of missing words
    print("Number of vocabulary words replaced by unk: ", len(missing_words))
    missing_words = [','.join(words) + '\n' for words in missing_words]
    with open(os_path.join(output_dir, embeddings_name + "_missing_words.txt"), 'wt') as outfile:
        outfile.writelines(missing_words)


class PreprocessGroupTraining:
    def __init__(self, flickr30k_sentences_dir):
        self.flickr30k_sentences_dir = flickr30k_sentences_dir
        self.regex_entities = re.compile('/EN#(\d+)/(\S+)')

    def preprocess_all(self, dataloader, output_path):
        all_features = list()

        for batch in dataloader:
            if batch is None:
                continue

            image_id = batch["example_id"]
            all_bbs = batch["all_bbs"]
            gt_entities = batch["gt_entities"]
            bb_entities = batch["bb_entities"]

            all_areas = box_areas(all_bbs)

            with open(os_path.join(self.flickr30k_sentences_dir, image_id + ".txt")) as f:
                sentences = ' '.join(f.readlines())

            # Split entities into different types
            entities_by_type = dict()
            for (entity_id, entity_type) in re.findall(self.regex_entities, sentences):
                types = entity_type.split('/')
                for t in types:
                    if (t in ['clothing', 'bodyparts', 'scene']) or (entity_id not in gt_entities):
                        continue
                    try:
                        entities_by_type[t].append(entity_id)
                    except KeyError:
                        entities_by_type[t] = [entity_id]

            for entity_type in entities_by_type:
                # Keep track of our current_bb_ids but avoid adding the same bb more than once
                current_bb_ids = set()
                for entity_id in entities_by_type[entity_type]:
                    bbs_ids = gt_entities[entity_id]
                    current_bb_ids.update(bbs_ids)

                if len(current_bb_ids) < 2:
                    continue

                # For the rest of the logic it is important that current_bb_ids are in a consistently ordered list
                current_bb_ids = list(current_bb_ids)

                current_bbs = all_bbs[current_bb_ids, :]
                current_areas = all_areas[current_bb_ids]

                current_bb_entities = [bb_entities[idx] for idx in current_bb_ids]

                # Calculate the x-centers and y-centers of the bbs
                centers_x = (current_bbs[:, 0] + current_bbs[:, 2]) / 2.0
                centers_y = (current_bbs[:, 1] + current_bbs[:, 3]) / 2.0

                # Calculate the features for each pair when ordering the boxes from left to right
                x_order = np.argsort(centers_x, axis=None)
                features = calculate_group_pair_features(x_order, current_bbs, centers_x, centers_y, current_areas,
                                                         _FLICKR30K_CATEGORIES[entity_type])
                labels = calculate_group_pair_labels(x_order, current_bb_entities)

                # Calculate the features for any additional pairs when ordering the boxes from top to bottom
                y_order = np.argsort(centers_y, axis=None)
                new_pairs = [i for i in range(len(y_order)-1)
                             if ((x_order[i] != y_order[i]) or (x_order[i+1] != y_order[i+1]))]
                if len(new_pairs) > 0:
                    additional_features = calculate_group_pair_features(y_order, current_bbs, centers_x, centers_y,
                                                                        current_areas,
                                                                        _FLICKR30K_CATEGORIES[entity_type])
                    features = np.concatenate([features, additional_features[new_pairs]])
                    additional_labels = calculate_group_pair_labels(y_order, current_bb_entities)
                    labels = np.concatenate([labels, additional_labels[new_pairs]])

                # Add the data (features + labels) from our current entity type
                all_features.append(np.concatenate([features, labels], axis=1))

        all_features = np.concatenate(all_features, axis=0)

        print("Num pairs:", len(all_features))
        print("Percent of pairs that are labeled as should-be-grouped:", all_features[:, 3].mean() * 100.0)
        print("Min/max/mean for each feature:")
        feature_names = ["x_distance", "y_distance", "size_difference"]
        for i in range(len(feature_names)):
            print(feature_names[i], all_features[:, i].min(), all_features[:, i].max(), all_features[:, i].mean())

        np.save(output_path, all_features, allow_pickle=False)


def calculate_group_pair_features(order, current_bbs, centers_x, centers_y, bb_areas, category=None):
    # Get the two x centers for each pair
    first_center_x = centers_x[order[:-1]]
    second_center_x = centers_x[order[1:]]

    # Get the two y centers for each pair
    first_center_y = centers_y[order[:-1]]
    second_center_y = centers_y[order[1:]]

    # Calculate the mean x size of the two bounding boxes
    mean_x = np.mean(np.concatenate([current_bbs[order[:-1], 2:3] - current_bbs[order[:-1], 0:1],
                                     current_bbs[order[1:], 2:3] - current_bbs[order[1:], 0:1]],
                                    axis=1),
                     axis=1)

    # Calculate the mean y size of the two bounding boxes
    mean_y = np.mean(np.concatenate([current_bbs[order[:-1], 3:4] - current_bbs[order[:-1], 1:2],
                                     current_bbs[order[1:], 3:4] - current_bbs[order[1:], 1:2]],
                                    axis=1),
                     axis=1)

    # Calculate the x and y center distances relative to the pair's mean sizes in each of those axes
    x_distance = np.abs(second_center_x - first_center_x) / mean_x
    y_distance = np.abs(second_center_y - first_center_y) / mean_y

    # Calculate the difference in area size by doing smaller / larger
    pair_areas = np.stack([bb_areas[order[:-1]], bb_areas[order[1:]]], axis=1)
    first_is_smaller = pair_areas[:, 0] < pair_areas[:, 1]
    idx_first = np.invert(first_is_smaller).astype(np.int)
    idx_second = first_is_smaller.astype(np.int)
    size_difference = pair_areas[range(len(pair_areas)), idx_first] / pair_areas[range(len(pair_areas)), idx_second]

    if category is None:
        features = np.concatenate([np.expand_dims(x_distance, axis=1),
                                   np.expand_dims(y_distance, axis=1),
                                   np.expand_dims(size_difference, axis=1)],
                                  axis=1)
    else:
        # Create a 1-hot vector of the category
        one_hot = np.zeros([len(x_distance), 5], dtype=np.float32)
        one_hot[:, category] = 1.0

        features = np.concatenate([np.expand_dims(x_distance, axis=1),
                                   np.expand_dims(y_distance, axis=1),
                                   np.expand_dims(size_difference, axis=1),
                                   one_hot],
                                  axis=1)

    return features


def calculate_group_pair_labels(order, bb_entities):
    # Figure out the labels based on whether these bbs share at least one grouped entity
    labels = np.zeros([len(order)-1, 1])
    for i in range(len(order)-1):
        if len(set(bb_entities[order[i]]).intersection(set(bb_entities[order[i + 1]]))) > 0:
            labels[i, 0] = 1.0

    return labels
