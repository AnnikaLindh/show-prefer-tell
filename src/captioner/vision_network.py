# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Show, Prefer and Tell: Incorporating User Preferences into Image Captioning (Lindh, Ross & Kelleher, 2023)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/show-prefer-tell
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from os import path as os_path
import json
import numpy as np
import cv2
import torch
import torchvision
from torchvision.ops import batched_nms
from model_wrapper import CaffeModel


class VisionNetwork:
    def __init__(self, model_load_paths, bb_grouping, region_ranker, region_ordering, vg_to_flickr30k_path, device):
        self.bb_grouping = bb_grouping
        self.region_ranker = region_ranker
        self.region_ordering = region_ordering
        self.device = device

        # Load the data to translate the vision network's categories into Flickr30k categories
        with open(vg_to_flickr30k_path) as infile:
            vg_to_flickr30k_data = json.load(infile)
            self.vg_idx_to_flickr30k_idx = vg_to_flickr30k_data["vg_idx_to_flickr30k_idx"]

        print("Loading vision model...")
        self.model_wrapper = CaffeModel(image_dir=None, device=self.device)
        self.model_wrapper.load_model(prototxt_path=model_load_paths[0],
                                      weights_path=model_load_paths[1],
                                      caffe_proto_path=model_load_paths[2])

        print("...vision model loaded!")

    def _extract_vision_network_data(self, image, override_boxes=None):
        # Prepare input for the vision network
        img_tensor, im_info, _ = self.model_wrapper.prepare_input(image=image, bboxes=None)
        img_tensor = img_tensor.to(device=self.device)
        im_info = im_info.to(device=self.device)

        # Extract bbs from the vision network
        if override_boxes is None:
            _ = self.model_wrapper.forward(data=img_tensor, im_info=im_info)
            rois = self.model_wrapper.model.blobs["rois"].data_
        else:
            rois = np.concatenate([np.zeros([len(override_boxes), 1]), override_boxes], axis=1)
            rois = torch.tensor(rois, dtype=torch.float32, device=self.device) * im_info[2]

        # Prepare the full image bbox to add to rois
        full_image_bbox = [[0, 0, im_info[1], im_info[0]]]
        full_image_bbox = torch.tensor(full_image_bbox, dtype=torch.float32, device=self.device) * im_info[2]
        level = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
        full_image_bbox = torch.cat((level, full_image_bbox), dim=1)
        rois = torch.cat((rois, full_image_bbox), dim=0)
        num_boxes = len(rois)

        # Extract the features from the rois + full image box
        _ = self.model_wrapper.forward(data=img_tensor, override_variables={"rois": rois}, im_info=im_info)
        blobs = self.model_wrapper.model.blobs

        # Get the full image features from the last box (the full image box)
        full_image_features = blobs["pool5_flat"].data_[-1:, :].unsqueeze(0)

        # Drop the full image and the levels column from the rois boxes and scale them back to the original scale
        rois = blobs["rois"].data_[:-1, 1:] / im_info[2]

        # Get the object probabilities, skipping the full image box
        object_probs_all = blobs["cls_prob"].data_
        object_probs_all = torch.reshape(object_probs_all, shape=(num_boxes, -1,))[:-1, :]

        feature_maps = blobs["pool5_flat"].data_

        object_probs_max, object_classes = torch.max(object_probs_all, dim=1)

        # Drop the background probs from the full probs
        object_probs_all = object_probs_all[:, 1:]

        # Gather all features into an object
        vision_data = VisionData(im_info=im_info, full_image_features=full_image_features, rois=rois,
                                 feature_maps=feature_maps, object_probs_max=object_probs_max,
                                 object_probs_all=object_probs_all, object_classes=object_classes)

        # Only keep boxes with a higher probability in one of the non-background classes
        fg_idx = torch.nonzero(object_classes != 0).squeeze(-1)
        vision_data.filter(keep_idx=fg_idx)

        # Compensate for including the background class in torch.max
        vision_data.object_classes += 1

        return vision_data

    def extract_features(self, image, num_regions, class_nms=0.3, min_confidence=0.3, override_boxes=None,
                         export_params=None):
        # Extract feature maps, class probabilities, etc from the vision network
        vision_data = self._extract_vision_network_data(image, override_boxes=override_boxes)

        # Derive the flickr30k classes from the vg classes
        vision_data.derive_flickr30k_object_classes(vg_idx_to_flickr30k_idx=self.vg_idx_to_flickr30k_idx,
                                                    device=self.device)

        # Only allow boxes with a minimum confidence score
        if override_boxes is None:
            keep_idx = torch.nonzero(vision_data.object_probs_30k > min_confidence).squeeze(-1)
            if len(keep_idx) > 0:
                vision_data.filter(keep_idx=keep_idx)

        # Perform class-sensitive NMS on the remaining boxes with predicted Flickr30k classes
        vision_data.class_nms(iou_threshold=0.3, class_type="flickr30k")

        if (self.bb_grouping is not None) and (len(vision_data.rois) > 1):
            # Get the relevant features for bb grouping
            bb_grouping_features = dict()
            bb_grouping_features["boxes"] = vision_data.rois.cpu().numpy()
            # Use the original width and height since we have scaled back rois to original image coordinates
            bb_grouping_features["image_height"] = vision_data.im_info[0].cpu().numpy()
            bb_grouping_features["image_width"] = vision_data.im_info[1].cpu().numpy()
            bb_grouping_features["categories"] = vision_data.object_classes_30k.cpu().numpy()
            bb_grouping_features["feature_maps"] = vision_data.feature_maps.cpu().numpy()

            grouped_entities = self.bb_grouping(bb_grouping_features)
            if len(grouped_entities) > 0:
                # Add the grouped entities and calculate their values
                vision_data.add_grouped_entities(grouped_entities)

                # Remove individual bbs that were grouped
                remove_ids = set([bb_id for entity_ids in grouped_entities for bb_id in entity_ids])
                keep_idx = list(set(range(len(vision_data.rois))).difference(remove_ids))
                vision_data.filter(keep_idx)

                # Perform class-sensitive NMS on the boxes (and grouped boxes) with predicted Flickr30k classes
                grouped_offset = len(vision_data.object_classes_30k) - len(grouped_entities)
                keep_idx = vision_data.class_nms(iou_threshold=class_nms, class_type="flickr30k")
                grouped_entities = [grouped_entities[i] for i in range(len(grouped_entities)) if grouped_offset+i in keep_idx]
        else:
            grouped_entities = list()

        # Get the relevant features for region selection
        region_ranking_features = dict()
        region_ranking_features["boxes"] = vision_data.rois.cpu().numpy()
        # Use the original width and height since we have scaled back rois to original image coordinates
        region_ranking_features["image_height"] = vision_data.im_info[0].cpu().numpy()
        region_ranking_features["image_width"] = vision_data.im_info[1].cpu().numpy()
        region_ranking_features["object_probs"] = vision_data.object_probs_30k.cpu().numpy()
        region_ranking_features["categories"] = vision_data.object_classes_30k.cpu().numpy()
        region_ranking_features["feature_maps"] = vision_data.feature_maps.cpu().numpy()
        region_ranking_features["grouped_entities"] = grouped_entities

        # Based on the box features so far, select which ones to keep with the Region Selector
        ranked_indices = self.region_ranker.inference(region_ranking_features)
        if len(ranked_indices) == 0:
            return None

        if num_regions > 0:
            selected_boxes_idx = ranked_indices[:num_regions]
        else:
            selected_boxes_idx = ranked_indices

        # Order the selected boxes
        ordered_idx = self.region_ordering(region_ranking_features, selected_boxes_idx)
        selected_boxes_idx = selected_boxes_idx[ordered_idx]

        # Export the image with selected boxes numbered by their order
        if export_params is not None:
            rois = (vision_data.rois + 0.5).int().cpu().numpy()

            display_boxes_with_numbers(output_dir=export_params["output_dir"],
                                       image_id=export_params["image_id"], image=image,
                                       boxes=rois,
                                       selected_boxes=selected_boxes_idx)

        # Find out which grouped entities we are keeping
        min_grouped_idx = len(vision_data.rois) - len(grouped_entities)
        grouped_entities = [grouped_entities[idx-min_grouped_idx] for idx in selected_boxes_idx
                            if idx >= min_grouped_idx]

        # Filter to keep only the selected boxes (in the correct order)
        vision_data.filter(selected_boxes_idx)

        cic_features = self._prepare_cic_features(vision_data, grouped_entities)

        return cic_features

    # Extract vision features from the detections that are closest to ground-truth bounding boxes for ablation tests
    def extract_closest_box_features(self, image, gt_boxes, gt_entities, gt_entity_order,
                                     class_nms=0.3, min_confidence=0.3, export_params=None):
        # Extract feature maps, class probabilities, etc from the vision network
        vision_data = self._extract_vision_network_data(image, override_boxes=None)

        # Derive the flickr30k classes from the vg classes
        vision_data.derive_flickr30k_object_classes(vg_idx_to_flickr30k_idx=self.vg_idx_to_flickr30k_idx,
                                                    device=self.device)

        # Only allow boxes with a minimum confidence score
        keep_idx = torch.nonzero(vision_data.object_probs_30k > min_confidence).squeeze(-1)
        if len(keep_idx) > 0:
            vision_data.filter(keep_idx=keep_idx)

        # Perform class-sensitive NMS on the remaining boxes with predicted Flickr30k classes
        vision_data.class_nms(iou_threshold=class_nms, class_type="flickr30k")

        # Find the best greedy matches mapping between RPN rois and gt entities (including merged group boxes)
        entity_to_rpn, bb_to_rpn = self._gt_to_rpn(rpn_detections=vision_data.rois,
                                                   gt_boxes=gt_boxes,
                                                   gt_entities=gt_entities,
                                                   gt_entity_order=gt_entity_order,
                                                   min_iou=class_nms)

        if (len(entity_to_rpn) == 0) and (len(bb_to_rpn) < 2):
            return None

        # Turn the selected rois boxes into relative coords
        im_info = vision_data.im_info
        im_size = torch.tensor([im_info[1], im_info[0], im_info[1], im_info[0]], device=self.device)
        scaled_rois = vision_data.rois / im_size

        selected_boxes = list()
        categories = list()
        # Start with the empty region and add each matched entity's region features
        region_features = [torch.zeros([1, 2053], device=self.device, dtype=torch.float32)]
        for entity_id in gt_entity_order:
            current_features = None

            if entity_id in entity_to_rpn:
                # We have a match to the solo or grouped entity
                num_boxes_per_region = torch.tensor([[1.0]], device=self.device, dtype=torch.float32)
                rpn_idx = entity_to_rpn[entity_id]
                current_features = torch.cat([vision_data.feature_maps[rpn_idx:rpn_idx+1, :],
                                              scaled_rois[rpn_idx:rpn_idx+1, :],
                                              num_boxes_per_region], dim=1)

                if export_params is not None:
                    # Add display data
                    categories.append(vision_data.object_classes_30k[rpn_idx])
                    selected_boxes.append(vision_data.rois[rpn_idx:rpn_idx+1, :])
            elif len(bb_to_rpn) >= 2:
                # Check if this is a grouped entity, and if so if we have at least 2 of the partial boxes
                bb_ids = gt_entities[entity_id]
                if len(bb_ids) > 1:
                    rpn_idx = [bb_to_rpn[current_id] for current_id in bb_ids if current_id in bb_to_rpn]
                    if len(rpn_idx) > 1:
                        num_boxes_per_region = torch.tensor([[len(rpn_idx)]], device=self.device, dtype=torch.float32)

                        # Get the min of the mins and the max of the maxes for all bbs in this region
                        merged_box = torch.tensor([[torch.min(scaled_rois[rpn_idx, 0]),
                                                    torch.min(scaled_rois[rpn_idx, 1]),
                                                    torch.max(scaled_rois[rpn_idx, 2]),
                                                    torch.max(scaled_rois[rpn_idx, 3])]], device=self.device)

                        # Take the average of the individual feature maps
                        merged_features = torch.mean(vision_data.feature_maps[rpn_idx, :], dim=0, keepdim=True)
                        current_features = torch.cat([merged_features,
                                                      merged_box,
                                                      num_boxes_per_region], dim=1)

                        if export_params is not None:
                            # Add display data
                            flickr30k_category = torch.argmax(torch.sum(vision_data.object_probs_30k[rpn_idx], dim=0))
                            categories.append(flickr30k_category)
                            merged_box_unscaled = torch.tensor([[torch.min(vision_data.rois[rpn_idx, 0]),
                                                                 torch.min(vision_data.rois[rpn_idx, 1]),
                                                                 torch.max(vision_data.rois[rpn_idx, 2]),
                                                                 torch.max(vision_data.rois[rpn_idx, 3])]],
                                                               device=self.device)
                            selected_boxes.append(merged_box_unscaled)

            if current_features is not None:
                region_features.append(current_features)

        num_regions = len(region_features) - 1  # -1 so we don't count the empty region
        if num_regions == 0:
            return None

        # Piece together the region features
        region_features = torch.cat(region_features, dim=0)
        region_start_indices = (1,)
        region_end_indices = (num_regions + 1,)  # +1 because the empty region is the first region in the matrix

        # Export the image with all boxes displayed
        if export_params is not None:
            rois = (torch.cat(selected_boxes, dim=0) + 0.5).int().cpu().numpy()

            selected_idx = np.asarray(list(range(num_regions)))

            display_boxes_with_numbers(output_dir=export_params["output_dir"],
                                       image_id=export_params["image_id"], image=image,
                                       boxes=rois,
                                       selected_boxes=selected_idx)

        return {'full_image_features': vision_data.full_image_features, 'region_features': region_features,
                'region_start_indices': region_start_indices, 'region_end_indices': region_end_indices}

    def _gt_to_rpn(self, rpn_detections, gt_boxes, gt_entities, gt_entity_order, min_iou):
        example_gt_boxes = list()
        example_gt_partial_boxes = list()
        partial_bb_ids = list()
        for entity_id in gt_entity_order:
            bb_ids = gt_entities[entity_id]

            if len(bb_ids) == 1:
                entity_box = torch.tensor(gt_boxes[bb_ids[0]], device=self.device, dtype=torch.float32)
            else:
                # Combine the group entity boxes into one large box that encompasses all of them
                entity_box = [np.min(gt_boxes[bb_ids, 0]), np.max(gt_boxes[bb_ids, 1]),
                              np.min(gt_boxes[bb_ids, 2]), np.max(gt_boxes[bb_ids, 3])]
                entity_box = torch.tensor(entity_box, device=self.device, dtype=torch.float32)

                # Also add the individual boxes to the partial boxes to simulate automatic grouping
                partial_bb_ids.extend(bb_ids)
                example_gt_partial_boxes.append(torch.tensor(gt_boxes[bb_ids], device=self.device, dtype=torch.float32))

            example_gt_boxes.append(entity_box)

        # Try to match each RPN and GT, going by the highest IoU matches overall
        example_gt_boxes = torch.stack(example_gt_boxes)
        entity_to_rpn = self._match_boxes_by_iou(detections=rpn_detections,
                                                 gt_boxes=example_gt_boxes,
                                                 gt_idx_mapping=gt_entity_order,
                                                 min_iou=min_iou)

        # Try to match each RPN to partial boxes (bbs that need to be grouped), going by the highest IoU matches overall
        if len(example_gt_partial_boxes) == 0:
            bb_to_rpn = list()
        else:
            example_gt_partial_boxes = torch.cat(example_gt_partial_boxes, dim=0)
            bb_to_rpn = self._match_boxes_by_iou(detections=rpn_detections,
                                                 gt_boxes=example_gt_partial_boxes,
                                                 gt_idx_mapping=partial_bb_ids,
                                                 min_iou=min_iou)

        return entity_to_rpn, bb_to_rpn

    # Try to match each detection and gt box, going by the highest IoU matches overall
    @staticmethod
    def _match_boxes_by_iou(detections, gt_boxes, gt_idx_mapping, min_iou):
        matches = dict()

        iou = calculate_iou(detections, gt_boxes)  # [num_detections x num_gt]
        iou_values, flattened_idx = torch.sort(iou.flatten(), descending=True)

        matched_rpn = set()
        matched_gts = set()
        row_length = iou.size(1)
        for i_match in range(len(iou_values)):
            if iou_values[i_match] < min_iou:
                break

            rpn_idx = int(flattened_idx[i_match] / row_length)
            if rpn_idx in matched_rpn:
                continue

            gt_idx = flattened_idx[i_match] % row_length
            if gt_idx in matched_gts:
                continue

            # Mark this RPN and GT as already matched
            matched_rpn.add(rpn_idx)
            matched_gts.add(gt_idx)

            matches[gt_idx_mapping[gt_idx]] = rpn_idx

        return matches

    def _prepare_cic_features(self, vision_data, grouped_entities):
        # Get the visual features from the blob for the regions of interest and prepare them according to the
        # old preprocessing code into the format expected by the CIC model
        region_features = vision_data.feature_maps
        num_regions = len(region_features)
        full_image_features = vision_data.full_image_features

        if num_regions == 0:
            region_start_indices = (0,)
            region_end_indices = (1,)

            # Add the empty region
            region_features = torch.zeros([1, 2053], device=self.device)
        else:
            region_start_indices = (1,)
            region_end_indices = (num_regions + 1,)  # +1 because we'll be starting with the empty region

            # Turn the selected rois boxes into relative coords
            im_info = vision_data.im_info
            im_size = torch.tensor([im_info[1], im_info[0], im_info[1], im_info[0]], device=self.device)
            scaled_rois = vision_data.rois / im_size

            # Add the relative coordinates and the num boxes per region to the features
            num_grouped = len(grouped_entities)
            if num_grouped == 0:
                num_boxes_per_region = torch.ones([num_regions, 1], device=self.device, dtype=torch.float32)
            elif num_grouped == num_regions:
                entity_bb_nums = [[len(bb_ids)] for bb_ids in grouped_entities]
                num_boxes_per_region = torch.tensor(entity_bb_nums, device=self.device, dtype=torch.float32)
            else:
                single_bb_nums = torch.ones([num_regions-num_grouped, 1], device=self.device, dtype=torch.float32)
                entity_bb_nums = [[len(bb_ids)] for bb_ids in grouped_entities]
                entity_bb_nums = torch.tensor(entity_bb_nums, device=self.device, dtype=torch.float32)
                num_boxes_per_region = torch.cat([single_bb_nums, entity_bb_nums])

            region_features = torch.cat((region_features, scaled_rois, num_boxes_per_region,), dim=1)

            # Add the empty region at the start of the region features
            region_features = torch.cat((torch.zeros([1, 2053], device=self.device), region_features,), dim=0)

        return {'full_image_features': full_image_features, 'region_features': region_features,
                'region_start_indices': region_start_indices, 'region_end_indices': region_end_indices}

    def order_cic_regions(self, cic_features, num_regions, class_nms=0.3, export_params=None):
        # Drop the empty region and the spatial features
        region_features = cic_features["region_features"][1:, :-5]
        override_variables = {"pool5": region_features.to(device=self.device)}
        start_from = "pool5_flat"
        _ = self.model_wrapper.forward(data=None, override_variables=override_variables, start_from=start_from)
        object_probs_all = self.model_wrapper.model.blobs["cls_prob"].data_

        # Sort out the dimensions and drop the background class
        object_probs_all = object_probs_all.reshape(len(object_probs_all), -1)[:, 1:]

        # Compute the Flickr30k classes
        rois = torch.tensor(cic_features["boxes"], device=self.device)
        vision_data = VisionData(None, None, rois, None, None, object_probs_all, None)
        vision_data.derive_flickr30k_object_classes(vg_idx_to_flickr30k_idx=self.vg_idx_to_flickr30k_idx,
                                                    device=self.device)

        if (num_regions > 0) and (self.region_ranker is not None):
            # Perform class-sensitive NMS on the boxes (and grouped boxes) with predicted Flickr30k classes
            keep_idx = vision_data.class_nms(iou_threshold=class_nms, class_type="flickr30k")
            keep_idx = keep_idx.cpu().numpy()

            # Get the relevant features for region selection
            region_ranking_features = dict()
            region_ranking_features["boxes"] = cic_features["boxes"][keep_idx, :]
            region_ranking_features["image_height"] = cic_features["image_height"]
            region_ranking_features["image_width"] = cic_features["image_width"]
            region_ranking_features["object_probs"] = vision_data.object_probs_30k.cpu().numpy()
            region_ranking_features["categories"] = vision_data.object_classes_30k.cpu().numpy()
            region_ranking_features["feature_maps"] = region_features[keep_idx, :]

            # Account for the initial empty region
            keep_idx = [0] + [idx+1 for idx in keep_idx]
            cic_features['region_features'] = cic_features['region_features'][keep_idx, :]

            region_ranking_features["num_boxes_per_region"] = cic_features["region_features"][1:, -1].cpu().numpy()

            # Based on the box features so far, select which ones to keep with the Region Selector
            ranked_indices = self.region_ranker.inference(region_ranking_features)
            selected_boxes_idx = ranked_indices[:num_regions]

            # Export the image with all boxes displayed
            if export_params is not None:
                image = cv2.imread(os_path.join(export_params["image_dir"], export_params["image_id"] + '.jpg'))

                display_boxes_with_classes(output_dir=export_params["output_dir"],
                                           image_id=export_params["image_id"], image=image,
                                           boxes=region_ranking_features["boxes"],
                                           flickr30k_categories=region_ranking_features["categories"],
                                           selected_idx=selected_boxes_idx)
        else:
            # Use all regions if num_regions <= 0
            selected_boxes_idx = list(range(len(cic_features["boxes"])))

            # These are still needed for ordering
            region_ranking_features = dict()
            region_ranking_features["boxes"] = cic_features["boxes"]
            region_ranking_features["image_height"] = cic_features["image_height"]
            region_ranking_features["image_width"] = cic_features["image_width"]
            region_ranking_features["object_probs"] = vision_data.object_probs_30k.cpu().numpy()
            region_ranking_features["categories"] = vision_data.object_classes_30k.cpu().numpy()
            region_ranking_features["feature_maps"] = region_features

        # Order the selected boxes
        ordered_idx = self.region_ordering(region_ranking_features, selected_boxes_idx)
        # Account for the initial empty region
        ordered_idx = [0] + [selected_boxes_idx[idx]+1 for idx in ordered_idx]

        # Copy the cic features but order the region features appropriately
        prepared_features = dict()
        prepared_features['full_image_features'] = cic_features['full_image_features']
        prepared_features['region_features'] = cic_features['region_features'][ordered_idx, :]
        prepared_features['region_start_indices'] = (1,)
        prepared_features['region_end_indices'] = (len(ordered_idx),)

        return prepared_features


class VisionData:
    def __init__(self, im_info, full_image_features, rois, feature_maps,  object_probs_max, object_probs_all,
                 object_classes):
        self.im_info = im_info
        self.full_image_features = full_image_features
        self.rois = rois
        self.feature_maps = feature_maps
        self.object_probs_max = object_probs_max
        self.object_probs_all = object_probs_all
        self.object_classes = object_classes

        self.object_probs_30k = None
        self.object_classes_30k = None

    def filter(self, keep_idx):
        if self.rois is not None:
            self.rois = self.rois[keep_idx]
        if self.feature_maps is not None:
            self.feature_maps = self.feature_maps[keep_idx]
        if self.object_probs_max is not None:
            self.object_probs_max = self.object_probs_max[keep_idx]
        if self.object_probs_all is not None:
            self.object_probs_all = self.object_probs_all[keep_idx]
        if self.object_classes is not None:
            self.object_classes = self.object_classes[keep_idx]

        if self.object_probs_30k is not None:
            self.object_probs_30k = self.object_probs_30k[keep_idx]
            self.object_classes_30k = self.object_classes_30k[keep_idx]

    def derive_flickr30k_object_classes(self, vg_idx_to_flickr30k_idx, device):
        sorted_probs, sorted_classes = torch.sort(self.object_probs_all, dim=-1, descending=True)

        # Find unique_classes in the top 3 of sorted_classes, then sum up the probs for each of those to choose a class
        category_probs = list(map(lambda i_box:
                                  category_with_total_prob(sorted_probs[i_box, :3], sorted_classes[i_box, :3],
                                                           vg_idx_to_flickr30k_idx),
                                  range(len(sorted_probs))))
        self.object_classes_30k = torch.tensor([category_probs[i_pair][0]
                                                for i_pair in range(len(category_probs))],
                                               dtype=torch.int64).to(device)
        self.object_probs_30k = torch.tensor([category_probs[i_pair][1]
                                              for i_pair in range(len(category_probs))]).to(device)

    def class_nms(self, iou_threshold, class_type, evaluate_only=False):
        if class_type == "flickr30k":
            object_probs = self.object_probs_30k
            object_classes = self.object_classes_30k
        elif class_type == "vg":
            object_probs = self.object_probs_max
            object_classes = self.object_classes
        else:
            assert False, "Unknown class_type: " + class_type

        keep_idx = batched_nms(boxes=self.rois, scores=object_probs, idxs=object_classes, iou_threshold=iou_threshold)

        if not evaluate_only:
            self.filter(keep_idx=keep_idx)

        return keep_idx

    def add_grouped_entities(self, grouped_entities):
        for bb_ids in grouped_entities:
            # Get the min of the mins and the max of the maxes for all bbs in this region
            entity_box = torch.tensor([[torch.min(self.rois[bb_ids, 0]),
                                        torch.min(self.rois[bb_ids, 1]),
                                        torch.max(self.rois[bb_ids, 2]),
                                        torch.max(self.rois[bb_ids, 3])]], device=self.rois.device)
            self.rois = torch.cat([self.rois, entity_box], dim=0)

            # Take the average of the individual feature maps
            entity_features = torch.mean(self.feature_maps[bb_ids, :], dim=0, keepdim=True)
            self.feature_maps = torch.cat([self.feature_maps, entity_features], dim=0)

            # Take the max of prob maxes
            prob_max, _ = torch.max(self.object_probs_max[bb_ids], dim=0, keepdim=True)
            self.object_probs_max = torch.cat([self.object_probs_max, prob_max], dim=0)

            # Take the maximum value for each of the individual probabilities
            probs_all, _ = torch.max(self.object_probs_all[bb_ids, :], dim=0, keepdim=True)
            self.object_probs_all = torch.cat([self.object_probs_all, probs_all], dim=0)

            # Calculate the object class
            _, object_class = torch.max(probs_all, dim=1)
            self.object_classes = torch.cat([self.object_classes, object_class], dim=0)

            # Only bbs of the same Flickr30k class are allowed to be grouped, so they should all be the same
            self.object_classes_30k = torch.cat([self.object_classes_30k, self.object_classes_30k[bb_ids[0]:bb_ids[0]+1]], dim=0)

            # Take the max prob of the bbs for the Flickr30k prob
            prob_flickr30k, _ = torch.max(self.object_probs_30k[bb_ids], dim=0, keepdim=True)
            self.object_probs_30k = torch.cat([self.object_probs_30k, prob_flickr30k], dim=0)


def category_with_total_prob(sorted_probs, sorted_classes, vg_idx_to_flickr30k_idx):
    top_classes = [vg_idx_to_flickr30k_idx[i_class] for i_class in sorted_classes]
    unique_classes = list(set(top_classes))

    if len(unique_classes) == 1:
        total_prob = sorted_probs.sum()
        top_category = top_classes[0]

        return top_category, total_prob

    top_classes = np.asarray(top_classes)
    category_probs = [sorted_probs[top_classes == current_class].sum() for current_class in unique_classes]
    top_unique_idx = np.asarray(category_probs).argmax()

    top_category = unique_classes[top_unique_idx]
    total_prob = category_probs[top_unique_idx]

    return top_category, total_prob


def draw_rectangles(img, color, thickness, boxes):
    for i_box in range(len(boxes)):
        # Draw the bb rectangle
        left_top = (int(boxes[i_box][0]), int(boxes[i_box][1]))
        right_bottom = (int(boxes[i_box][2]), int(boxes[i_box][3]))
        img = cv2.rectangle(img, left_top, right_bottom, color, thickness)

    return img


def draw_texts(img, color, thickness, boxes, scores, classes, starting_number=0):
    font = cv2.FONT_HERSHEY_PLAIN

    for i_box in range(len(boxes)):
        left_top = (int(boxes[i_box][0]), int(boxes[i_box][1]))
        right_bottom = (int(boxes[i_box][2]), int(boxes[i_box][3]))

        # Draw text information above or below the bb
        if starting_number == 0:
            pos = (left_top[0], left_top[1] + 8)
        else:
            pos = (left_top[0], right_bottom[1] - 4)

        # Build a single line of info
        info = ''
        # Ordering of the boxes
        info += str(i_box + starting_number) + ':'
        # Flickr30kEntities object class ID
        if classes is not None:
            info += " {0:d}".format(int(classes[i_box]))
        # Confidence score in brackets
        info += " ({0:.2f})".format(scores[i_box])

        img = cv2.putText(img, info, org=pos, fontFace=font, fontScale=1, color=color, thickness=thickness)

    return img


def display_boxes_with_classes(output_dir, image_id, image, boxes, flickr30k_categories, selected_idx):
    flickr30k_category_names = ['people', 'animals', 'instruments', 'vehicles', 'other', 'clothing', 'bodyparts']

    # Clone the image variable so we don't corrupt the input
    img = image.copy()

    # Figure out which of the full range of boxes were rejected
    rejected_idx = np.ones([len(boxes)], dtype=np.int32)
    rejected_idx[selected_idx] = 0
    rejected_idx = rejected_idx.nonzero()

    if len(rejected_idx) > 0:
        # Draw thin red boxes for all rejected detections
        thickness = 1
        color = (0, 0, 255)
        img = draw_rectangles(img, color, thickness, boxes[rejected_idx, :])
        flickr30k_classes = [flickr30k_category_names[cat] for cat in flickr30k_categories[rejected_idx]]
        img = draw_texts_classes(img, color, thickness, boxes[rejected_idx, :], flickr30k_classes)

    # Save the rejected boxes on their own image to avoid clutter on the other one
    cv2.imwrite(os_path.join(output_dir, image_id + '_rejected.jpg'), img)

    # Start with a fresh copy of the image for the accepted boxes
    img = image.copy()

    if len(selected_idx) > 0:
        # Draw dark yellow boxes and text info for selected detections
        thickness = 2
        color = (0, 190, 255)
        img = draw_rectangles(img, color, thickness, boxes[selected_idx, :])
        thickness = 1
        flickr30k_classes = [flickr30k_category_names[cat] for cat in flickr30k_categories[selected_idx]]
        img = draw_texts_classes(img, color, thickness, boxes[selected_idx, :], flickr30k_classes)

    # Save the resulting image
    cv2.imwrite(os_path.join(output_dir, image_id + '_accepted.jpg'), img)


def display_boxes_with_numbers(output_dir, image_id, image, boxes, selected_boxes):
    # Clone the image variable so we don't corrupt the input
    img = image.copy()

    # Display ordered boxes
    if len(selected_boxes) > 0:
        # Draw red boxes and text info
        thickness = 2
        color = (0, 0, 250)
        img = draw_rectangles(img, color, thickness, boxes[selected_boxes, :])
        numbers = [str(i_idx + 1) for i_idx in range(len(selected_boxes))]
        img = draw_texts_classes(img, color, thickness, boxes[selected_boxes, :], numbers)

    # Save the resulting image
    cv2.imwrite(os_path.join(output_dir, image_id + '_numbered.jpg'), img)


def draw_texts_classes(img, color, thickness, boxes, flickr30k_classes):
    font = cv2.FONT_HERSHEY_PLAIN

    for i_box in range(len(boxes)):
        # Draw text information inside the bb
        left_top = (int(boxes[i_box][0]), int(boxes[i_box][1]))
        pos = (left_top[0] + 8, left_top[1] + 60)

        img = cv2.putText(img, flickr30k_classes[i_box], org=pos, fontFace=font, fontScale=4, color=color,
                          thickness=thickness)

    return img


def probs_to_categories(box_probs, box_idx, top_k, idx_to_vg_category, vg_idx_to_flickr30k_idx,
                        flickr30k_category_names):
    sorted_probs, sorted_classes = torch.sort(box_probs[box_idx, :], dim=-1, descending=True)
    vg_categories = [[idx_to_vg_category[sorted_classes[i_box][i_order]] for i_order in range(top_k)]
                     for i_box in range(len(sorted_classes))]
    flickr30k_categories = [[flickr30k_category_names[vg_idx_to_flickr30k_idx[sorted_classes[i_box][i_order]]]
                            for i_order in range(top_k)] for i_box in range(len(sorted_classes))]

    return sorted_probs[:, :top_k].cpu().numpy(), vg_categories, flickr30k_categories


# Returns an array with the indices of the potential overlappers with an IoU of 0.3 or more
def find_overlapping_idx(base_boxes, overlappers):
    # Calculate the intersection over the overlapper area (percentage) where each col belongs to one possible overlapper
    overlapped_percent = calculate_iou(base_boxes, overlappers)

    # Find the unique column indices where the IoU is at or above 0.3
    overlapped_idx = np.unique(np.where(overlapped_percent.cpu().numpy() > 0.3)[1])

    return overlapped_idx


# Returns a matrix [num_dets, num_gts] with the IoU for all combinations of detections to gt_boxes
def calculate_iou(detections, gt_boxes):
    left_min, left_max = _min_max_coordinates(detections, gt_boxes, col_number=0)
    top_min, top_max = _min_max_coordinates(detections, gt_boxes, col_number=1)
    right_min, right_max = _min_max_coordinates(detections, gt_boxes, col_number=2)
    bottom_min, bottom_max = _min_max_coordinates(detections, gt_boxes, col_number=3)

    # Calculate the intersection area between each detection (rows) and gt_box (cols)
    intersection_width = right_min - left_max
    intersection_width[intersection_width < 0] = 0
    intersection_height = bottom_min - top_max
    intersection_height[intersection_height < 0] = 0
    intersection = intersection_width * intersection_height

    # Calculate the union area between each detection (rows) and gt_box (cols)
    detection_areas = box_areas(detections).view(-1, 1).expand(-1, len(gt_boxes))
    gt_areas = box_areas(gt_boxes).view(1, -1).expand(len(detections), -1)
    combined_areas = detection_areas + gt_areas
    unions = combined_areas - intersection  # avoid counting the overlapped area twice

    # Calculate the intersection over union area (percentage) between each detection (rows) and gt_box (cols)
    iou = intersection / unions

    return iou


# Get the min and max values for 1 coordinate (left, top, right OR bottom) for each detection-to-gt_box combo
def _min_max_coordinates(base_boxes, overlappers, col_number):
    num_base = len(base_boxes)
    num_overlappers = len(overlappers)

    # Build a matrix with 1 row per detection
    base_coords = base_boxes[:, col_number].unsqueeze(1)
    base_coords = base_coords.expand([num_base, num_overlappers])

    # Build a matrix with 1 col per GT box
    overlapper_coords = overlappers[:, col_number].unsqueeze(0)
    overlapper_coords = overlapper_coords.expand([num_base, num_overlappers])

    # Get the smallest and largest of this coordinate for all combinations
    coord_min = torch.min(base_coords, overlapper_coords)
    coord_max = torch.max(base_coords, overlapper_coords)

    return coord_min, coord_max


# (right - left) * (bottom - top) = area  aka  (x_max - x_min) * (y_max - y_min) = area
def box_areas(boxes):
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return areas
