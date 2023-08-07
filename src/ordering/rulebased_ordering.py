# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Show, Prefer and Tell: Incorporating User Preferences into Image Captioning (Lindh, Ross & Kelleher, 2023)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/show-prefer-tell
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np
from preprocessing.preprocess_cv_features import box_relative_centers, box_areas


FLICKR30K_CATEGORIES = ['people', 'animals', 'instruments', 'vehicles', 'other', 'clothing', 'bodyparts']


# 1. Choose a starting box based on:
#    a) Lowest category (except for clothing and bodyparts)
#    b) Largest area
# 2. REPEAT: Choose next relating to previous box that was not clothing, bodyparts or overlapping, based on:
#    a) Overlapping clothing if previous box was <= 1 (person or animal)
#    b) Overlapping bodyparts if previous box was <= 1 (person or animal)
#    c) Overlapping of any other type
#    d) Closest of remaining
def order_rulebased(region_ranking_features, selected_boxes_idx):
    boxes = region_ranking_features["boxes"][selected_boxes_idx]
    categories = region_ranking_features["categories"][selected_boxes_idx]

    if len(boxes) == 0:
        return np.asarray([], dtype=np.int)
    elif len(boxes) == 1:
        return np.asarray([0])

    box_sizes = box_areas(boxes)
    cat_clothing = FLICKR30K_CATEGORIES.index("clothing")
    cat_bodyparts = FLICKR30K_CATEGORIES.index("bodyparts")

    ordered_idx = list()
    remaining_idx = list(range(len(boxes)))

    # 1. CHOOSE STARTING BOX
    # a) Find lowest category (aside from clothing and bodyparts)
    cat_idx = list()
    for cat in range(6):
        # Skip clothing and bodyparts at this stage
        if (cat == cat_clothing) or (cat == cat_bodyparts):
            continue

        cat_idx = (categories == cat).nonzero()[0]
        if len(cat_idx) > 0:
            break

    # If no results were found from our preferred categories, then choose from any boxes
    if len(cat_idx) == 0:
        cat_idx = remaining_idx

    if len(cat_idx) == 1:
        # If there was only 1 choice, choose that box
        previous_box_idx = cat_idx[0]
    else:
        # b) If there are multiple choices, choose the one with the largest area
        largest_idx = np.argmax(box_sizes[cat_idx])
        previous_box_idx = cat_idx[largest_idx]

    # Actually add the chosen box (and remove from list of remaining)
    ordered_idx.append(previous_box_idx)
    remaining_idx.remove(previous_box_idx)

    # 2. CHOOSE NEXT BOX BASED ON PREVIOUS (ignore clothes/bodyparts and overlapping as previous boxes)
    while len(remaining_idx) > 1:
        # Find any overlapping boxes
        overlapping_idx, x_overlaps, y_overlaps = _find_overlapping(primary_box=boxes[(previous_box_idx,), :],
                                                                    other_boxes=boxes[remaining_idx, :])

        if len(overlapping_idx) > 0:
            # Convert into np array of box indices from the full list of boxes
            overlapping_idx = np.asarray(remaining_idx)[overlapping_idx]

            # If the previous box was a person or animal, we start with any overlapping clothes and bodyparts
            if categories[previous_box_idx] <= 1:
                # a) Find the ones with category == clothing
                priority_overlapping = (categories[overlapping_idx] == cat_clothing)
                priority_overlapping = overlapping_idx[priority_overlapping]

                # Add them in order of largest to smallest
                if len(priority_overlapping) > 0:
                    size_order = list(np.argsort(box_sizes[priority_overlapping]))
                    size_order.reverse()
                    ordered_idx.extend(priority_overlapping[size_order])

                    # Keep track of which indices are remaining after we added the ones in priority_overlapping
                    overlapping_idx = np.asarray([idx for idx in overlapping_idx if idx not in priority_overlapping], dtype=np.int)
                    remaining_idx = [idx for idx in remaining_idx if idx not in priority_overlapping]

                # b) Find the ones with category == bodyparts
                if len(overlapping_idx) > 0:
                    priority_overlapping = (categories[overlapping_idx] == cat_bodyparts)
                    priority_overlapping = overlapping_idx[priority_overlapping]

                    # Add them in order of largest to smallest
                    if len(priority_overlapping) > 0:
                        size_order = list(np.argsort(box_sizes[priority_overlapping]))
                        size_order.reverse()
                        ordered_idx.extend(priority_overlapping[size_order])

                        # Keep track of which indices are remaining after we added the ones in priority_overlapping
                        overlapping_idx = np.asarray([idx for idx in overlapping_idx if idx not in priority_overlapping], dtype=np.int)
                        remaining_idx = [idx for idx in remaining_idx if idx not in priority_overlapping]

            # c) Add any remaining overlapping_idx in order of largest to smallest
            if len(overlapping_idx) > 0:
                size_order = list(np.argsort(box_sizes[overlapping_idx]))
                size_order.reverse()
                ordered_idx.extend(overlapping_idx[size_order])
                remaining_idx = [idx for idx in remaining_idx if idx not in overlapping_idx]

            # Quit if we added the last boxes
            if len(remaining_idx) == 0:
                break

        # d) Find the closest of the remaining boxes relative to our previous box (before any overlapping)
        # Find the closest based on manhattan center-to-center distance (which does not penalize having uneven
        # distances between the axes)
        distances = _manhattan_center_distance(main_box=boxes[previous_box_idx], other_boxes=boxes[remaining_idx, :])
        closest_idx = np.argmin(distances)

        # Convert to the absolute index
        previous_box_idx = remaining_idx[closest_idx]

        # Add this box and remove from the remaining indices
        ordered_idx.append(previous_box_idx)
        remaining_idx.remove(previous_box_idx)

    # Make sure to add the final remaining idx
    if len(remaining_idx) == 1:
        ordered_idx.append(remaining_idx[0])

    return np.asarray(ordered_idx)


def _find_overlapping(primary_box, other_boxes):
    x_overlap_idx, x_overlaps = _find_axis_overlap(primary_box, other_boxes, axis='x')
    y_overlap_ids, y_overlaps = _find_axis_overlap(primary_box, other_boxes, axis='y')

    overlapping_idx = x_overlap_idx & y_overlap_ids

    return overlapping_idx, x_overlaps, y_overlaps


def _find_axis_overlap(primary_box, other_boxes, axis):
    if axis == 'x':
        axis = 0
    elif axis == 'y':
        axis = 1

    primary_half_length = (primary_box[:, axis+2] - primary_box[:, axis]) / 2.0
    other_half_lengths = (other_boxes[:, axis+2] - other_boxes[:, axis]) / 2.0

    primary_center = primary_half_length + primary_box[:, axis]
    other_centers = other_half_lengths + other_boxes[:, axis]
    center_distances = np.abs(other_centers - primary_center)

    overlaps = (other_half_lengths + primary_half_length) - center_distances

    return overlaps > 0, overlaps


def _manhattan_center_distance(main_box, other_boxes):
    x_centers = (other_boxes[:, 2] - other_boxes[:, 0]) / 2 + other_boxes[:, 0]
    y_centers = (other_boxes[:, 3] - other_boxes[:, 1]) / 2 + other_boxes[:, 1]

    main_x = (main_box[2] - main_box[0]) / 2 + main_box[0]
    main_y = (main_box[3] - main_box[1]) / 2 + main_box[1]

    x_distances = np.abs(main_x - x_centers)
    y_distances = np.abs(main_y - y_centers)

    total_distances = x_distances + y_distances

    return total_distances


class IdentityOrder:
    def __call__(self, region_ranking_features, selected_boxes_idx, device):
        return np.asarray(list(range(len(selected_boxes_idx))))
