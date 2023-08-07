# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Show, Prefer and Tell: Incorporating User Preferences into Image Captioning (Lindh, Ross & Kelleher, 2023)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/show-prefer-tell
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
import json


def find_shared_ids(model_names, num_regions_min, num_regions_max, results_dir):
    num_regions_range = list(range(num_regions_min, num_regions_max+1))
    shared_imgids = {num_regions: num_regions_range for num_regions in num_regions_range}
    shared_exids = {num_regions: num_regions_range for num_regions in num_regions_range}

    for num_regions in num_regions_range:
        image_ids = None
        example_ids = None

        for model_name in model_names:
            captions_path = os.path.join(results_dir, f"CAPTIONS_{model_name}_{num_regions}.json")
            if os.path.exists(captions_path):
                with open(captions_path, "rt") as f:
                    current_ids = list(json.load(f)["generated_captions"].keys())

                if len(current_ids) == 0:
                    # If there are no captions for this number of regions, then there cannot be any shared ids for it
                    del shared_imgids[num_regions]
                    del shared_exids[num_regions]
                    break

                if '_' in current_ids[0]:
                    if example_ids is None:
                        example_ids = set(current_ids)
                    else:
                        # Only keep the example ids shared with previous example ids
                        example_ids = example_ids.intersection(current_ids)

                    if image_ids is None:
                        image_ids = {ex_id[:-2] for ex_id in example_ids}
                    else:
                        # Also remove any image ids that are not part of any unique example ids
                        image_ids = image_ids.intersection({ex_id[:-2] for ex_id in example_ids})
                else:
                    if image_ids is None:
                        image_ids = set(current_ids)
                    else:
                        # Take care of removing not-shared example_ids after the full loop
                        image_ids = image_ids.intersection(current_ids)
            else:
                # If there are no captions for this number of regions, then there cannot be any shared ids for it
                del shared_imgids[num_regions]
                del shared_exids[num_regions]
                break

        # Check if this number of regions has been deleted due to some model lacking examples for it
        if num_regions not in shared_imgids:
            continue

        if len(image_ids) == 0:
            del shared_imgids[num_regions]
            del shared_exids[num_regions]
            continue

        shared_imgids[num_regions] = list(image_ids)
        if example_ids is not None:
            # example_ids may contain examples that have no corresponding image id in image_ids
            example_ids = [ex_id for ex_id in example_ids if ex_id[:-2] in image_ids]
            shared_exids[num_regions] = example_ids
        else:
            shared_exids[num_regions] = list()  # to avoid issues when dumping to json format

    all_model_names = '_'.join(model_names)
    with open(os.path.join(results_dir, f"SHARED_IDS_{all_model_names}.json"), "wt") as f:
        json.dump({"image_ids": shared_imgids, "example_ids": shared_exids}, f)


if __name__ == "__main__":
    models = ["full_group_sh", "full_nogroup_sh"]
    find_shared_ids(model_names=models, num_regions_min=1, num_regions_max=7, results_dir="../results/")

    models = ["gt", "cic", "matched", "full_nogroup_sh"]
    find_shared_ids(model_names=models, num_regions_min=1, num_regions_max=7, results_dir="../results/")
