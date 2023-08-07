# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Show, Prefer and Tell: Incorporating User Preferences into Image Captioning (Lindh, Ross & Kelleher, 2023)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/show-prefer-tell
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from __future__ import division

from xml.etree import ElementTree
from os import path as os_path
import re
import json
from textblob import TextBlob, Word
from textblob.wordnet import Synset
from nltk.corpus.reader.wordnet import WordNetError
from preprocessor import Preprocessor, _print_missing
import matplotlib


# Prevent errors due to lack of display on the server nodes
matplotlib.use('Agg')
from matplotlib import pyplot as plt


_REGPATTERN_REGION_IDS = re.compile('\/EN#(\d+)\/(\S+)')
_REGPATTERN_REPLACE_ENTITIES = re.compile('\[\/en#\d+\/\S+\s(.+?)]')
_REGPATTERN_ALLOWED_CHARACTERS = re.compile('([a-z0-9- ]+|__EOC__)')
_REGPATTERN_EXTRA_WHITESPACE = re.compile(' +')
_REGPATTERN_IMAGE_ID = re.compile('^0*(\d+).')

_SYN_TO_CATEGORY = {Synset('person.n.01'): 'people',
                    Synset('animal.n.01'): 'animals',
                    Synset('musical_instrument.n.01'): 'instruments',
                    Synset('clothing.n.01'): 'clothing',
                    Synset('external_body_part.n.01'): 'bodyparts',
                    Synset('vehicle.n.01'): 'vehicles'}
_FLICKR30K_CATEGORIES = ['people', 'animals', 'instruments', 'vehicles', 'other', 'clothing', 'bodyparts']


def _export_hist(data, filepath, title, bins=100):
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.savefig(
        fname=filepath,
        dpi='figure',
        bbox_inches='tight'
    )
    plt.close()


def flickr30k_region_stats(splits_path, raw_dir, outfile):
    """
    Generates histograms and prints stats about how many regions per example in the Flickr30k data
    """
    num_regions = list()

    with open(splits_path, 'rt') as splits_file:
        image_ids = json.load(splits_file)['splits']['train']

    for image_id in image_ids:
        # Get all the region id lists for this image
        try:
            with open(os_path.join(raw_dir, image_id + '_raw.json'), 'rt') as rawfile:
                annotations = json.load(rawfile)['annotations']
        except IOError:
            continue

        num_regions.extend([len(ann['entity_ids']) for ann in annotations])

    _export_hist(data=num_regions, filepath=outfile + "_all.png", title="Number of regions per caption", bins=20)

    print("\nSTATS ALL")
    _print_hist_stats(num_regions)


def _print_hist_stats(numbers):
    stats = dict()
    for num in numbers:
        try:
            stats[num] += 1
        except KeyError:
            stats[num] = 1

    unique_numbers = sorted(list(stats.keys()))
    print([(num, stats[num]) for num in unique_numbers])


# Builds a dict of image_id -> [captions] for all captions, and separately for captions of each number of entities
def create_gt_dicts(image_splits_path, raw_dir, out_dir):
    with open(image_splits_path, 'rt') as splits_file:
        image_splits = json.load(splits_file)['splits']

    for split in image_splits:
        gts = {'all': dict()}

        for image_id in image_splits[split]:
            # Gather all the captions for this image into a list
            try:
                with open(os_path.join(raw_dir, image_id + '_raw.json'), 'rt') as rawfile:
                    annotations = json.load(rawfile)['annotations']

                gts['all'][image_id] = list()
                for ann in annotations:
                    gts['all'][image_id].append(ann['caption'])

                    # Add each caption into the dict with corresponding number of captions
                    num_regions = len(ann['entity_ids'])
                    if num_regions not in gts:
                        gts[num_regions] = dict()
                    try:
                        gts[num_regions][image_id].append(ann['caption'])
                    except KeyError:
                        gts[num_regions][image_id] = [ann['caption']]

            except FileNotFoundError:
                continue

        print("Number of images for split " + split, len(gts['all']))

        # Store this split's ground truth captions
        with open(os_path.join(out_dir, 'gt_captions_' + split + '.json'), 'wt') as f_outfile:
            json.dump(gts, f_outfile)


def format_gt_as_candidates(gt_path, output_dir):
    with open(gt_path, 'rt') as f:
        gt_data = json.load(f)

    for caption_type in gt_data:
        captions = gt_data[caption_type]
        candidates = {f"{image_id}_{i_ann}": [captions[image_id][i_ann]]
                      for image_id in captions for i_ann in range(len(captions[image_id]))}

        if caption_type == "all":
            caption_type = 0
        output_path = os_path.join(output_dir, f"CAPTIONS_gt_{caption_type}.json")
        with open(output_path, 'wt') as f:
            json.dump({'generated_captions': candidates}, f)


def create_unique_multi_splits(image_splits_path, raw_dir, output_dir, all_num_regions):
    with open(image_splits_path, 'rt') as f:
        splits = json.load(f)['splits']

    for num_regions in all_num_regions:
        # Gather only the examples with a unique sequence or unique unsorted list of entities for each image
        unique_unsorted_examples = dict()
        unique_sequence_examples = dict()

        for split in splits:
            unique_unsorted_examples[split] = list()
            unique_sequence_examples[split] = list()

            for image_id in splits[split]:
                # Keep track of entity combination uniqueness for this particular image
                image_unique_unsorted = list()
                image_unique_sequences = list()

                try:
                    with open(os_path.join(raw_dir, image_id + '_raw.json'), 'rt') as f:
                        annotations = json.load(f)['annotations']
                except FileNotFoundError:
                    continue

                for i_ann in range(len(annotations)):
                    current_ids = annotations[i_ann]['entity_ids']

                    if num_regions > 0:
                        # Only add examples with enough regions
                        if len(current_ids) < num_regions:
                            continue

                        # Cut down to the requested number of regions
                        current_ids = current_ids[:num_regions]
                    if current_ids not in image_unique_sequences:
                        image_unique_sequences.append(current_ids)
                        unique_sequence_examples[split].append(annotations[i_ann]['example_id'])

                    # Sort them here so that any permutation of ordering generates a match
                    sorted_ids = sorted(current_ids)
                    if sorted_ids not in image_unique_unsorted:
                        image_unique_unsorted.append(sorted_ids)
                        unique_unsorted_examples[split].append(annotations[i_ann]['example_id'])

        with open(os_path.join(output_dir, f"splits_unique_unsorted_{num_regions}.json"), 'wt') as f_outfile:
            json.dump({'splits': unique_unsorted_examples}, f_outfile)

        with open(os_path.join(output_dir, f"splits_unique_sequences_{num_regions}.json"), 'wt') as f_outfile:
            json.dump({'splits': unique_sequence_examples}, f_outfile)


def map_vg_categories_to_entity_categories(vg_classes_path, output_dir):
    """
    Map each word in the object vocab to one of the entity categories in Flickr30k Entities based on their WordNet tree
    people      - Synset('person.n.01')
    animals     - Synset('animal.n.01')
    instruments - Synset('musical_instrument.n.01')
    clothing    - Synset('clothing.n.01')
    bodyparts   - Synset('external_body_part.n.01')
    vehicles    - Synset('vehicle.n.01')
    other       - (anything that doesn't fit into the other categories)
    """

    with open(vg_classes_path, 'rt') as data_file:
        object_classes = data_file.readlines()

    idx_to_flickr30k_idx = list()
    idx_to_full_category = list()
    idx_to_plural = list()
    vg_category_indices = dict()
    flickr30k_category_to_idx = dict()
    category_idx = 0
    for category in _FLICKR30K_CATEGORIES:
        vg_category_indices[category] = list()
        flickr30k_category_to_idx[category] = category_idx
        category_idx += 1

    idx = 0
    # Map each predicted object type idx to one of the Flick30k Entities categories + inflection
    for class_name in object_classes:
        class_name = class_name.strip()
        category, inflection = _find_entity_category(class_name)
        idx_to_flickr30k_idx.append(flickr30k_category_to_idx[category])
        idx_to_full_category.append(category + ',' + inflection)
        print(class_name, '--->', category, inflection)

        # Create a shortcut map to tell which categories are plural
        idx_to_plural.append(inflection == 'plural')

        # Keep track of which indices are associated with each of the categories
        vg_category_indices[category].append(idx)
        idx += 1

    # Store main mapping information
    with open(os_path.join(output_dir, 'flickr30k_category_info.json'), 'wt') as output_file:
        json.dump({"vg_idx_to_flickr30k_idx": idx_to_flickr30k_idx,
                   "flickr30k_category_to_idx": flickr30k_category_to_idx,
                   "flickr30k_idx_to_category": _FLICKR30K_CATEGORIES,
                   "idx_to_full_category": idx_to_full_category, "vg_idx_to_plural": idx_to_plural,
                   "vg_category_indices": vg_category_indices}, output_file)


# Select the first matching category, default to 'other'
def _find_entity_category(word):
    inflection = 'singular'

    # Replace spaces with underscore
    word = word.replace(' ', '_')

    synsets = set()
    primary_synsets = set()
    # Sometimes there are two versions included of a word
    for w in word.split(','):
        word_singular = Word(w).singularize()
        try:
            primary_synsets.add(Synset(word_singular + '.n.01'))
        except WordNetError:
            pass  # this happens if the synset did not exist

        synsets.update(word_singular.get_synsets(pos='n'))
        if word_singular != w:
            inflection = 'plural'

    # First try the primary synsets to avoid finding a less relevant category on a lower branch
    while len(primary_synsets) > 0:
        next_synsets = set()
        for syn in primary_synsets:
            if syn in _SYN_TO_CATEGORY:
                return _SYN_TO_CATEGORY[syn], inflection

            next_synsets.update(syn.hypernyms())

        primary_synsets = next_synsets

    # Continue with the other synsets
    while len(synsets) > 0:
        next_synsets = set()
        for syn in synsets:
            if syn in _SYN_TO_CATEGORY:
                return _SYN_TO_CATEGORY[syn], inflection

            next_synsets.update(syn.hypernyms())

        synsets = next_synsets

    # If no synset matched our categories, categorize this as 'other'
    return 'other', inflection


class EntityRawPreprocessor(Preprocessor):
    def __init__(self, datadir, output_dir):
        self.datadir = datadir
        self.output_dir = output_dir
        self.bb_dir = os_path.join(self.datadir, 'Annotations')
        self.caption_dir = os_path.join(self.datadir, 'Sentences')

        assert os_path.isdir(self.output_dir), "output_dir not found: " + self.output_dir
        assert os_path.isdir(self.bb_dir), "datadir is not a directory with a subfolder called Annotations: " + self.datadir
        assert os_path.isdir(self.caption_dir), "datadir is not a directory with a subfolder called Sentences: " + self.datadir

        self.num_img_entities = list()
        self.num_img_bbs = list()

    def _preprocess_single(self, image_id):
        # Prepare output json structure
        annotations = []
        all_entities = {}  # entity_id: [bb_ids]
        all_bbs = list()  # [{'x_min': x_min, 'y_min': y_min, 'x_max': x_max, 'y_max': y_max, 'entities': [entity_ids]}]

        with open(os_path.join(self.bb_dir, image_id + '.xml'), 'rb') as region_file:
            example_data = ElementTree.parse(region_file).getroot()

            # Find all bounding box objects info from the xml file
            for bb_data in example_data.findall('object'):
                # Find all bounding boxes for this object (if any)
                bounding_boxes = bb_data.findall('bndbox')

                # If there is at least one bb, store each of them and assign them to their respective entities
                if len(bounding_boxes) > 0:
                    entity_names = [entity_name.text for entity_name in bb_data.findall('name')]

                    # In case some examples store multiple bounding boxes (normally it's one per object)
                    current_bb_ids = list()
                    for bb in bounding_boxes:
                        # The next added bb will be at the location of the current length of the all_bbs list
                        current_bb_ids.append(len(all_bbs))

                        # Use the original image x and y coordinates (needed for the bottom-up ROIs)
                        current_bb = {'x_min': int(bb[0].text),
                                      'y_min': int(bb[1].text),
                                      'x_max': int(bb[2].text),
                                      'y_max': int(bb[3].text),
                                      'entities': entity_names}

                        all_bbs.append(current_bb)

                    # Add this bounding box to all its entities
                    for entity_name in entity_names:
                        # If this entity already exists, extend the current bb id list with the new bb ids
                        if entity_name in all_entities:
                            all_entities[entity_name].extend(current_bb_ids)
                        else:
                            # If this is a new entity, store the bb ids we used in its dict place
                            all_entities[entity_name] = current_bb_ids.copy()

        with open(os_path.join(self.caption_dir, image_id + '.txt')) as caption_file:
            current_captions = caption_file.readlines()

        # Clean captions and mark entity locations, and store the entity order
        i_ann = -1
        for caption in current_captions:
            # This needs to be updated even if this annotation is not added since it represents the original number
            i_ann += 1

            # Keep track of the order that the entities are mentioned in this caption
            current_entity_ids = list()

            # Keep track of which entities should be skipped due to lacking bbs
            skip_entities = list()
            i_chunk = 0

            # Find all entity markers in this caption
            re_matches = re.findall(_REGPATTERN_REGION_IDS, caption)
            for (entity_id, entity_type) in re_matches:
                if entity_id not in all_entities:
                    # Skip entity for this entity's text chunk to merge it with the next one
                    skip_entities.append(i_chunk)
                else:
                    # Keep this entity
                    current_entity_ids.append(entity_id)

                i_chunk += 1

            # Only include annotations that have at least 1 entity with at least 1 bb
            if len(current_entity_ids) == 0:
                print("Skipping example without bounding boxes:", image_id + '_' + str(i_ann))
                continue

            # Lowercase the caption
            caption = caption.lower()

            # Replace entity markers with the entity text and the nextentity marker
            caption = re.sub(_REGPATTERN_REPLACE_ENTITIES, r'\1 nextentity', caption)

            # Keep only allowed characters
            caption = re.findall(_REGPATTERN_ALLOWED_CHARACTERS, caption)
            caption = ''.join(caption)

            # Remove extra (double) whitespaces, and any at the start and end
            caption = re.sub(_REGPATTERN_EXTRA_WHITESPACE, ' ', caption).strip()

            # Build the list of indices for the nextentity markers while removing them from the text
            next_entity_indices = list()
            cleaned_caption = list()
            i_cleaned_words = 0  # The first token will be BOC which is at index=0
            i_markers = 0
            for current_word in TextBlob(caption).words:
                if current_word == "nextentity":
                    # Skip markers for chunks that have no bb data
                    if i_markers not in skip_entities:
                        # Mark the last real word's index as where to request to a new entity attention map
                        next_entity_indices.append(i_cleaned_words)

                    # Increment the nextentity marker index
                    i_markers += 1
                else:
                    # Add normal words to the cleaned caption and increment the word index
                    cleaned_caption.append(current_word)
                    i_cleaned_words += 1

            # Add Beginning Of Caption and End Of Caption tokens
            cleaned_caption = ['BOC'] + cleaned_caption + ['EOC']

            # Add full caption, tokenized caption, next entity indices and entities to the annotation structure
            annotations.append({'caption': ' '.join(cleaned_caption[1:-1]), 'tokens': cleaned_caption,
                                'next_entity_indices': next_entity_indices, 'entity_ids': current_entity_ids,
                                'example_id': image_id + '_' + str(i_ann)})  # for matching .npz cic files

        num_annotations = len(annotations)
        if num_annotations == 0:
            print("Skipping image due to no entities in any annotations for image_id:", image_id)
            return

        # Only keep the entities in all_entities that appear in at least one annotation
        all_valid_entities = {entity_id: all_entities[entity_id]
                              for ann in annotations for entity_id in ann['entity_ids']}

        # Only keep the entity_ids for a bb that appear in all_valid_entitites
        # And only keep bbs that appear in at least one valid entity, while updating their ids based on their new index
        all_valid_bbs = list()
        id_offset = 0  # decrement for every skipped bb
        all_offsets = list()  # offsets for each old bb_id (so their index in this list needs to equal the old index)
        for i_bb in range(len(all_bbs)):
            entity_ids = [entity_id for entity_id in all_bbs[i_bb]['entities'] if entity_id in all_valid_entities]
            if len(entity_ids) > 0:
                all_bbs[i_bb]['entities'] = entity_ids
                all_valid_bbs.append(all_bbs[i_bb])
            else:
                id_offset -= 1

            # Keep track of how to adjust the bb_ids in all_valid_entities to agree with the new list order
            all_offsets.append(id_offset)  # we do this even for skipped ids to keep the index the same

        if id_offset < 0:
            # Update the bb_ids in all_valid_entities to account for where in the list a bb_id was dropped
            all_valid_entities = {entity_id: [old_bb_id + all_offsets[old_bb_id]
                                              for old_bb_id in all_entities[entity_id]]
                                  for entity_id in all_valid_entities}

        # Gather stats about number of entities and bbs
        self.num_img_bbs.append(len(all_bbs))
        self.num_img_entities.append(len(all_valid_entities))

        with open(os_path.join(self.output_dir, image_id + '_raw.json'), 'wt') as outfile:
            json.dump({'all_entities': all_valid_entities, 'all_entity_ids': list(all_valid_entities.keys()),
                       'all_bbs': all_valid_bbs, 'annotations': annotations},
                      outfile)
