# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper:
# Show, Prefer and Tell: Incorporating User Preferences into Image Captioning (Lindh, Ross & Kelleher, 2023)
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/show-prefer-tell
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from __future__ import division

import sys
import os
from builtins import staticmethod

sys.path.append(os.getcwd())

from evaluation.speaksee import Bleu, Rouge, Meteor, Cider, Spice
import numpy as np
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')


class EvaluateCaptions:
    def __init__(self):
        self.gts = None
        self.scorers = None

    def setup(self, metrics, gt_path, example_ids):
        # Load all ground truth captions for text metrics evaluation
        with open(gt_path) as gt_file:
            self.gts = json.load(gt_file)

        # Load scorers
        self.scorers = dict()
        for metric in metrics:
            if metric == 'BLEU-1':
                self.scorers[metric] = Bleu(n=1)
            elif metric == 'BLEU-2':
                self.scorers[metric] = Bleu(n=2)
            elif metric == 'BLEU-3':
                self.scorers[metric] = Bleu(n=3)
            elif metric == 'BLEU-4':
                self.scorers[metric] = Bleu(n=4)
            elif metric == 'ROUGE-L':
                self.scorers[metric] = Rouge()
            elif metric == 'METEOR':
                self.scorers[metric] = Meteor()
            elif metric == 'CIDEr':
                # Provide the full set of gt captions to ensure correct document frequency counts
                self.scorers[metric] = Cider(gts=self.gts['all'])
            elif metric.startswith('SPICE'):
                self.scorers[metric] = Spice()
            else:
                print("WARNING: Unknown metric:", metric)

        # Filter by provided example_ids - must be done AFTER initializing CIDEr with the full gt_all set
        if example_ids is not None:
            for gt_type in self.gts:
                self.gts[gt_type] = self._filter_examples(self.gts[gt_type], example_ids)
                if len(self.gts[gt_type]) == 0:
                    assert gt_type != 'all', "No gt_captions available for the example_ids:" + str(example_ids)
                    del self.gts[gt_type]

    @staticmethod
    def _filter_examples(captions, example_ids):
        if example_ids is None:
            return captions
        else:
            return {id: captions[id] for id in captions if id in example_ids}

    @staticmethod
    def prepare_predictions(example_ids, predictions):
        return {k: [v] for k, v in zip(example_ids, predictions)}

    def standard_metrics(self, candidate_captions, gt_captions):
        metric_scores = dict()
        metric_details = dict()

        for metric in self.scorers:
            if len(candidate_captions) > 0:
                score, details = self._compute_score(candidate_captions, gt_captions, metric)
                metric_scores[metric] = score
                metric_details[metric] = details
            else:
                metric_scores[metric] = np.nan
                metric_details[metric] = dict()

        return metric_scores, metric_details

    def _compute_score(self, candidate_captions, labels, metric):
        score, details = self.scorers[metric].compute_score(gts=labels, res=candidate_captions)
        if isinstance(score, list):
            # This happens for the BLEU-n scores which return scores from BLEU-1 to BLEU-n
            score = score[-1]
            details = details[-1]
        elif metric == "SPICE":
            details = {example_id: details[example_id]['All']['f'] for example_id in details}
        elif metric.startswith("SPICE-"):
            spice_category = metric[6:]
            details = {example_id: details[example_id][spice_category]['f'] for example_id in details}
            score = np.nanmean(list(details.values()))

        # Convert to typical way of displaying metrics (multiply by 100)
        score *= 100.0
        details = {example_id: details[example_id]*100.0 for example_id in details}

        return score, details

    def _drop_annotations(self, gts, candidates):
        empty_ids = list()
        for image_id in candidates:
            # Create a new list rather than using .remove() to avoid changing the original list
            gts[image_id] = [caption for caption in gts[image_id] if caption != candidates[image_id][0]]
            if len(gts[image_id]) == 0:
                empty_ids.append(image_id)

        for image_id in empty_ids:
            del gts[image_id]
            del candidates[image_id]

        return gts, candidates

    def evaluate_candidate_captions(self, candidate_captions, num_regions=0, evaluate_gt=False):
        # Compare candidate captions to gt
        if num_regions > 0:
            try:
                gt_captions = self.gts[str(num_regions)]
            except KeyError:
                print("No gt captions for num_regions={num_regions}")
                return None, None, None
        else:
            gt_captions = self.gts['all']

        filtered_captions = self._filter_examples(candidate_captions, gt_captions.keys())
        filtered_gt = self._filter_examples(gt_captions, candidate_captions.keys())
        if evaluate_gt:
            filtered_gt, filtered_captions = self._drop_annotations(filtered_gt, filtered_captions)

        # Skip evaluation if there are no gt captions for this choice of num_regions
        if len(filtered_gt) == 0:
            return None, None, None

        results_scores, results_details = self.standard_metrics(filtered_captions, filtered_gt)
        num_captions = len(filtered_captions)
        print(results_scores)
        print("Number of evaluated captions:", num_captions)

        return results_scores, results_details, num_captions
