from __future__ import annotations

import logging

import numpy as np
import random
import tqdm
from datasets import Dataset

from mteb.encoder_interface import Encoder, EncoderWithQueryCorpusEncode
from mteb.MTEBResults import ScoresDict

from ..evaluation.evaluators import ClusteringEvaluator
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AbsTaskClustering(AbsTask):
    """Abstract class for Clustering tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        sentences: list of str
        labels: list of str
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _evaluate_subset(
        self, model: EncoderWithQueryCorpusEncode | Encoder, dataset: Dataset, **kwargs
    ) -> ScoresDict:
        v_measures = []
        for cluster_set in tqdm.tqdm(dataset, desc="Clustering"):
            ds = cluster_set
            if bool(self.debug_downsample):
                combined = list(zip(*[ds[split] for split in ds.keys()]))
                random.shuffle(combined)
                n = min(self.debug_downsample, min([len(ds[split]) for split in ds.keys()]))
                combined = combined[0:n]
                ds = {v:[e[i] for e in combined] for i,v in enumerate(ds.keys())}
                logger.info(f"Downsampled to {n} samples.")
            evaluator = ClusteringEvaluator(
                ds["sentences"],  # type: ignore
                ds["labels"],  # type: ignore
                **kwargs,
            )
            metrics = evaluator(model)
            v_measures.append(metrics["v_measure"])

        v_mean = np.mean(v_measures)
        v_std = np.std(v_measures)
        scores = {"v_measure": v_mean, "v_measure_std": v_std, "v_measures": v_measures}
        self._add_main_score(scores)
        return scores
