# Ranking unexplored phase fields.
# 13.08.2020
# Andrij Vasylenko 
# and.vasylenko@gmail.com
#
# This programme 
# 1) reads the list of compositions in ICSD,
# where oxidation states are specified
# 2) and extracts a training set of phase fields:
#    binaries
#    ternaries
#    quaternaries
#    quinaries
#    etc. with the specified ox. states
# 3) AE, VAE and other models can be trained on the training set
# for detection of the outliers in the training and testing sets
# and ranking the unexplored phase fields of choice in terms of the degree
# of 'outlying' (degree = reconstruction error). That is hypothesised to 
# correlate with similarity with the 'known' phase fields, and hence,
# with the likelihood of forming a stable composition within a phase field.

import sys
from pathlib import Path
from typing import Optional
import itertools as it
from ranking_phase_fields.parse_icsd import parse_input, parse_icsd
from ranking_phase_fields.generate_study import generate_study, permute
from ranking_phase_fields.features import sym2num, numatoms
from ranking_phase_fields.models import rank, get_latent_vectors, latent_file
from ranking_phase_fields.train_and_validate import train_model, validate
from ranking_phase_fields.logger import get_logger

logger = get_logger(__name__)

class PhaseFieldRanker:
    def __init__(self, input_file: str = "config.yaml") -> None:
        self.input_file = input_file
        self.params = None
        self.training = []
        self.testing = []

    def load_params(self) -> None:
        logger.info(f"Loading input parameters from '{self.input_file}'")
        self.params = parse_input(self.input_file)

    def prepare_training_data(self) -> None:
        if self.params is None:
            raise RuntimeError("Input parameters not loaded")
        logger.info("Parsing ICSD training data")
        self.training = parse_icsd(
            self.params.phase_fields,
            self.params.anions_train,
            self.params.nanions_train,
            self.params.cations_train,
            self.params.icsd_file
        )

    def generate_testing_data(self) -> None:
        if self.params is None:
            raise RuntimeError("Input parameters not loaded")
        logger.info(f"Generating testing data for unexplored {self.params['phase_fields']} phase fields")
        self.testing = generate_study(
            self.params.phase_fields,
            self.params.elements_test,
            self.training
        )

    def train(self):
        if self.params is None:
            raise RuntimeError("Input parameters not loaded")
        logger.info("Starting model training")
        return train_model(
            self.params.phase_fields,
            self.params.features,
            self.training,
            self.params.method,
            numatoms(self.params.phase_fields),
            self.params.average_runs
        )

    def cross_validate(self, threshold, nnet):
        if self.params is None:
            raise RuntimeError("Input parameters not loaded")
        if self.params.get('cross-validate', False) == 'True':
            logger.info("Performing cross-validation")
            validate(
                self.params.phase_fields,
                self.params.features,
                self.training,
                self.params.method,
                numatoms(self.params.phase_fields),
                threshold,
                nnet
            )

    def encode_latent_space(self, clft, trained_results, dataset, suffix: str):
        latent_vectors = get_latent_vectors(dataset, self.params.features, clft)
        latent_file(trained_results, latent_vectors, f"{self.params.phase_fields}_{self.params.method}_{suffix}.pkl")

    def rank_testing_data(self, clft, trained):
        if self.params is None:
            raise RuntimeError("Input parameters not loaded")

        logger.info("=" * 45)
        logger.info(f"Representing each element with {self.params.features} features.")
        logger.info("=" * 45)

        testing_permuted = permute(self.testing)
        testing_numeric = sym2num(testing_permuted, self.params.features)

        testing_results = rank(
            clft,
            self.params.phase_fields,
            self.params.features,
            trained,
            testing_numeric,
            self.params.method,
            numatoms(self.params.phase_fields),
            self.params.average_runs]
        )
        return testing_results

    def run_study(self) -> None:
        logger.info("=" * 55)
        logger.info("RANKING OF THE PHASE FIELDS BY LIKELIHOOD WITH ICSD DATA")
        logger.info("Similar phase fields to those found in ICSD are hypothesized to yield stable compositions.")
        logger.info("The similarity is measured via encoding and decoding vectorized phase fields with VAE.")
        logger.info("Author: Andrij Vasylenko | Date: 2020-08-13")
        logger.info("=" * 55)

        self.load_params()
        self.prepare_training_data()
        self.generate_testing_data()
        self.trainer = Trainer(
            model_name=self.params.method,
            phase_fields=self.params.phase_fields,
            features=self.params.features,
            natom=numatoms(self.params.phase_fields),
            average_runs=self.params.average_runs
        )

        x_, threshold, nnet, training_results = self.trainer.train(self.training)

        if self.params.method == 'VAE_encoder':
            latent_vecs = self.trainer.get_latent_vectors(self.training)

        if self.params.cross_validate:
            self.trainer.validate(self.training, threshold)

        testing_results = self.rank_testing_data(clft, trained)

        logger.info("Finished successfully. Exiting.")

def main(input_file: Optional[str] = None):
    if input_file is None:
        input_file = "config.yaml"
    root_dir = Path(__file__).resolve().parent.parent
    ranker = PhaseFieldRanker(root_dir / input_file)
    ranker.run_study()

if __name__ == "__main__":
    input_file_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(input_file_arg)
