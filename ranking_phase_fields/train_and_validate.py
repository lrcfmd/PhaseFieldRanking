import numpy as np
from typing import List, Tuple, Optional

from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
from ranking_phase_fields.vae_encoder import VAE_
from ranking_phase_fields.generate_study import permute
from ranking_phase_fields.features import sym2num, num2sym, vec2name
from ranking_phase_fields.score_utils import scale, mse, average_permutations, getout
from ranking_phase_fields.logger import get_logger

logger = get_logger(__name__)

class Trainer:
    def __init__(self, model_name: str, phase_fields: str, features: str, natom: int, average_runs: int = 1):
        self.model_name = model_name
        self.phase_fields = phase_fields
        self.features = features
        self.natom = natom
        self.average_runs = average_runs
        self.nnet = self._build_network()

    def _build_network(self) -> List[int]:
        """Configure hidden layers based on features and atom counts"""
        ndes = len(self.features)
        return [
            int(ndes * self.natom / 2), int(ndes * self.natom / 4), int(ndes * self.natom / 8), int(ndes * self.natom / 16),
            self.natom,
            int(ndes * self.natom / 16), int(ndes * self.natom / 8), int(ndes * self.natom / 4), int(ndes * self.natom / 2)
        ]

    def _choose_model(self):
        """Select model instance"""
        clfs = {
            'AE': AutoEncoder(hidden_neuron_list=self.nnet, contamination=0.1, epoch_num=15),
            'VAE': VAE(encoder_neuron_list=self.nnet[:5], decoder_neuron_list=self.nnet[4:], contamination=0.1, epoch_num=13),
            'VAE_encoder': VAE_(encoder_neurons=self.nnet[:5], decoder_neurons=self.nnet[4:], gamma=5, epochs=23, latent_dim=2)
        }
        return clfs[self.model_name]

    def run_training(self, x_train: List) -> Tuple[np.ndarray, float, List[int], List]:
        """Train model, compute scores, and save results"""
        x_ = permute(x_train)
        x_ = sym2num(x_, self.features)

        logger.info(f"Training {self.model_name} model on {len(x_train)} samples.")
        model = self._choose_model()
        model.fit(x_)
        scores = np.array(model.decision_scores_)
        threshold = 0.5 * (max(scores) + min(scores))

        if self.average_runs == 1:
            logger.info("Averaging training scores and writing results.")
            train_scaled = scale(scores)
            training_results = average_permutations(self.natom, x_, self.features, scores, train_scaled)
            getout(training_results, f"{self.phase_fields}_{self.model_name}_{self.features}_training_scores.csv", "Norm. scores")
        else:
            training_results = []

        return x_, threshold, self.nnet, training_results

    def rank(self, x_train: np.ndarray, x_test: np.ndarray) -> dict:
        """
        Train and rank test samples against trained ICSD data, with optional averaging over multiple runs.
        """
        if self.clf is None:
            self.clf = self._choose_model()

        if self.average_runs == 1:
            y_test_scores = self.clf.decision_function(x_test)
            y_test_scaled = scale(y_test_scores)
            logger.info(f"Writing single-run test scores to {self.phase_fields}_{self.model_name}_{self.features}_test_scores.csv")
            
            results = average_permutations(self.natom, x_test, self.features, y_test_scores, y_test_scaled)
            getout(results, f"{self.phase_fields}_{self.model_name}_{self.features}_test_scores.csv", "Norm. scores")
            return results

        else:
            y_test_scores = np.zeros(len(x_test))
            y_train_scores = np.zeros(len(x_train))

            logger.info(f"Training {self.model_name} on {len(x_train)} phase fields and averaging over {self.average_runs} runs.")

            for i in range(self.average_runs):
                logger.info(f"{self.model_name} run {i + 1}/{self.average_runs}")
                self.clf.fit(x_train)

                train_scores_run = np.asarray(self.clf.decision_scores_)
                test_scores_run = np.asarray(self.clf.decision_function(x_test))

                y_train_scores += train_scores_run
                y_test_scores += test_scores_run

                if i == 0:
                    train_stack = train_scores_run[None, :]
                    test_stack = test_scores_run[None, :]
                else:
                    train_stack = np.vstack([train_stack, train_scores_run])
                    test_stack = np.vstack([test_stack, test_scores_run])

            y_train_scores /= self.average_runs
            y_test_scores /= self.average_runs

            var_train = mse(train_stack, y_train_scores)
            var_test = mse(test_stack, y_test_scores)

            logger.info(f"Writing averaged train scores to {self.phase_fields}_{self.model_name}_{self.features}_train_scores.csv")
            results_train = average_permutations(self.natom, x_train, self.features, y_train_scores, var_train)
            getout(results_train, f"{self.phase_fields}_{self.model_name}_{self.features}_train_scores.csv", "Variance from avg score")

            logger.info(f"Writing averaged test scores to {self.phase_fields}_{self.model_name}_{self.features}_test_scores.csv")
            results_test = average_permutations(self.natom, x_test, self.features, y_test_scores, var_test)
            getout(results_test, f"{self.phase_fields}_{self.model_name}_{self.features}_test_scores.csv", "Variance from avg score")

            return results_test

    def get_latent_vectors(self, x_data: List) -> Optional[np.ndarray]:
        """Get latent representation (only for VAE_encoder)"""
        if not hasattr(self.clf, 'encoder'):
            logger.warning("Latent vectors requested but encoder not available.")
            return None
        x_ = sym2num(x_data, self.features)
        return self.clf.encoder(x_)[-1].numpy()

    def validate(self, x_data: List, threshold: float) -> None:
        """Perform 5-fold cross-validation"""
        logger.info("Starting 5-fold cross-validation.")
        l = len(x_data) // 5

        for i in range(5):
            val_set = x_data[i * l : (i + 1) * l]
            val_train = [d for idx, d in enumerate(x_data) if not (i * l <= idx < (i + 1) * l)]

            val_train_p = permute(val_train)
            val_set_p = permute(val_set)

            val_train_n = sym2num(val_train_p, self.features)
            val_set_n = sym2num(val_set_p, self.features)

            clf_v = self._choose_model()
            logger.info(dir(clf_v))
            clf_v.fit(val_train_n)

            prediction = np.asarray(clf_v.decision_function(val_set_n))
            val_error = len(prediction[prediction > threshold]) / len(val_set_n) * 100

            logger.info(f"Validation subset {i+1}: error = {val_error:.2f}%")
            with open(f'Validation_{self.phase_fields}_errors.dat', 'a') as f:
                f.write(f"Validation error of subset {i}: {val_error:.2f}%\n")

            thr_array = threshold * np.ones(len(prediction))
            results = average_permutations(self.natom, val_set_n, self.features, prediction, thr_array)
            getout(results, f'Validation_{self.phase_fields}_{self.model_name}_subset{i+1}.csv', 'threshold')

