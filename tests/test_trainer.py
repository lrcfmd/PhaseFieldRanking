import pytest
import numpy as np
from ranking_phase_fields.train_and_validate import Trainer

def test_trainer_train(monkeypatch):
    # Dummy training data
    x_train = [['Al', 'Be', 'C']] * 100

    trainer = Trainer(model_name='VAE', phase_fields='ternary', features='mat2vec', natom=3)
    x_, threshold, nnet, results = trainer.run_training(x_train)

    assert len(results) == 1
    assert isinstance(results, dict)
