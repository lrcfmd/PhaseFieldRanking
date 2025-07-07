import pytest
import numpy as np
from ranking_phase_fields.train_and_validate import Trainer

def test_trainer_train(monkeypatch):
    # Dummy training data
    x_train = [['Al', 'Be', 'C']] * 100

#   # Patch sym2num and permute to avoid actual heavy logic
#    monkeypatch.setattr('ranking_phase_fields.train_and_validate.sym2num', lambda x, y: np.array([[0.1]*10]*len(x)))
#    monkeypatch.setattr('ranking_phase_fields.train_and_validate.permute', lambda x: x)

    trainer = Trainer(model_name='VAE', phase_fields='ternary', features='mat2vec', natom=3)
    x_, threshold, nnet, results = trainer.run_training(x_train)

    assert len(results) == 1
    assert isinstance(results, dict)
