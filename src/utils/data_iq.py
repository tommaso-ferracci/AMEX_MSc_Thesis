import numpy as np
import xgboost as xgb

# Adapted from https://github.com/seedatnabeel/Data-IQ.

class DataIQ_xgb:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        # Placeholders for probabilities
        self._correct_probas = None

    def on_epoch_end(self, bst, iteration=1):
        # Predict probabilities
        dmatrix = xgb.DMatrix(self.X_train)
        probas = bst.predict(dmatrix, iteration_range=(0, iteration))
        batch_correct_probas = np.where(self.y_train == 1, probas, 1 - probas)

        # Store probabilities
        if self._correct_probas is None:
            self._correct_probas = np.expand_dims(batch_correct_probas, axis=-1)
        else:
            self._correct_probas = np.hstack((self._correct_probas, np.expand_dims(batch_correct_probas, axis=-1)))

    @property
    def correct_probas(self) -> np.ndarray:
        return self._correct_probas

    @property
    def confidence(self) -> np.ndarray:
        return np.mean(self._correct_probas, axis=-1)

    @property
    def aleatoric(self):
        preds = self._correct_probas
        return np.mean(preds * (1 - preds), axis=-1)
