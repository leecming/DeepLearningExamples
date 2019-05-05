"""
Simple one layer dense against fixed representation of first token
from pre-trained BERT models
"""
import os
from typing import List
import numpy as np
import pandas as pd
import tables
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
# pylint: disable=no-name-in-module
import tensorflow as tf
from tensorflow.python.keras import layers, optimizers, losses
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF debug messages
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1  # JIT compilation
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
K.set_session(sess)  # set this TensorFlow session as the default session for Keras


class FixedFeaturesDense:
    def __init__(self):
        self.seed = 1337
        self.fixed_features_dim = 1024 * 4
        self.num_folds = 4
        self.batch_size = 128
        self.epochs = 4
        self.num_neurons = 256
        self.target_cols = ['toxic',
                            'severe_toxic',
                            'obscene',
                            'threat',
                            'insult',
                            'identity_hate']
        self.save_predict_path = 'data/fixedfeatures_dense.csv'

        self.raw_train_df = pd.read_csv('data/train.csv')
        self.raw_test_df = pd.read_csv('data/test.csv')
        self.fixed_features_path = 'data/output_L24_H1024.h5'
        print('train csv shape: {}'.format(self.raw_train_df.shape))
        print('test csv shape: {}'.format(self.raw_test_df.shape))
        # confirm all 0/1 values
        assert all(self.raw_train_df[self.target_cols].apply(lambda x: x.unique() == [0, 1]))

    def generate_train_kfolds_indices(self) -> List:
        """
        Seeded kfolds cross validation indices using just a range(len) call
        :return: (training index, validation index)-tuple list
        """
        seeded_kf = KFold(n_splits=self.num_folds, random_state=self.seed, shuffle=True)
        print('generated train kfold indices...')
        return [(train_index, val_index) for train_index, val_index in
                seeded_kf.split(range(len(self.raw_train_df)))]

    def build_dense_model(self) -> Model:
        """
        build and return model using standard optimizer and loss
        :return:
        """
        feature_input = layers.Input(shape=(self.fixed_features_dim,))
        dense_output = layers.Dense(self.num_neurons, activation='relu')(feature_input)
        dense_output = layers.Dense(6, activation='sigmoid')(dense_output)

        dense_model = Model(feature_input, dense_output)
        dense_model.compile(optimizer=optimizers.Adam(),
                            loss=losses.binary_crossentropy)

        print('generated model...')

        return dense_model

    def fit_model_on_fold(self, compiled_model: Model, curr_fold_indices):
        """
        trains compiled (but previously unfitted) model against given indices
        :param compiled_model:
        :param curr_fold_indices:
        :return:
        """
        train_indices, val_indices = curr_fold_indices

        with tables.open_file(self.fixed_features_path, mode='r') as f:
            train_sequences = f.root.data
            print('fixed features shape: {}'.format(train_sequences.shape))
            x_train = np.take(train_sequences, train_indices, axis=0)
            x_val = np.take(train_sequences, val_indices, axis=0)
            print('train X shape: {}'.format(x_train.shape))
            print('val X shape: {}'.format(x_val.shape))

        y_train = self.raw_train_df[self.target_cols].iloc[train_indices].values
        y_val = self.raw_train_df[self.target_cols].iloc[val_indices].values

        compiled_model.fit(x_train, y_train,
                           batch_size=self.batch_size,
                           epochs=self.epochs,
                           validation_data=(x_val, y_val))

        val_roc_auc_score = roc_auc_score(y_val,
                                          compiled_model.predict(x_val,
                                                                 batch_size=self.batch_size, verbose=0))
        print('ROC-AUC val score: {0:.4f}'.format(val_roc_auc_score))

        # test_predictions = compiled_model.predict(test_sequences, batch_size=self.batch_size, verbose=0)

        return val_roc_auc_score, None

    def run_end_to_end(self):
        """
        per the tin, runs text loading, preprocessing and model building and training
        dumps predictions to CSV in same folder
        :return:
        """
        kfold_indices = self.generate_train_kfolds_indices()

        fold_roc_auc_scores = []
        fold_predictions = []
        for i in range(self.num_folds):
            built_model = self.build_dense_model()
            curr_fold_results = self.fit_model_on_fold(built_model, kfold_indices[i])
            fold_roc_auc_scores.append(curr_fold_results[0])
            fold_predictions.append(curr_fold_results[1])
        print('mean val AUC: {0:.4f}'.format(np.mean(fold_roc_auc_scores)))
        # mean_predictions_df = pd.DataFrame(np.mean(fold_predictions, axis=0),
        #                                    columns=self.target_cols)
        # predicted_test = pd.concat([self.raw_test_df, mean_predictions_df], axis=1)
        # predicted_test.to_csv(self.save_predict_path)


if __name__ == '__main__':
    FixedFeaturesDense().run_end_to_end()
