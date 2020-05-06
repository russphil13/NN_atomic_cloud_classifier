from csv import reader
from custom.model_development import getstats_fromstream, ImageDataset
from custom.model_development import load_datasets, make_datasets
from custom.model_development import RepeatedKFolds, SearchCV
#import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC

set_types = ['training', 'test']
num_ex = [260, 64]
datasets_dict = dict(zip(set_types, num_ex))

class_names = ['empty', 'cloud']
seed = 52

params_file = Path('/home/bob/development/experiment_data/2018/'
                   'cloudparams_s20.csv')

dir_training_images = ('/home/bob/development/atomic_cloud_training_data/'
                       'training_data')

path_images_left = Path(dir_training_images, 'left_clouds/seed_13')
path_images_right = Path(dir_training_images, 'right_clouds/seed_31')

dir_model_main = ('/home/bob/development/neural_nets/atomic_cloud/'
                  'dnn_binary_classify/models')

path_model_id = Path(dir_model_main, f'num_ex_{num_ex[0]}_seed_{seed}')

#make_datasets(class_names, datasets_dict, path_images_left, path_model_id, seed)

num_pixels, scale_max, _ = getstats_fromstream(path_model_id,
                                               path_images_left)

batch_size = 16
epochs = 100
val_count = 40

model_params = {'input_dim': num_pixels,
                'dense_layers': 2,
                'nodes_per_layer': [[5, 10, 20]],
                'hidden_act': ['relu']}
#                'dropout_layers': [[0.15, 0.25]]}

modelDNN = Sequential()

search = SearchCV(modelDNN,
                  model_params,
                  cv)

metrics = {'auc': AUC()}

earlystop_params = {'monitor': 'val_auc',
                    'min_delta': 0.001,
                    'patience': 10}
early_stop = EarlyStopping(**earlystop_params) 

fit_params = {'path_data': path_images_left,
             'path_model_id': path_model_id,
             'metrics': metrics,
             'batch_size': batch_size,
             'epochs': epochs,
             'verbose': 0,
             'callbacks': [early_stop],
             'rescale': 1/scale_max,
             'seed': 137}

search.fit(**fit_params)
search.display_results(metrics)

# Run this to check performance on the test set.
#model_L_params = {'colsample_bytree': 0.35,
#                  'gamma': 0.1,
#                  'learning_rate': 0.001,
#                  'max_depth': 9,
#                  'min_child_weight': 5,
#                  'n_estimators': 500,
#                  'random_state': 10,
#                  'reg_alpha': 0,
#                  'reg_lambda': 100,
#                  'subsample': 0.4,
#                  'random_state': 10,
#                  'verbosity': 0}
#model_L = XGBClassifier(**model_L_params)
#model_L.fit(X_train_scl_L, y_train_L, eval_set=eval_set,
#            eval_metric='auc', early_stopping_rounds=20,
#            verbose=False)
#print("Detailed classification report:")
#print()
#y_true, y_pred = y_test_L, model_search.predict(X_test_scl_L)
#print(classification_report(y_true, y_pred))
#print()


