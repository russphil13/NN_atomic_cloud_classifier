from csv import reader, writer
from itertools import chain, product
import numpy as np
from numpy.random import default_rng
from os import path
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import Accuracy, Precision, Recall

class BatchHistory(Callback):
    """Callback giving metrics for each training batch.
    
    The metrics are loss and accuracy. This is used primarily when the
    number of epochs is small.
    """
    
    def on_train_begin(self, logs={}):
        self.loss = []
        self.accuracy = []
    
    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))

class ImageDataset():
    """Pipeline for streaming image data."""
    
    def __init__(self,
                 path_data,
                 path_model_id,
                 set_name,
                 batch_size=32,
                 n_epochs=1,
                 rescale=None,
                 shuffle=True,
                 seed=None,
                 val_count=0):
        """Initialization.

        Generates pipelines for streaming data during training.
        
        Args
            path_data: Path object. Directory to all images for each
                class.
            path_model_id: Path object. Directory for files pertaining
                to a particular model.
            set_name: String. Name of dataset corresponding a file with
                list of images. Options are 'training' for
                'training_set.csv' and test for 'test_set.csv'.
            batch_size: Int. Batch size for stochastic gradient descent.
            rescale: Float. Multiplicative factor to apply to data.
            shuffle: Bool. Shuffle data? True/False.
            seed: Int. Seed for random shuffling.
        """

        self.path_data = path_data
        self.path_model_id = path_model_id
        self.set_name = set_name
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rescale = rescale        
        self.shuffle = shuffle
        self.seed = seed
        self.val_count = val_count
        
        self.n_examples = 0
        self.class_le = LabelEncoder()

        ds_1, ds_2 = self._prepare_sets()
        if self.set_name is 'training':
            self.ds_primary = self._train_pipeline(*ds_1)
            if val_count != 0:
                self.ds_secondary = self._val_pipeline(*ds_2)
            else:
                self.ds_secondary = None
        elif self.set_name is 'test':
            self.ds_primary, self.ds_secondary = self._test_pipeline(*ds_1)
    
    def get_datasets(self):
        """Get the training and validation datasets.
        
        Returns: Tuple of length=2 containing TF Datasets.
        """
        return self.ds_primary, self.ds_secondary
    
    def _load_images_labels(self):
        """Load images and labels from file.
        
        Returns: TF Dataset of image-label pairs.
        """
        path_dataset_file = self.path_model_id.joinpath(f'{self.set_name}_set.csv')
        
        with path_dataset_file.open(mode='r', newline='') as f:
            csv_reader = reader(f, delimiter=',')
            rows = list(csv_reader)

        if self.shuffle:
            rng = default_rng(self.seed)
            rng.shuffle(rows)
            
        self.n_examples = len(rows)

        ds_files = tf.data.Dataset.from_tensor_slices(
            [path.join(str(self.path_data), f'label_{row[1]}', row[0])
                for row in rows])
        
        ds_images = ds_files.map(self._load_preprocess_image)

        class_labels_enc = self.class_le.fit_transform(
            [row[1] for row in rows])

        ds_labels = tf.data.Dataset.from_tensor_slices(
            class_labels_enc)

        return ds_images, ds_labels

    def _load_preprocess_image(self, image_file):
        """Load image from file and perform preprocessing.
        
        Args
            image_file: String. Full path and filename of image.
            
        Returns: Tensor of a single dimension containing the pixel
            values.
        """
        image_raw = tf.io.read_file(image_file)

        image = self._preprocess_image(image_raw)

        return image

    def _prepare_sets(self):
        """Prepare training and validation datasets.
        
        Returns: Tuple of length=2 containing TF Datasets.
        """

        ds_images, ds_labels = self._load_images_labels()

        ds_images_2 = ds_images.take(self.val_count)
        ds_labels_2 = ds_labels.take(self.val_count)
        ds_images_1 = ds_images.skip(self.val_count)
        ds_labels_1 = ds_labels.skip(self.val_count)

        ds_1 = (ds_images_1, ds_labels_1)
        ds_2 = (ds_images_2, ds_labels_2)

        return ds_1, ds_2
    
    def _preprocess_image(self, image_raw):
        """Convert raw binary to float64 and scale the pixel values.
        
        Args
            image_raw: Byte array of pixel values.
            
        Returns: Tensor.
        """

        image = tf.io.decode_raw(image_raw, tf.float64)
        
        if self.rescale is not None:
            image_out = image * self.rescale
        else:
            image_out = image

        return image_out

    def _test_pipeline(self, ds_images, ds_labels):
        """Create a training dataset pipeline.
        
        Returns: TF Dataset.
        """
        
        ds_images_out = (ds_images.batch(self.batch_size)
                                  .prefetch(3))
        ds_labels_out = (ds_labels.batch(self.batch_size)
                                  .prefetch(3))

        return ds_images_out, ds_labels_out

    def _train_pipeline(self, ds_images, ds_labels):
        """Create a training dataset pipeline.
        
        Returns: TF Dataset.
        """
        train_count = self.n_examples - self.val_count
        steps_per_epoch = int(train_count // self.batch_size)
        repeat_count = self.n_epochs * steps_per_epoch

        ds_zip = tf.data.Dataset.zip((ds_images, ds_labels))
        ds = (ds_zip.shuffle(train_count, seed=self.seed+10,
                             reshuffle_each_iteration=True)
                    .repeat(count=repeat_count)
                    .batch(self.batch_size)
                    .prefetch(3))

        return ds

    def _val_pipeline(self, ds_images, ds_labels):
        """Create a validation dataset pipeline.
        
        Returns: TF Dataset."""
        
        ds_zip = tf.data.Dataset.zip((ds_images, ds_labels))
        if self.val_count != 0:
            ds = (ds_zip.repeat(count=self.n_epochs)
                        .batch(self.val_count)
                        .prefetch(3))

        return ds

class RepeatedKFolds():
    """Repeated K-Folds cross-validator.
    
    Generates row indices for obtaining training and
    holdout/validation sets."""

    def __init__(self,
                 n_splits=5,
                 n_repeats=3,
                 seed=None):
        """Initialization.
        
        Args
            n_splits: Int. Number of splits for each fold.
            n_repeats: Int. Number of k-folds to generate.
            seed: Int. Seed for random number generator.
        """

        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.seed = seed

        self.split_sizes = np.zeros(self.n_splits, dtype=np.int8)
        self.split_indices = np.array(
            [(0, 0) for n in range(self.n_splits)], dtype=np.int8)

    def get_repeat_folds(self, n_examples):
        """Generate the training and holdout set folds n_repeat times.
        
        Args
            n_examples: Int. Total number of examples to generate
                training and holdout sets from.
        
        Returns: List of length n_splits * n_repeats. Contains 
            training-holdout set pairs."""

        self.split_sizes = self._get_split_sizes(n_examples)
        self.split_indices = self._get_split_indices()

        repeat_folds = []
        seed = self.seed
        
        for n in range(self.n_repeats):
            splits = self._get_splits(n_examples, seed)
            folds = self._get_folds(splits)
            repeat_folds.append(folds)
            
            seed += 10

        return list(chain(*repeat_folds))

    def _get_folds(self, splits):
        """Generate the training-holdout folds from a set of splits.
        
        Args
            splits: List of length n_splits. Each element is a list
                containing the row numbers from the full dataset that
                will compose the split.

        Returns: List of length n_splits. Each element is a list of
            length=2 containing training-holdout set pairs.
        """

        train = [splits.copy() for i in range(self.n_splits)]
        holdout = [train[i].pop(i) for i in range(self.n_splits)]
        train_flat = [list(chain(*row)) for row in train]

        return list(zip(train_flat, holdout))

    def _get_splits(self, n_examples, seed):
        """Generate the splits for cross-valiation.
        
        Args
            n_examples: Int. Total number of examples to generate
                training and holdout sets from.
            seed: Int. Seed for random number generator.
        
        Returns: List of length n_splits. Each element is a list
            containing the row numbers from the full dataset that will
            compose each split.
        """

        if seed is not None:
            rng = default_rng(seed)
        else:
            rng = default_rng()

        data_rows = list(range(n_examples))
        rng.shuffle(data_rows)

        split_rows = [data_rows[pair[0] : pair[1]]
                       for pair in self.split_indices]

        return split_rows

    def _get_split_indices(self):
        """Generate the indices for splitting the dataset.
        
        Returns: List of length n_splits. Each element is a tuple
            containing start and end indices for splitting the 
            dataset.
        """

        cumsum = np.cumsum(
            np.concatenate((np.array([0], dtype=np.int8), self.split_sizes)))
        
        fold_inds = np.array(
            [(cumsum[n], cumsum[n + 1]) for n in range(self.n_splits)])

        return fold_inds

    def _get_split_sizes(self, n_examples):
        """Generate the size of each split.
        
        Args
            n_examples: Int. Total number of examples to generate
                training and holdout sets from.
        
        Returns: List of ints of length=n_splits."""

        min_ex = (int(n_examples // self.n_splits)
                  * np.ones(self.n_splits, dtype=np.int8))
    
        rem = np.array(
            [1 if i < n_examples % self.n_splits else 0
                for i in range(self.n_splits)],
            dtype=np.int8)

        return np.add(min_ex, rem)

class SearchCV():
    """Optimal hyper-parameter search using cross-validation."""

    def __init__(self,
                 model,
                 params,
                 cv):
        """Initialization.
        
        Args
            model: Model object. Right now this supports dense NN
                with TF Sequential object.
            params: Dict: Model hyper-parameters as keywords with
                appropriate numbers/lists as values to search.
            cv: Cross-validator object. Use RepeatedKFolds above.
        """

        self.model = model

        self.params = params

        self.cv = cv

        self.class_le = LabelEncoder()
        self.files_labels = []
        self.rescale = 1
        self.example_count = 0

        self.mean = {}
        self.std = {}
        self.param_combs = []

    def display_results(self, metrics):
        """Display sorted results for each metric.
        
        Args
            metrics: Dict. Contains strings as keys corresponding to
                metric function as values.
        """

        means = (self.cv_results_[f'mean_{k}'] for k in metrics.keys())
        stds = (self.cv_results_[f'std_{k}'] for k in metrics.keys())
        metrics_results = ([m, s, self.cv_results_['params']]
                                for m, s in zip(means, stds))

        results_bymetric = {
            k:v for k, v in zip(metrics.keys(), metrics_results)}

        for k, v in results_bymetric.items():
            sorted_results = sort_results(v)
            print(f'Results for {k} metric:')
            print()
            for _, params, std, mean in sorted_results:
                print(f'{mean:0.03f} +/- {std:0.03f} for {params!r}')
            print()

    def fit(self,
            path_data,
            path_model_id,
            metrics=['accuracy'],
            batch_size=32,
            epochs=1,
            verbose=2,
            callbacks=None,
            shuffle=True,
            seed=None,
            validation_steps=1,
            rescale=1):
        """Fit the model.
        
        Args
            path_data: Path object. Directory containing class and
                image files.
            path_model_id: Path object. Directory containing files
                pertaining to a particular model.
            metrics: List of strings. Strings correspond to names
                for metric functions.
            batch_size: Int. Batch size of training.
            epochs: Int. Number of epochs to train for.
            verbose: Int. Verbosity setting.
            callbacks: Callback object. Code to execute after each
                epoch. This will likely be EarlyStopping().
            shuffle: Bool. Shuffle data?
            seed: Int. Seed for shuffling data.
            validation_steps: Int. Total number of validation batches
                to evaluate at the end of each epoch.
            rescale: Float. Multiplicative factor for rescaling
                data.
        """

        self.rescale = rescale
        dir_data =  str(path_data)

        fixed_params = {'epochs': epochs,
                        'verbose': verbose,
                        'callbacks': callbacks,
                        'shuffle': shuffle,
                        'validation_steps': validation_steps}

        self.mean = {k: [] for k in metrics.keys()}
        self.std = {k: [] for k in metrics.keys()}

        self._read_data_file(path_model_id)

        cv_sets = self.cv.get_repeat_folds(self.example_count)
        n_cvsets = len(cv_sets)

        param_search, model_fix_params = self._get_param_combs()

        for model_params in param_search:
            self._build_network(**model_fix_params,
                                 **model_params)

            self.model.compile(loss='binary_crossentropy',
                               optimizer='adam',
                               metrics=[v for v in metrics.values()])

            metric_results = {k: [] for k in metrics.keys()}
            epochs_results = []

            for i in range(n_cvsets):
                train_count =  len(cv_sets[i][0])
                val_count = len(cv_sets[i][1])
                steps_per_epoch = int(train_count // batch_size)
                repeat_count = epochs * steps_per_epoch

                ds_train = self._train_pipeline(dir_data, cv_sets[i][0],
                                                repeat_count, batch_size,
                                                shuffle, seed)

                ds_val = self._val_pipeline(dir_data, cv_sets[i][1],
                                            epochs, val_count)

                var_params = {'x': ds_train,
                              'validation_data': ds_val,
                              'steps_per_epoch': steps_per_epoch}

                hs = self.model.fit(**var_params, **fixed_params)

                epochs_results.append(self._find_epochs(hs))
                self._update_metric_results(hs, metrics, metric_results)
                
                if seed is not None:
                    seed += 10

            self._update_stats(metric_results)

            epochs_mean = np.around(np.mean(epochs_results))
            epochs_std = np.around(np.std(epochs_results), decimals=1)
            model_params['training_epochs'] = f'{epochs_mean} +/- {epochs_std}'
            self.param_combs.append(model_params)

        self._cv_results(metrics)

    def _build_network(self,
                        input_dim,
                        dense_layers,
                        nodes_per_layer=None,
                        hidden_act='relu',
                        output_act='sigmoid',
                        dropout_layers=None):
        """Builds a dense network.
        
        Args
            input_dim: Int. Number of features at input.
            dense_layers: Int. Number of hidden dense layers.
            nodes_per_layer: List of ints of length=dense_layers.
                Number of nodes in each hidden layer.
            hidden_act: String. Activation function to use in each
                hidden layer.
            output_act: String. Activation function to use in the
                output layer.
            dropout_layers: List of float of length=dense_layers-1.
                Fraction of inputs to drop before each hidden dense
                layer.
        """

        if nodes_per_layer is None:
            nodes = [10] * dense_layers
        else:
            nodes = nodes_per_layer

        if dropout_layers is None:
            do_layers = [0] * dense_layers
        else:
            do_layers = dropout_layers

        self.model.add(Dense(nodes[0], input_dim=input_dim,
                             activation=hidden_act))

        if dense_layers > 1:
            for l in range(1, dense_layers - 1):
                if do_layers[l - 1] != 0:
                    self.model.add(Dropout(do_layers[l - 1]))

                self.model.add(Dense(nodes[l], activation=hidden_act))

        self.model.add(Dense(1, activation=output_act))

    def _cv_results(self, metrics):
        """Compile the results from the parameter search.
        
        Args
            metrics: Dict. Metrics that are evaluated by the model
                during training.
        """

        stats = {'mean': self.mean, 'std': self.std}
        self.cv_results_ = {f'{stat}_{k}': res[k]
                            for stat, res in stats.items()
                            for k in metrics.keys()}
        self.cv_results_['params'] = self.param_combs

    def _find_epochs(self, history):
        """Determine the number of epochs before early stopping.

        Args
            history: History object. Returned after fitting model.

        Returns: Int.
        """
        
        epoch_count = len(history.history['val_loss'])

        return epoch_count

    def _get_dataset(self, dir_data, row_indices):
        """Get images and labels for a training/validation dataset.
        
        Args
            dir_data: String. Directory containing the images for each
                class.
            row_indices: List of ints. The rows from the full dataset
                to form the training/validation dataset.
        
        Returns: TF Dataset of image-label pairs.
        """

        ds_files, ds_labels = self._get_ds_files_labels(
            dir_data, row_indices)

        ds_images = ds_files.map(self._load_preprocess_image)

        ds_images_labels = tf.data.Dataset.zip((ds_images, ds_labels))

        return ds_images_labels

    def _get_ds_files_labels(self, dir_data, row_indices):
        """Get files and labels for a dataset.
        
        Args
            dir_data: String. Directory containing the images for each
                class.
            row_indices: List of ints. The rows from the full dataset
                to form the training/validation dataset.
        
        Returns: Tuple of length=2. Contains a pair of TF Datasets for
            image files and class labels.
        """

        ds_files = tf.data.Dataset.from_tensor_slices(
            [path.join(dir_data, self.files_labels[ind][0])
                for ind in row_indices])

        ds_labels = tf.data.Dataset.from_tensor_slices(
            [self.files_labels[ind][1] for ind in row_indices])

        return ds_files, ds_labels

    def _get_param_combs(self):
        """Generates all the parameters combinations to search.
        
        Returns: Tuple of length=2. The first element is a list of
            the parameter combinations as dicts. The second element is
            a dict containing the model parameters that will be held
            fixed.
        """

        search_keys = [k for k, v in self.params.items() if type(v) is list]

        layer_combs = [list(product(*self.params[k]))
                       if type(self.params[k][0]) is list
                       else self.params[k]
                       for k in search_keys]

        combs = list(product(*layer_combs))

        comb_dict = [{k: v for k, v in zip(search_keys, comb)}
                     for comb in combs]

        fixed_params = {k: v for k, v in self.params.items()
                        if k not in search_keys}
        
        return comb_dict, fixed_params

    def _load_preprocess_image(self, image_file):
        """Load image from file and perform preprocessing.
        
        Args
            image_file: String. Full path and filename of image.
            
        Returns: Tensor of a single dimension containing the pixel
            values.
        """

        image_raw = tf.io.read_file(image_file)

        image = self._preprocess_image(image_raw)

        return image

    def _preprocess_image(self, image_raw):
        """Convert raw binary to float64 and scale the pixel values.
        
        Args
            image_raw: Byte array of pixel values.
            
        Returns: Tensor.
        """

        image = tf.io.decode_raw(image_raw, tf.float64)

        return image * self.rescale

    def _read_data_file(self, path_model_id):
        """Read all image filenames and labels from a file.
        
        Args
            path_model_id: Path object. Directory containing the files
                pertaining to a particular model.
        """

        path_dataset_file = path_model_id.joinpath('training_set.csv')

        with path_dataset_file.open(mode='r', newline='') as f:
            csv_reader = reader(f, delimiter=',')
            rows = list(csv_reader)

        self.example_count = len(rows)

        img_files = [path.join(f'label_{row[1]}', row[0]) for row in rows]
        enc_labels = self.class_le.fit_transform([row[1] for row in rows])
        
        self.files_labels = [[img_files[i], enc_labels[i]]
                             for i in range(self.example_count)]

    def _train_pipeline(self,
                        dir_data,
                        train_rows,
                        repeat_count,
                        batch_size,
                        shuffle=True,
                        seed=None):
        """Create a training dataset pipeline.

        Args
            dir_data: String. Directory containing all the image
                files for each class.
            train_rows: List of ints. Each element is the row number
                to use from the full dataset.
            batch_size: Int. Batch size for training.
            seed: Int. Seed for random shuffling.
            repeat_count: Int. Number of times to repeat the dataset.
        
        Returns: TF Dataset.
        """
        
        train_count = len(train_rows)
        ds_train = self._get_dataset(dir_data, train_rows)

        if shuffle:
            ds_train = (ds_train.shuffle(train_count, seed=seed+10,
                                         reshuffle_each_iteration=True)
                        .repeat(count=repeat_count)
                        .batch(batch_size)
                        .prefetch(3))
        else:
            ds_train = (ds_train.repeat(count=repeat_count)
                        .batch(batch_size)
                        .prefetch(3))

        return ds_train

    def _val_pipeline(self,
                      dir_data,
                      val_rows,
                      repeat_count,
                      batch_size):
        """Create a validation dataset pipeline.

        Args
            dir_data: String. Directory containing all the image
                files for each class.
            val_rows: List of ints. Each element is the row number
                to use from the full dataset.
            batch_size: Int. Batch size for training.
            repeat_count: Int. Number of times to repeat the dataset.
        
        Returns: TF Dataset.
        """

        ds_val = self._get_dataset(dir_data, val_rows)
        ds_val = (ds_val.repeat(count=repeat_count)
                  .batch(batch_size)
                  .prefetch(3))

        return ds_val

    def _update_metric_results(self, history, metrics, metric_results):
        """"""

        for k in metrics.keys():
           metric_results[k].append(history.history[f'val_{k}'][-1])

    def _update_stats(self, metric_results):
        """"""

        for k in metric_results.keys():
            self.mean[k].append(np.mean(metric_results[k]))
            self.std[k].append(np.std(metric_results[k]))

def build_dense_network(model,
                        input_dim,
                        dense_layers,
                        nodes_per_layer=None,
                        hidden_act='relu',
                        output_act='sigmoid',
                        dropout_layers=None):
    """Builds a dense network.
    
    Args
        input_dim: Int. Number of features at input.
        dense_layers: Int. Number of hidden dense layers.
        nodes_per_layer: List of ints of length=dense_layers.
            Number of nodes in each hidden layer.
        hidden_act: String. Activation function to use in each
            hidden layer.
        output_act: String. Activation function to use in the
            output layer.
        dropout_layers: List of float of length=dense_layers-1.
            Fraction of inputs to drop before each hidden dense
            layer.
    """

    if nodes_per_layer is None:
        nodes = [10] * dense_layers
    else:
        nodes = nodes_per_layer

    if dropout_layers is None:
        do_layers = [0] * dense_layers
    else:
        do_layers = dropout_layers

    model.add(Dense(nodes[0], input_dim=input_dim,
                    activation=hidden_act))

    if dense_layers > 1:
        for l in range(1, dense_layers - 1):
            if do_layers[l - 1] != 0:
                model.add(Dropout(do_layers[l - 1]))

            model.add(Dense(nodes[l], activation=hidden_act))

    model.add(Dense(1, activation=output_act))

    return model

def classification_report(model, ds_test_images, ds_test_labels, threshold=0.5):
    """"""

    true_iter = ds_test_labels.as_numpy_iterator()
    y_true = np.hstack(list(true_iter))
    y_pred = model.predict(ds_test_images).flatten()

    class_labels = np.unique(y_true)
    depth = class_labels.shape[0]
    
    y_true_oh = tf.one_hot(y_true, depth=depth)
    y_pred_oh = tf.one_hot(np.where(y_pred < threshold, 0, 1), depth=depth)
    
    results = {'Accuracy': [], 'Precision': [], 'Recall': []}

    m = Accuracy()
    _ = m.update_state(y_true, np.around(y_pred).astype(int))
    results['Accuracy'].append(m.result().numpy())
    results['Precision'].append(" ")
    results['Recall'].append(" ")
    
    prec = [Precision(class_id=n) for n in class_labels]
    rec = [Recall(class_id=n) for n in class_labels]

    for p, r in zip(prec, rec):
        p.update_state(y_true_oh, y_pred_oh)
        r.update_state(y_true_oh, y_pred_oh)
        results['Accuracy'].append(" ")
        results['Precision'].append(p.result().numpy())
        results['Recall'].append(r.result().numpy())

    row_labels = ['All' if i == 0 else f'Class {i-1}'
                  for i in range(depth + 1)]

    return pd.DataFrame(data=results, index=row_labels)

def create_directory_structure(path_main):
    """Creates the directory structure for the model data.
    
    Args
        path_main: Path object. Destination for model files.
    """

    if not path_main.exists():
        path_main.mkdir(parents=True)

def get_path_image(path_data, label, filename):
    """Get a Path object for an image file.
    
    Args
        pata_data: Path object. Parent directory for alls
            training images.
        label: String. Class label for image, specifies subfolder the
            image resides in..
        filename: String. Filename for image.
    
    Returns: Path object. Full path for image file.
    """

    return path_data.joinpath(f'label_{label}', filename)

def getstats_fromimage(path_data, label, filename):
    """Calculate stats of interest from the image.
    
    Args
        path_data: Path object. Parent directory for all
            training images.
        label: String. Class label for image, specifies subfolder the
            image resides in.
        filename: String. Filename for image.
    
    Returns: 4 element tuple.
    """
    path_image = get_path_image(path_data, label, filename)
    image = np.fromfile(path_image, np.float64)

    max_ = np.amax(image)
    min_ = np.amin(image)
    mean = np.mean(image)
    std = np.std(image)

    return max_, min_, mean, std

def getstats_fromstream(path_model_id, path_data):
    """Find parameters for scaling data by streaming images.
    
    Needed when dataset is too large to fit in memory.
    
    Args
        path_model_id: Path object. Directory of CSV file containing
            list of images in the training set.
        path_data: Path object. Parent directory for all
           training images.
    
    Returns: Two element tuple. Scaling parameters for data.
    """

    path_dataset_file = path_model_id.joinpath('training_set.csv')

    with path_dataset_file.open(mode='r', newline='') as f:
        csv_reader = reader(f, delimiter=',')
        rows = list(csv_reader)

    num_pixels = np.fromfile(get_path_image(path_data, rows[0][1], rows[0][0]),
                             np.float64).shape[0]

    stats_byimage = np.array([getstats_fromimage(path_data, row[1], row[0])
                     for row in rows])

    max_ = np.amax(stats_byimage[:,0])
    min_ = np.amin(stats_byimage[:, 1])

    return num_pixels, max_, min_

def load_datasets(path_sets, path_images):
    """Load images and labels for all sets (training, test, etc.)
    
    Args
        path_sets: Path object. Directory to sets where images to use
            for each class are specified in a CSV file.
        path_images: Path object. Directory to all images for each class.
        
    Returns: List. First half of the entries are lists containing the 
        datasets. Second half of the entries are lists containing the
        corresponding labels.
    """
    dataset_files = tuple(path_set_file.name 
        for path_set_file in path_sets.glob('*.csv'))

    set_names = [dataset_file[: dataset_file.find('_')]
                 for dataset_file in dataset_files]
    
    if len(dataset_files) == 3:
        name_order = ['training', 'validation', 'test']
        set_order = tuple(dataset_files.index(f'{name}_set.csv')
                          for name in name_order)
        num_sets = 3
    else:
        training_index = dataset_files.index('training_set.csv')
        set_order = (training_index, 1 - training_index)
        num_sets = 2

    images_and_labels = [None] * num_sets * 2
    
    for k in range(num_sets):
        path_dataset_file = path_sets.joinpath(dataset_files[set_order[k]])

        with path_dataset_file.open(mode='r', newline='') as f:
            csv_reader = reader(f, delimiter=',')
            dataset = list(csv_reader)

        path_dataset_images = [path_images.joinpath(f'label_{row[1]}', row[0])
                               for row in dataset]

        images_and_labels[k] = np.array([np.fromfile(path_image, np.float64)
                                         for path_image
                                         in path_dataset_images])

        images_and_labels[k+num_sets] = [row[1] for row in dataset]

    return images_and_labels

def make_datasets(class_names, dataset_dict, path_source, path_dest, seed):
    """Prepares training, test, validation sets.
    
    Args
        class_names: E.g., dog, cat, fish.
        dataset_dict: Dictionary containing number of examples of each
            class.
        path_source: Path object. Location of all example images.
        path_dest: Path object. Destination for model files.
        seed: Set random seed for randomly choosing data.
    """
    
    create_directory_structure(path_dest)

    path_alldata = [path_source.joinpath(f'label_{class_}')
                    for class_ in class_names]

    path_imagefiles = [class_path.glob('*.bin')
                       for class_path in path_alldata]

    size = sum([v for k, v in dataset_dict.items()])
    rng = default_rng(seed)

    datasets_by_class = np.array([rng.choice([image_file.name
                                  for image_file in image_filelist],
                                  size=size, replace=False)
                                  for image_filelist in path_imagefiles])

    dataset_labels = np.array([np.full(size, class_)
                               for class_ in class_names])

    if not path_dest.exists():
        path_dest.mkdir(parents=True)

    start=0
    for set_name, n_examples in dataset_dict.items():
        stop = start + n_examples

        filename = f'{set_name}_set.csv'
        path_file = path_dest.joinpath(filename)
        
        images = datasets_by_class[:,start:stop].flatten()
        labels = dataset_labels[:,start:stop].flatten()
        rows = np.transpose(np.vstack((images, labels))).tolist()

        with path_file.open(mode='w', newline='') as f:
            csv_writer = writer(f)
            csv_writer.writerows(rows)

        start = n_examples

def sort_results(metric_results):
    """Sort metric performance results by the mean.
    
    Args
        metric_results: List of lists. First two lists are mean and
            standard deviation of metric (accuracy, ...) from cross
            validation while tuning model parameters. Last list is
            a dictionary of the corresponding values of the model
            hyper-parameters being tuned.
    
    Returns: A numpy array of shape (n_param_combinations, 3)
        where n_param_combinations is the number of hyper-parameter
        combinations tested while tuning the model."""

    means, stds, params_list = metric_results
    dtype = [('index', int), ('params_list', object), ('std', float), ('mean', float)]

    #Sort will fail when attempting to rank based on the
    #dictionary 'params_list' when encountering identical mean and
    #standard deviations. To avoid this, use a list of distinct
    #integers to break the tie.
    values = zip(range(len(means)), params_list, stds, means)

    a = np.sort(np.array(list(values), dtype=dtype),
                kind='mergesort', order=['mean', 'std', 'index'])
    return np.flip(a, axis=-1)

