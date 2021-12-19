#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
'''


import os
import sys
import math
import array
import pickle
import zipfile
import requests
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from Utils_ import gini
import scipy.sparse as sps
import matplotlib.pyplot as plt

class DataReader(object):
    """
    Generic class that implements utilities for datasets
    """

    # MANUALLY SET this taking into account project root
    # datasets_dir = os.path.join(CONSTANTS['root_dir'], 'datasets')
    all_datasets_dir = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, use_cols=None, split_ratio=(0.6, 0.2, 0.2), stratified_on='item_popularity', sample=1.0,
                 header=False, delim=',', implicit=False, remove_top_pop=0.0, use_local=True, force_rebuild=False,
                 save_local=True, min_ratings_user=1, min_ratings_item=1, duplicate='first', verbose=True, seed=1234):
        """
        Constructor
        """

        super().__init__()

        if use_cols is None:
            use_cols = {'user_id': 0, 'item_id': 1, 'rating': 2}

        if sum(split_ratio) != 1.0 or len(split_ratio) != 3:
            raise AttributeError('Split ratio of train, test and validation must sum up to 1')

        self.use_local = use_local
        self.force_rebuild = force_rebuild
        self.save_local = save_local
        self.min_ratings_user = min_ratings_user
        self.min_ratings_item = min_ratings_item
        self.verbose = verbose

        self.use_cols = use_cols
        self.split_ratio = split_ratio
        self.stratified_on = stratified_on  #TODO: stratification as popularity and considering user/item as class (sklearn-like)
        self.sample = sample
        self.header = header
        self.delimiter = delim
        self.remove_top_pop = remove_top_pop
        self.duplicate = duplicate

        if len(use_cols) < 3:
            self.implicit = True
        else:
            self.implicit = implicit

        self.config = {
            'use_cols': self.use_cols,
            'split_ratio': self.split_ratio,
            'stratified_on': self.stratified_on,
            'sample': self.sample,
            'header': self.header,
            'delimiter': self.delimiter,
            'implicit': self.implicit,
            'remove_top_pop': self.remove_top_pop,
            'min_ratings_user': self.min_ratings_user,
            'min_ratings_item': self.min_ratings_item,
            'duplicate': self.duplicate,
            'seed': seed
        }

    def build_local(self, ratings_file, split=True):
        """
        Builds sparse matrices from ratings file

        Parameters
        ----------
        ratings_file: str
            The full path to the ratings' file

        split: boolean, default True
            Flag indicating whether to build only the full URM or also to split it in train-validation-test
        """
        if os.path.isfile(ratings_file):
            self.URM = self.build_URM(file=ratings_file, use_cols=self.use_cols, delimiter=self.delimiter,
                                      header=self.header, save_local=self.save_local, remove_top_pop=self.remove_top_pop,
                                      duplicate=self.duplicate, sample=self.sample, verbose=self.verbose)

            if split:
                self.URM_train, \
                self.URM_test, \
                self.URM_validation = self.split_urm(self.URM, split_ratio=self.split_ratio, save_local=self.save_local,
                                                     implicit=self.implicit, min_ratings_user=self.min_ratings_user,
                                                     min_ratings_item=self.min_ratings_item, verbose=self.verbose,
                                                     save_dir=os.path.dirname(ratings_file))

            try:
                with open(os.path.join(os.path.dirname(ratings_file), 'config.pkl'), 'wb') as f:
                    pickle.dump(self.config, f)
            except AttributeError:
                print('config is not initialized in ' + self.__class__.__name__ + '! No config saved!', file=sys.stderr)

        else:
            print(ratings_file + ' not found. Building remotely...')
            self.build_remote()

    def get_ratings_file(self):
        """
        Downloads the dataset and sets self.ratings_file. If downlaoded file is a zip file, it extracts it, otherwise
        it saves a .csv file
        """
        downloaded_file = self.download_url(self.url, verbose=self.verbose)
        extension = os.path.splitext(downloaded_file)[1]
        if extension == '.zip':
            zfile = zipfile.ZipFile(downloaded_file)
            try:
                self.ratings_file = zfile.extract(self.data_file,
                                    os.path.join(self.all_datasets_dir, os.path.dirname(downloaded_file)))
                # Archive will be deleted
                os.remove(downloaded_file)
            except (FileNotFoundError, zipfile.BadZipFile):
                print('Either file ' + self.data_file + ' not found or ' + os.path.split(self.url)[-1] + ' is corrupted',
                      file=sys.stderr)
                raise
        else:
            self.ratings_file = downloaded_file

    def build_remote(self, split=True):
        """
        Builds sparse matrices
        """
        self.get_ratings_file()
        self.URM = self.build_URM(file=self.ratings_file, use_cols=self.use_cols, delimiter=self.delimiter,
                                    header=self.header, save_local=self.save_local, remove_top_pop=self.remove_top_pop,
                                    duplicate=self.duplicate, sample=self.sample, verbose=self.verbose)

        if split:
            self.URM_train, \
            self.URM_test, \
            self.URM_validation = self.split_urm(self.URM, split_ratio=self.split_ratio, save_local=self.save_local,
                                                 implicit=self.implicit, min_ratings_user=self.min_ratings_user,
                                                 min_ratings_item=self.min_ratings_item, verbose=self.verbose,
                                                 save_dir=os.path.dirname(self.ratings_file))

        try:
            with open(os.path.join(os.path.dirname(self.ratings_file), 'config.pkl'), 'wb') as f:
                pickle.dump(self.config, f)
        except AttributeError:
            print('config is not initialized in ' + self.__class__.__name__ + '!', file=sys.stderr)
            raise

    def download_url(self, url, verbose=True):
        """
        Downloads the file found at url.
        To be used to download datasets which are then saved in self.datasets_dir

        Parameters
        ----------
        url: str
            URL where file is located.
        verbose: boolean, default True
            Boolean value whether to show logging
        desc: str
            Description to be used in the download progress bar

        Returns
        -------
        path_to_file: str
            absolute path of the downloaded file from the PROJECT ROOT
        """

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = 0
            if 'content-length' in response.headers.keys():
                total_size = int(response.headers['content-length'])
            chunk_size = 1024 * 4

            filename = url.split('/')[-1]
            abs_path = os.path.join(self.all_datasets_dir,  self.dataset_dir, filename)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)

            desc = 'Downloading ' + self.DATASET_NAME + ' from ' + url
            pbar = tqdm(total=total_size, desc=desc, unit='B', unit_scale=True, unit_divisor=1024, disable=not verbose and total_size != 0)

            with open(abs_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(chunk_size)
                pbar.close()
            return abs_path

        else:
            raise requests.HTTPError('Request for download returned with status ' + response.status_code)

    def download_kaggle_dataset(self, dataset, files='all', verbose=True):
        """
        Downloads a dataset from Kaggle as specified by parameter dataset.
        Please set username and Kaggle API key in ~/.kaggle/kaggle.json.

        :param dataset: Name of the dataset as specified by Kaggle of the format <owner>/<dataset-name>.
                Can be searched by running `kaggle datasets list -s DATASET_NAME`
        :param files: Name of the file to download. Can be found by running `kaggle datasets files DATASET_NAME`.
                If `all` then all the files of the dataset will be downloaded
        :param verbose: Default True
        """


        # By default kaggle.json should reside within ~/.kaggle
        kaggle_filepath = os.path.expanduser('~/.kaggle/kaggle.json')

        # If kaggle.json does not exist
        if not os.path.exists(kaggle_filepath):
            raise IOError('File kaggle.json not found in ~/.kaggle. Please place it there and rerun.')

            # Create it and store it
            # if self.defs.username is None and self.defs.key is None:
            #     raise ValueError('Credentials must be provided either through kaggle.json or username and key in order to download datasets from Kaggle.com.')

            # else:
            #     if verbose:
            #         print(kaggle_filepath + ' missing. Using hardcoded username and key from Utils.py to create it.')
            #     kaggle_dict = {}
            #     kaggle_dict['username'] = self.defs.username
            #     kaggle_dict['key'] = self.defs.key
            #     os.makedirs(os.path.dirname(kaggle_filepath), exist_ok=True)
            #     with open(kaggle_filepath, 'w') as f:
            #         json.dump(kaggle_dict, f)
            #     subprocess.run(['chmod', '600', kaggle_filepath])

        # First create a folder inside datasets with the name of the dataset (without the <owner> part)
        dataset_path = os.path.join(self.all_datasets_dir, dataset.split('/')[-1])
        os.makedirs(dataset_path, exist_ok=True)

        # kaggle is installed in bin folder where python is
        kaggle_cmdpath = os.path.join(os.path.dirname(sys.executable), 'kaggle')

        # Run kaggle command through subprocess
        if files == 'all':
            subprocess.run([kaggle_cmdpath, 'datasets', 'download', dataset, '-p', dataset_path, '--force'])
        elif isinstance(files, list):
            for f in files:
                subprocess.run([kaggle_cmdpath, 'datasets', 'download', dataset, '-p', dataset_path, '--force', '-f', f])
        elif isinstance(files, str):
            subprocess.run([kaggle_cmdpath, 'datasets', 'download', dataset, '-p', dataset_path, '--force', '-f', files])
        else:
            raise ValueError('files argument accepts either `all`, a single filename or a list of filenames.')

        # Unzip all files downloaded and delete zip files
        if verbose:
            print('Extracting downloaded files. Archive files will be removed.')
        for filename in os.listdir(dataset_path):
            fpath = os.path.join(dataset_path, filename)
            if os.path.isfile(fpath) and os.path.splitext(filename)[1] == '.zip':
                zfile = zipfile.ZipFile(fpath)
                zfile.extractall(path=dataset_path)
                os.remove(fpath)
        #TODO: need to find a fast way to merge all files downloaded

    def read_interactions(self,
                          file,
                          use_cols={'user_id': 0, 'item_id': 1, 'rating': 2},
                          delimiter=',',
                          header=False,
                          duplicate='first',
                          verbose=True):
        """
        Reads the interactions data file and fills the rows, columns and data arrays

        Parameters
        ----------
        file: str
            Absolute/relative path from the project root to the file with the ratings.

        use_cols: dict, default {'user_id':0, 'item_id':1, 'rating':2}
            Columns to be used from the file as dict. DO NOT change dict keys.

        delimiter: str, default `,`
            Delimiter of the file.

        header: boolean, default False
            Flag indicating whether interactions' file has a header.

        duplicate: str [`first`, `last`], default `first`
            In case of duplicate interactions, keep only the first/last one.

        verbose: boolean, default True

        Returns:
        --------
        rows: array.array
        cols: array.array
        data: array.array
        """

        rows = array.array('I')
        cols = array.array('I')
        data = array.array('f')

        unique_interactions = {}

        with open(file, 'r') as f:

            if header:
                f.readline()

            if verbose:
                print('Filling the full URM...')

            # # Netflix Dataset requires preprocessing and it's big so
            # # it is better to generate the matrices in one reading
            # if netflix_process:
            #     current_item_id = -1

            #     for line in f:
            #         row_data = line.split(delimiter)

            #         if len(row_data) == 1:
            #             # This is an item id
            #             current_item_id = int(row_data[0][:-2])
            #         else:
            #             rows.append(int(row_data[0]))
            #             data.append(int(row_data[1]))
            #             cols.append(current_item_id)

            # else:
            if self.implicit:
                for i, line in enumerate(f):
                    row_data = line.split(delimiter)
                    r = int(row_data[use_cols['user_id']])
                    c = int(row_data[use_cols['item_id']])

                    key = str(r) + '_' + str(c)
                    if unique_interactions.get(key, False):
                        continue
                    else:
                        unique_interactions[key] = i

                    rows.append(r)
                    cols.append(c)
                    data.append(1.0)

            else:
                for i, line in enumerate(f):
                    row_data = line.split(delimiter)
                    r = int(row_data[use_cols['user_id']])
                    c = int(row_data[use_cols['item_id']])
                    d = float(row_data[use_cols['rating']])

                    key = str(r) + '_' + str(c)
                    idx = unique_interactions.get(key, False)
                    if idx:
                        if duplicate == 'last':
                            data[idx] = d
                        continue
                    else:
                        unique_interactions[key] = i

                    rows.append(r)
                    cols.append(c)
                    data.append(d)

        del unique_interactions
        return rows, cols, data

    def remove_coldstart_items(self, URM):
        URM = URM.tocsc()
        col_indices_mask = URM.sum(axis=0).A1 > 0
        return URM[:, col_indices_mask].tocoo()

    def build_URM(self,
                  file,
                  use_cols={'user_id': 0, 'item_id': 1, 'rating': 2},
                  delimiter=',',
                  header=False,
                  save_local=True,
                  remove_top_pop=0.0,
                  duplicate='first',
                  sample=1.0,
                  verbose=True):
        """
        Builds the URM from interactions data file.

        Parameters
        ----------
        file: str
            Absolute/relative path from the project root to the file with the ratings.

        use_cols: dict, default {'user_id':0, 'item_id':1, 'rating':2}
            Columns to be used from the file as dict. DO NOT change dict keys.

        delimiter: str, default `,`
            Delimiter of the file.

        header: boolean, default False
            Flag indicating whether interactions' file has a header.

        save_local: boolean, default True
            Flag indicating whether the URM should be saved locally.

        remove_top_pop: float, default 0.0
            Fraction of most popular items to be removed from the final URM.

        duplicate: str [`first`, `last`], default `first`
            In case of duplicate interactions, keep only the first/last one.

        sample: float, default 1.0
            Ratio of the dataset to sample, user-wise.

        verbose: boolean, default True
            Flag indicating whether logging should be printed out.


        Returns
        -------
        URM: scipy.sparse.coo_matrix
            The full URM in COO format.
        """

        rows, cols, data = self.read_interactions(file, use_cols, delimiter, header, duplicate, verbose)

        unique_items, item_counts = np.unique(cols, return_counts=True)

        if remove_top_pop > 0.0:
            k = int(np.floor(len(unique_items) * remove_top_pop))
            sorted_indices = np.argsort(item_counts)[::-1]
            unique_items = unique_items[sorted_indices][k:]
            col_mask = np.isin(cols, unique_items)
            cols = np.frombuffer(cols, dtype=np.int32)[col_mask]
            rows = np.frombuffer(rows, dtype=np.int32)[col_mask]
            if not isinstance(data, np.ndarray):
                data = np.frombuffer(data, dtype=np.float32)[col_mask]
            else:
                data = data[col_mask]

        unique_users = np.unique(rows)

        shape = (len(unique_users), len(unique_items))

        self.row_to_user = dict(zip(unique_users, range(len(unique_users))))
        self.col_to_item = dict(zip(unique_items, range(len(unique_items))))

        coo_rows = pd.Series(rows).map(self.row_to_user).values
        coo_cols = pd.Series(cols).map(self.col_to_item).values

        self.URM = sps.coo_matrix((data, (coo_rows, coo_cols)), shape=shape, dtype=np.float32)

        # If sample is different from 0, sample at random row-wise
        if sample != 1.0:
            number_rows_remaining = int(self.URM.shape[0] * sample)
            remaining_rows = np.random.randint(low=0, high=self.URM.shape[0], size=number_rows_remaining)
            self.URM = self.remove_coldstart_items(self.URM.tocsr()[remaining_rows])

        if save_local:
            if verbose:
                print('Saving full URM locally...')

            sps.save_npz(os.path.join(os.path.dirname(file), 'URM'), self.URM, compressed=True)
            np.save(os.path.join(os.path.dirname(file), 'row_to_user'), self.row_to_user, allow_pickle=True)
            np.save(os.path.join(os.path.dirname(file), 'col_to_item'), self.col_to_item, allow_pickle=True)

        # Delete arrays to save space
        data, rows, cols = None, None, None

        return self.URM

    def split_urm(self, URM=None, split_ratio=(0.6, 0.2, 0.2), save_local=True, implicit=False, min_ratings_user=2,
                  min_ratings_item=1, verbose=True, save_dir=None):
        """
        Creates sparse matrices from full URM.

        Parameters
        ----------
        URM: scipy.sparse.coo_matrix
            The full URM in COO format.

        split_ratio: tuple, default (0.6, 0.2, 0.2) meaning 0.6 train, 0.2 test, 0.2 validation
            Train-Test-Validation split ratio. Must sum to 1.

        save_local: boolean, default True
            Flag indicating whether to save the resulting sparse matrices locally.

        implicit: boolean, default False
            Flag indicating whether the interactions should be implicit. If True the column of interactions is substituted with ones.

        min_ratings_user: int, default 2 (one for the train set, the other for the test set)
            Number of ratings that each user must have in order to be included in any of the splits.

        min_ratings_item: int, default 1
            Number of ratings that each item must have.

        verbose: boolean, default True
            Flag indicating whether to print logging.

        save_dir: str, default None
            Directory where to save the sparse matrices.


        Returns
        -------
        URM_train: scipy.sparse.csr_matrix
            URM in CSR format to be used for training.

        URM_test: scipy.sparse.csr_matrix
            URM in CSR format to be used for testing.

        URM_validation: scipy.sparse.csr_matrix
            URM in CSR format to be used for validation.
        """

        if URM is None:
            try:
                URM = self.URM
            except AttributeError:
                print('URM is not initialized in ' + self.__class__.__name__ + '!', file=sys.stderr)
                raise

        if implicit:
            URM.data = np.ones(len(URM.data))

        URM_csr = sps.csr_matrix(URM)

        # Compute the dense k-l-core
        if min_ratings_user + min_ratings_item > 2:
            if verbose:
                if min_ratings_user >= 2 and min_ratings_item >= 2:
                    print('Computing the dense core by removing users with less than ' + str(min_ratings_user) +
                          ' ratings and items with less than ' + str(min_ratings_item) + ' ratings...')
                elif min_ratings_user >= 2:
                    print('Removing users with less than ' + str(min_ratings_user) + ' ratings...')
                elif min_ratings_item >= 2:
                    print('Removing items with less than ' + str(min_ratings_item) + ' ratings...')

            done = False
            while not done:
                if min_ratings_user >= 2:
                    user_mask = np.ediff1d(URM_csr.indptr) >= min_ratings_user
                    URM_csr = URM_csr[user_mask]
                    URM_csr = self.remove_coldstart_items(URM_csr).tocsr()

                if min_ratings_item >= 2:
                    URM_csr = URM_csr.T.tocsr()
                    item_mask = np.ediff1d(URM_csr.indptr) >= min_ratings_item
                    URM_csr = URM_csr[item_mask]
                    URM_csr = self.remove_coldstart_items(URM_csr).tocsr()
                    URM_csr = URM_csr.T.tocsr()

                # compute again user and item masks
                user_mask = np.ediff1d(URM_csr.indptr) < min_ratings_user
                item_mask = np.ediff1d(URM_csr.T.tocsr().indptr) < min_ratings_item

                done = sum(user_mask) + sum(item_mask) == 0

        URM_csr.eliminate_zeros()

        if verbose:
            print('Splitting the full URM into train, test and validation matrices...')

        # choice = np.random.choice(['train', 'test', 'valid'], p=split_ratio, size=len(URM.data))

        # Keep the `split_ratio` per user and not per total ratings.
        # Doing this through iteration, need to find a better solution
        choice = []
        for u in range(URM_csr.shape[0]):
            indices = URM_csr.indices[URM_csr.indptr[u]: URM_csr.indptr[u+1]]
            no_interactions = len(indices)
            if no_interactions == 1:
                choice.extend(['train'])
            elif no_interactions == 2:
                if split_ratio[1] == 0:
                    first_choice = ['train', 'validation'][np.random.randint(2)]
                    second_choice = 'train' if first_choice == 'validation' else 'validation'
                else:
                    first_choice = ['train', 'test'][np.random.randint(2)]
                    second_choice = 'train' if first_choice == 'test' else 'test'
                choice.extend([first_choice, second_choice])
            else:
                selection = np.random.choice(['train', 'test', 'valid'], p=split_ratio, size=no_interactions)

                if (selection == 'train').sum() == 0 or \
                    (split_ratio[1] != 0 and (selection == 'test').sum() == 0) or \
                    (split_ratio[2] != 0 and (selection == 'validation').sum() == 0):
                    no_trains = int(no_interactions * split_ratio[0])
                    no_tests = math.ceil(no_interactions * split_ratio[1])

                    selection = np.array(['train'] * no_interactions)
                    possibilities = np.array(range(no_interactions))
                    select_trains = np.random.choice(possibilities, size=no_trains, replace=False)
                    remaining_possibilities = list(set(possibilities).difference(set(select_trains)))
                    select_tests = np.random.choice(remaining_possibilities, size=no_tests, replace=False)
                    select_validation = list(set(remaining_possibilities).difference(set(select_tests)))
                    selection[select_tests] = 'test'
                    selection[select_validation] = 'validation'

                choice.extend(selection.tolist())

        URM = sps.coo_matrix(URM_csr)
        del URM_csr

        choice = np.array(choice)
        shape = URM.shape
        URM_train = sps.coo_matrix((URM.data[choice == 'train'], (URM.row[choice == 'train'], URM.col[choice == 'train'])), shape=shape, dtype=np.float32)
        URM_test = sps.coo_matrix((URM.data[choice == 'test'], (URM.row[choice == 'test'], URM.col[choice == 'test'])), shape=shape, dtype=np.float32)
        URM_validation = sps.coo_matrix((URM.data[choice == 'valid'], (URM.row[choice == 'valid'], URM.col[choice == 'valid'])), shape=shape, dtype=np.float32)

        self.URM_train = URM_train.tocsr()
        self.URM_test = URM_test.tocsr()
        self.URM_validation = URM_validation.tocsr()

        if save_local and save_dir is not None:
            if verbose:
                print('Saving matrices locally...')

            sps.save_npz(os.path.join(save_dir, 'URM_train'), self.URM_train, compressed=True)
            sps.save_npz(os.path.join(save_dir, 'URM_test'), self.URM_test, compressed=True)
            sps.save_npz(os.path.join(save_dir, 'URM_validation'), self.URM_validation, compressed=True)

        return self.URM_train, self.URM_test, self.URM_validation

    def get_CV_folds(self, URM=None, folds=10, verbose=True):
        """
        Generator function implementing cross-validation from interactions data file.

        :param URM: URM to use for generating the folds. If None, the attribute URM of the class will be used.
        :param folds: Number of CV folds
        :param verbose: True to print logging

        Yields train and test matrices in CSR format
        """

        if verbose:
            print('Generating train and test folds...')

        if URM is None:
            try:
                URM = self.URM
            except AttributeError:
                print('URM is not initialized in ' + self.__class__.__name__ + '!', file=sys.stderr)
                raise

        choice = np.random.choice(range(folds), size=len(URM.data))
        shape = URM.shape
        for i in range(folds):
            URM_test = sps.coo_matrix((URM.data[choice == i], (URM.row[choice == i], URM.col[choice == i])), shape=shape, dtype=np.float32)
            URM_train = sps.coo_matrix((URM.data[choice != i], (URM.row[choice != i], URM.col[choice != i])), shape=shape, dtype=np.float32)
            yield URM_train.tocsr(), URM_test.tocsr()

    def get_URM_full(self, transposed=False):
        try:
            if transposed:
                return self.URM.T
            else:
                return self.URM
        except AttributeError:
            print('URM is not initialized in ' + self.__class__.__name__ + '!', file=sys.stderr)
            raise

    def get_URM_train(self, transposed=False):
        try:
            if transposed:
                return self.URM_train.T.tocsr()
            return self.URM_train
        except AttributeError:
            print('URM_train is not initialized in ' + self.__class__.__name__ + '!')
            raise

    def get_URM_test(self, transposed=False):
        try:
            if transposed:
                return self.URM_test.T.tocsr()
            return self.URM_test
        except AttributeError:
            print('URM_test is not initialized in ' + self.__class__.__name__ + '!', file=sys.stderr)
            raise

    def get_URM_validation(self, transposed=False):
        try:
            if transposed:
                return self.URM_validation.T.tocsr()
            return self.URM_validation
        except AttributeError:
            print('URM_validation is not initialized in ' + self.__class__.__name__ + '!', file=sys.stderr)
            raise

    def process(self, split=True):
        """
        Read prebuild sparse matrices or generate them from ratings file.
        """

        # Check if files URM_train, URM_test and URM_validation OR URM already exists first
        # If not, build locally the sparse matrices using the ratings' file

        if self.use_local:
            ratings_file = os.path.join(self.all_datasets_dir, self.dataset_dir, self.data_file)
            self.matrices_path = os.path.join(self.all_datasets_dir, os.path.dirname(ratings_file))

            train_path = os.path.join(self.matrices_path, 'URM_train.npz')
            test_path = os.path.join(self.matrices_path, 'URM_test.npz')
            valid_path = os.path.join(self.matrices_path, 'URM_validation.npz')
            urm_path = os.path.join(self.matrices_path, 'URM.npz')

            # Read the build config and compare with current build
            config_path = os.path.join(self.matrices_path, 'config.pkl')
            if os.path.isfile(config_path):
                with open(config_path, 'rb') as f:
                    config = pickle.load(f)

                try:
                    if self.config != config:
                        if self.verbose:
                            print('Local matrices built differently from requested build. Setting force_rebuild = True.')
                        self.force_rebuild = True
                except AttributeError:
                    print('config is not initialized in ' + self.__class__.__name__ + '!', file=sys.stderr)
                    self.force_rebuild = True

            else:
                if self.verbose:
                    print('Configuration file not found. Setting force_rebuild = True.')
                self.force_rebuild = True

            if not self.force_rebuild:
                if os.path.isfile(train_path) and os.path.isfile(test_path) and os.path.isfile(valid_path):
                    if self.verbose:
                        print('Loading train, test and validation matrices locally...')

                    self.URM_train = sps.load_npz(train_path)
                    self.URM_test = sps.load_npz(test_path)
                    self.URM_validation = sps.load_npz(valid_path)

                    if os.path.isfile(urm_path):
                        self.URM = sps.load_npz(urm_path)

                elif os.path.isfile(urm_path):
                    if self.verbose:
                        print('Building from full URM...')

                    if os.path.isfile(urm_path):
                        self.URM = sps.load_npz(urm_path)

                        if split:
                            self.URM_train, \
                            self.URM_test, \
                            self.URM_validation = self.split_urm(self.URM, split_ratio=self.split_ratio,
                                                                 save_local=self.save_local,
                                                                 min_ratings_user=self.min_ratings_user,
                                                                 verbose=self.verbose,
                                                                 save_dir=os.path.dirname(urm_path))
                    else:
                        if self.verbose:
                            print("Full URM not found. Building from ratings' file...")

                        if os.path.exists(ratings_file):
                            self.build_local(ratings_file, split)
                        else:
                            self.build_remote(split)
                else:
                    if self.verbose:
                        print("Matrices not found. Building from ratings' file...")

                    if os.path.exists(ratings_file):
                        self.build_local(ratings_file, split)
                    else:
                        self.build_remote(split)

            else:
                if self.verbose:
                    print("Rebuilding asked. Building from ratings' file...")

                if os.path.exists(ratings_file):
                    self.build_local(ratings_file, split)
                else:
                    self.build_remote(split)

        # Either remote building asked or ratings' file is missing
        else:
            self.build_remote(split)

    def describe(self, save_plots=False, path=None):
        """
        Describes the full URM
        """

        print('Dataset:', self.DATASET_NAME)

        try:
            # The URM is assumed to have shape users x items
            # We change the data to implicit to count the number of interactions
            explicit_data = self.URM.data
            self.URM.data = np.ones(len(self.URM.data))
            no_users = self.URM.shape[0]
            no_items = self.URM.shape[1]
            ratings = self.URM.nnz
            density = ratings / no_users / no_items
            items_per_user = self.URM.sum(axis=1).A1
            profile_length_95th = items_per_user[items_per_user <= np.percentile(items_per_user, 95)]
            users_per_item = self.URM.sum(axis=0).A1
            cold_start_users = int(np.sum(np.where(items_per_user == 0)))
            mean_item_per_user = np.mean(items_per_user)
            min_item_per_user = int(np.min(items_per_user))
            max_item_per_user = int(np.max(items_per_user))
            unique_ratings = np.unique(explicit_data)
            gini_index = gini(users_per_item)

            print('Users: {:d}\nItems: {:d}\nRatings: {:d}\nDensity: {:.5f}%\nCold start users: {:d}\n'
                    'Minimum items per user: {:d}\nMaximum items per user: {:d}\nAvg.items per user: {:.2f}\n'
                    'Gini index: {:.2f}\nUnique ratings:{}\n'
                  .format(no_users, no_items, ratings, density*100, cold_start_users,
                        min_item_per_user, max_item_per_user, mean_item_per_user, gini_index, unique_ratings))

            plt.style.use('fivethirtyeight')
            # sns.set_style('darkgrid')
            sns.set_context('paper', font_scale=1.75)

            fig1, ax1 = plt.subplots(figsize=(10, 10))
            ax1 = sns.distplot(items_per_user, rug=False, kde=False, label='Distribution of per-user number of interactions', ax=ax1)
            ax1.set_ylabel('users', fontsize=20)
            ax1.set_xlabel('no. interactions per user', fontsize=20)

            fig3, ax3 = plt.subplots(figsize=(10, 10))
            ax3 = sns.distplot(profile_length_95th, rug=False, kde=False, label='Distribution of 95th percentile per-user number of interactions', ax=ax3)
            ax3.set_ylabel('users', fontsize=20)
            ax3.set_xlabel('no. interactions per user', fontsize=20)

            # fig2, ax2 = plt.subplots(figsize=(10, 10))
            # sns.distplot(users_per_item, rug=False, kde=False, label='count_items', axlabel='interactions', ax=ax2)
            # ax2.set_ylabel('items')
            # ax2.set_xlabel('no. interactions per item')

            if save_plots:
                fig1.savefig(os.path.join(path if path is not None else self.matrices_path, self.DATASET_NAME + '_user_interaction_distr.png'), bbox_inches="tight")
                fig3.savefig(os.path.join(path if path is not None else self.matrices_path, self.DATASET_NAME + '_95th_interaction_distr.png'), bbox_inches="tight")
                # fig2.savefig(os.path.join(path if path is not None else self.matrices_path, self.DATASET_NAME + '_item_interaction_distr.png'), bbox_inches="tight")
            else:
                plt.show()
        except AttributeError:
            print('URM is not initialized in ' + self.__class__.__name__ + '!', file=sys.stderr)
            raise
