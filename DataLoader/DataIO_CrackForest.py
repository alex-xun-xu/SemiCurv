## Data IO Class for Crack Forest Dataset

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import scipy.io as scio
import copy
from skimage.morphology import square, binary_closing
import pathlib

addUnlab_path_dict = {
    'Crack500': '../Dataset/Crack500/CRACK500-20200128T063606Z-001/CRACK500/Cutomized'}


class DataIO():

    def __init__(self, batch_size, seed_split=0, seed_label=0, label_percent=0.05,
                 data_path=os.path.abspath(os.path.join(pathlib.Path(__file__).parent.resolve(),'../Dataset/CrackForest-dataset')),
                 train_val_test_split=(81, 9.5, 9.5),
                 add_unlab='None', swap_label=False):
        '''
        Initialize DataIO
        :param batch_size: the batchsize fo training inference
        :param seed_split: the random seed for dataset split generation
        :param seed_label: the random seed for labeled train set generation
        :param label_percent: percentage of labeled train set
        :param data_path: path to dataset
        :param add_unlab: additional unlabeled data
        '''

        self.data_path = data_path  # the path to dataset
        self.num_all_data = -1  # number of all data/samples in this dataset
        self.original_image_size = (320, 480)
        self.train_val_test_split = train_val_test_split  # train validation and testing data split in percent
        self.label_percent = label_percent  # percentage of labeled data
        self.seed_split = seed_split  # seed to generate data split
        self.seed_label = seed_label  # seed to generate labeled training set
        self.batch_size = batch_size  # batch size
        self.add_unlab = add_unlab  # additional unlabeled dataset
        self.swap_label = swap_label  # swap label 0 with 1

        self.train_index = None
        self.val_index = None
        self.test_index = None
        self.num_train = None
        self.num_val = None
        self.num_test = None

        ## Initialize data sample pointer
        self.InitPointer()

    def LoadDataset(self):
        '''
        load the entire dataset.
        :return:
        '''

        ## Load Images & GT
        no_labeled_data = 0  # counter for labeled data
        data_idx = 0  # index for all loaded data sample
        self.all_data = {}
        img_path = os.path.join(self.data_path, 'image')
        gt_path = os.path.join(self.data_path, 'groundTruth')
        gt_thin_path = os.path.join(self.data_path, 'groundTruth_thin')
        ## List all images, GTs
        img_files = os.listdir(img_path)
        img_files.sort()
        gt_files = os.listdir(gt_path)
        gt_files.sort()
        ## Collect image, gt files
        for img_file in img_files:
            img_name = img_file.split('.')[0]  # image name
            # img = plt.imread(os.path.join(img_path, img_file)).astype(np.float32)  # load image array
            img = plt.imread(os.path.join(img_path, img_file))  # load image array
            # Load image gt
            gt_filepath = os.path.join(gt_path, '{}.mat'.format(img_name))
            if os.path.exists(gt_filepath):
                array = scio.loadmat(gt_filepath)['groundTruth']
                boundaries = array['Boundaries'][0, 0].astype(np.uint8)
                seg = (array['Segmentation'][0, 0]).astype(np.uint8)
                seg = binary_closing(np.maximum(boundaries, (seg == 2)), square(2)) * 255
                seg = (seg > 127.).astype('float32')  # convert it to binary (0 and 1)
                no_labeled_data += 1
            else:
                seg = None
            # Load thinned ground-truth
            gt_thin_filepath = os.path.join(gt_thin_path, '{}_thin.mat'.format(img_name))
            if os.path.exists(gt_thin_filepath):
                seg_thin = scio.loadmat(gt_thin_filepath)['groundTruth']
            else:
                seg_thin = None

            if self.swap_label:
                # swap label 1 with 0
                if seg is not None:
                    seg = 1. - seg
                if seg_thin is not None:
                    seg_thin = 1. - seg_thin

            self.all_data.update(
                {img_name: {'img': img, 'gt': seg, 'gt_thin': seg_thin, 'name': img_name, 'index': data_idx}})
            data_idx += 1

        self.num_all_data = no_labeled_data

        #### Load Additional Unlabeled Data
        self.all_add_data = {}  # additional unlabeled data
        self.add_train_names = []
        if self.add_unlab != 'None':
            addUnlab_filepath = os.path.join(addUnlab_path_dict[self.add_unlab], 'train4CFD.mat')
            tmp = scio.loadmat(addUnlab_filepath)
            allImgNames = tmp['allImgNames'][0].split(' ')
            data_idx = 0
            for img, gt, img_name in zip(tmp['allImgs'], tmp['allGTs'], allImgNames):

                if self.swap_label:
                    # swap label 1 with 0
                    if seg is not None:
                        seg = 1. - seg
                    if seg_thin is not None:
                        seg_thin = 1. - seg_thin

                self.all_data.update(
                    {img_name: {'img': img, 'gt': gt, 'gt_thin': None, 'name': img_name, 'index': data_idx}})
                data_idx += 1
                self.add_train_names.append(img_name)

    def CountClassImbalance(self):
        '''
        Count Class Imbalance
        :return:
        '''

        pos_pixel_ratio = []
        for key in self.all_data:
            gt = self.all_data[key]['gt']
            if gt is None:
                continue
            try:
                pos_pixel_ratio.append(np.mean(gt))
            except:
                a = 1

        return np.mean(pos_pixel_ratio)

    def GenerateSplit(self, split_filepath=None):
        '''
        Generate data split with specified labeled data percentage.
        :param label_percent:
        :return:
        '''

        if split_filepath is None:
            ## generate random shuffle of whole dataset
            data_index = np.arange(0, self.num_all_data)
            np.random.seed(self.seed_split)
            np.random.shuffle(data_index)

            ## Split Train, Val, Test
            self.num_train = int(np.ceil(self.train_val_test_split[0] / 100 * self.num_all_data))
            self.num_val = int(np.ceil(self.train_val_test_split[1] / 100 * self.num_all_data))
            self.num_test = int(self.num_all_data - self.num_train - self.num_val)

            ## Generate Split
            self.train_index = np.sort(data_index[0:self.num_train])
            self.val_index = np.sort(data_index[self.num_train:self.num_train + self.num_val])
            self.test_index = np.sort(data_index[self.num_train + self.num_val::])

        else:
            fid = open(split_filepath, 'r')
            ## Collect Train index
            line = fid.readline()
            files = line.split(', ')
            self.train_names = []  # training file names list
            for img_name in files:
                img_name = img_name.split('.')[0][1::]
                self.train_names.append(img_name)
            self.train_names.sort()
            # self.train_names = np.sort(np.array(self.train_names)) - 1
            self.num_train = len(self.train_names)

            ## Collect Val index
            line = fid.readline()
            files = line.split(', ')
            self.val_names = []  # validation file names list
            for img_name in files:
                img_name = img_name.split('.')[0][1::]
                self.val_names.append(img_name)
            self.val_names.sort()
            self.num_val = len(self.val_names)

            ## Collect Test index
            line = fid.readline()
            files = line.split(', ')
            self.test_names = []
            for img_name in files:
                img_name = img_name.split('.')[0][1::]
                self.test_names.append(img_name)
            self.test_names.sort()
            self.num_test = len(self.test_names)

        ## Generate Labeled Training Data Index
        self.num_train_labeled = int(np.ceil(self.num_train * self.label_percent))  # number of labeled train data
        self.num_train_unlabeled = self.num_train - self.num_train_labeled  # number of unlabeled train data
        tmp_index = copy.deepcopy(self.train_names)
        np.random.seed(self.seed_label)
        np.random.shuffle(tmp_index)
        self.train_labeled_names = tmp_index[0:self.num_train_labeled]
        self.train_labeled_names_active = self.train_labeled_names.copy()  # index used for slicing training data
        self.train_unlabeled_names = tmp_index[self.num_train_labeled::]
        self.train_unlabeled_names_active = self.train_unlabeled_names.copy()  # index used for slicing training data

        ## Generate Labeled/Unlabeled Training Set batchsize
        #
        #   Generate batchsize proportionally for labeled and unlabeled training data
        #
        self.batch_size_train_labeled = int(np.ceil(self.batch_size * (self.num_train_labeled /
                                                                       (
                                                                                   self.num_train_labeled + self.num_train_unlabeled))))
        self.batch_size_train_unlabeled = self.batch_size - self.batch_size_train_labeled

    def GenerateSplit_EqLabUnlab(self, split_filepath=None, lab_ratio=0.5):
        '''
        Generate data split with specified labeled data percentage.
        :param lab_ratio: the ratio of labeled data in a single mini-batch
        :return:
        '''

        if split_filepath is None:
            ## generate random shuffle of whole dataset
            data_index = np.arange(0, self.num_all_data)
            np.random.seed(self.seed_split)
            np.random.shuffle(data_index)

            ## Split Train, Val, Test
            self.num_train = int(np.ceil(self.train_val_test_split[0] / 100 * self.num_all_data))
            self.num_val = int(np.ceil(self.train_val_test_split[1] / 100 * self.num_all_data))
            self.num_test = int(self.num_all_data - self.num_train - self.num_val)

            ## Generate Split
            self.train_index = np.sort(data_index[0:self.num_train])
            self.val_index = np.sort(data_index[self.num_train:self.num_train + self.num_val])
            self.test_index = np.sort(data_index[self.num_train + self.num_val::])

        else:
            fid = open(split_filepath, 'r')
            ## Collect Train index
            line = fid.readline()
            files = line.split(', ')
            self.train_names = []  # training file names list
            for img_name in files:
                img_name = img_name.split('.')[0][1::]
                self.train_names.append(img_name)
            self.train_names.sort()
            # self.train_names = np.sort(np.array(self.train_names)) - 1
            self.num_train = len(self.train_names)

            ## Collect Val index
            line = fid.readline()
            files = line.split(', ')
            self.val_names = []  # validation file names list
            for img_name in files:
                img_name = img_name.split('.')[0][1::]
                self.val_names.append(img_name)
            self.val_names.sort()
            self.num_val = len(self.val_names)

            ## Collect Test index
            line = fid.readline()
            files = line.split(', ')
            self.test_names = []
            for img_name in files:
                img_name = img_name.split('.')[0][1::]
                self.test_names.append(img_name)
            self.test_names.sort()
            self.num_test = len(self.test_names)

        ## Generate Labeled Training Data Index
        self.num_train_labeled = int(np.ceil(self.num_train * self.label_percent))  # number of labeled train data
        self.num_train_unlabeled = self.num_train - self.num_train_labeled \
                                   + len(
            self.add_train_names)  # number of unlabeled train data (including additional unlabeled data)

        tmp_index = copy.deepcopy(self.train_names)
        np.random.seed(self.seed_label)
        np.random.shuffle(tmp_index)
        self.train_labeled_names = tmp_index[0:self.num_train_labeled]
        self.train_labeled_names_active = self.train_labeled_names.copy()  # index used for slicing training data
        self.train_unlabeled_names = tmp_index[self.num_train_labeled::] + self.add_train_names
        self.train_unlabeled_names_active = self.train_unlabeled_names.copy()  # index used for slicing training data

        ## Generate Labeled/Unlabeled Training Set batchsize
        #
        #   Generate batchsize proportionally for labeled and unlabeled training data
        #
        self.batch_size_train_labeled = int(np.ceil(self.batch_size * lab_ratio))
        self.batch_size_train_unlabeled = self.batch_size - self.batch_size_train_labeled

    def GetDatasetMeanVar(self):
        ## Get dataset mean
        avg_cnt = 0  # average/mean counter
        tmp_allpix = []  # tmp container for all pixels
        tmp_allmask = []  # tmp container for all mask pixels
        for img_name in self.train_names:
            tmp_allpix.append(self.all_data[img_name]['img'].reshape(-1, 3))
            if self.all_data[img_name]['gt'] is not None:
                tmp_allmask.append(self.all_data[img_name]['gt'].reshape(-1))
        tmp_allpix = np.concatenate(tmp_allpix, axis=0)
        self.mean = np.mean(tmp_allpix, axis=0)
        self.stddev = np.std(tmp_allpix, axis=0)
        self.stddev[self.stddev == 0] = 1e-6

        tmp_allmask = np.concatenate(tmp_allmask, axis=0)
        self.mean_pos = np.mean(tmp_allmask)
        ## Overwrite mean and stddev
        # self.mean = np.array([129.58336, 131.12408, 132.89655],dtype=np.float32)
        # self.stddev = np.array([19.735506, 17.652266, 18.440851],dtype=np.float32)

    def InitDataset(self, split_filepath=None):

        self.LoadDataset()
        self.GenerateSplit(split_filepath)
        self.GetDatasetMeanVar()

    def InitDataset_EqLabUnlab(self, split_filepath=None, lab_ratio=1., seed=0):

        self.LoadDataset()
        self.GenerateSplit_EqLabUnlab(split_filepath, lab_ratio)
        self.GetDatasetMeanVar()

    def InitPointer(self):
        '''
        initialize data sample pointer
        :return:
        '''

        self.train_labeled_ptr = 0
        self.train_unlabeled_ptr = 0
        self.val_ptr = 0
        self.test_ptr = 0

    def ShuffleTrainSet(self):
        '''
        Shuffle training set
        :return:
        '''

        # self.train_labeled_index_active = self.train_labeled_index.copy()    # index used for slicing training data
        # self.train_unlabeled_index_active = self.train_unlabeled_index.copy()     # index used for slicing training data
        np.random.shuffle(self.train_labeled_names_active)
        np.random.shuffle(self.train_unlabeled_names_active)

    def InitNewEpoch(self):
        '''
        initialize a new training epoch. First shuffle training set and then initialize data loader
        :return:
        '''
        self.ShuffleTrainSet()
        self.InitPointer()

    def NextTrainBatch_FullSup(self):
        '''
        return the next batch training labeled samples only
        :return:
        '''

        ## Initialize All return variables
        train_data = {'labeled': {'data': None, 'gt': None, 'gt_thin': None, 'name': None},
                      'unlabeled': {'data': None, 'gt': None, 'gt_thin': None, 'name': None}}
        FinishEpoch = False

        ## Check reaching the end of dataset
        start_labeled_train = self.train_labeled_ptr  # start index for labeled train set
        end_labeled_train = self.train_labeled_ptr + self.batch_size_train_labeled  # end index for labeled train set
        if start_labeled_train >= self.num_train_labeled:
            # stop when reaching the end of training set
            FinishEpoch = True
            return FinishEpoch, train_data

        if end_labeled_train > self.num_train_labeled:
            # reached the end of dataset
            index_labeled_train = np.arange(start_labeled_train, self.num_train_labeled)
            index_labeled_train = np.concatenate([index_labeled_train,
                                                  np.random.choice(index_labeled_train,
                                                                   end_labeled_train - self.num_train_labeled)], axis=0)
        else:
            # continue train with new batch of data
            index_labeled_train = np.arange(start_labeled_train, end_labeled_train)

        ## Slice labeled/unlabeled train sample
        # slice labeled train set
        train_labeled_data = np.stack(
            [self.all_data[self.train_labeled_names_active[i]]['img'] for i in index_labeled_train])
        train_labeled_gt = np.stack(
            [self.all_data[self.train_labeled_names_active[i]]['gt'] for i in index_labeled_train])
        train_labeled_gt_thin = np.stack(
            [self.all_data[self.train_labeled_names_active[i]]['gt_thin'] for i in index_labeled_train])
        train_labeled_name = np.stack(
            [self.all_data[self.train_labeled_names_active[i]]['name'] for i in index_labeled_train])

        # train_labeled_gt = self.all_data['gt'][self.train_labeled_index_active[index_labeled_train]]
        # train_labeled_name = self.all_data['name'][self.train_labeled_index_active[index_labeled_train]]
        # slice unlabeled train set
        if self.batch_size_train_unlabeled > 0:
            index_unlabeled_train = np.arange(self.train_unlabeled_ptr,
                                              self.train_unlabeled_ptr + self.batch_size_train_unlabeled)
            index_unlabeled_train = np.mod(index_unlabeled_train,
                                           self.num_train_unlabeled)  # recylce the unlabeled data if reached the end of unlabeled data
            train_unlabeled_data = np.stack(
                [self.all_data[self.train_unlabeled_names_active[i]]['img'] for i in index_unlabeled_train])
            train_unlabeled_gt = np.stack(
                [self.all_data[self.train_unlabeled_names_active[i]]['gt'] for i in index_unlabeled_train])
            train_unlabeled_gt_thin = np.stack(
                [self.all_data[self.train_unlabeled_names_active[i]]['gt_thin'] for i in index_unlabeled_train])
            train_unlabeled_name = np.stack(
                [self.all_data[self.train_unlabeled_names_active[i]]['name'] for i in index_unlabeled_train])
        else:
            train_unlabeled_data = None
            train_unlabeled_gt = None
            train_unlabeled_gt_thin = None
            train_unlabeled_name = None

        ## Update data sample pointer
        self.train_labeled_ptr += self.batch_size_train_labeled
        self.train_unlabeled_ptr += self.batch_size_train_unlabeled

        train_data['labeled']['data'] = train_labeled_data
        train_data['labeled']['gt'] = train_labeled_gt
        train_data['labeled']['gt_thin'] = train_labeled_gt_thin
        train_data['labeled']['name'] = train_labeled_name
        train_data['unlabeled']['data'] = train_unlabeled_data
        train_data['unlabeled']['gt'] = train_unlabeled_gt
        train_data['unlabeled']['gt_thin'] = train_unlabeled_gt_thin
        train_data['unlabeled']['name'] = train_unlabeled_name

        return FinishEpoch, train_data

    def NextTrainBatch(self):
        '''
        return the next batch training samples
        :return:
        '''

        ## Initialize All return variables
        train_data = {'labeled': {'data': None, 'gt': None, 'gt_thin': None, 'name': None},
                      'unlabeled': {'data': None, 'gt': None, 'gt_thin': None, 'name': None}}
        FinishEpoch = False

        ## labeled set index
        start_labeled_train = self.train_labeled_ptr  # start index for labeled train set
        end_labeled_train = self.train_labeled_ptr + self.batch_size_train_labeled  # end index for labeled train set
        index_labeled_train = np.arange(start_labeled_train, end_labeled_train)
        # recylce the unlabeled data if reached the end of unlabeled data
        index_labeled_train = np.mod(index_labeled_train, self.num_train_labeled)

        ## unlabeled set index
        start_unlabeled_train = self.train_unlabeled_ptr  # start index for unlabeled train set
        end_unlabeled_train = self.train_unlabeled_ptr + self.batch_size_train_unlabeled  # end index for unlabeled train set
        index_unlabeled_train = np.arange(start_unlabeled_train, end_unlabeled_train)
        # recylce the unlabeled data if reached the end of unlabeled data
        index_unlabeled_train = np.mod(index_unlabeled_train, self.num_train_unlabeled)

        ## Check reaching the end of train set
        #
        #   reaching the end of train set means both labeled and unlabeled data are iterated at least once.
        if start_labeled_train >= self.num_train_labeled and start_unlabeled_train >= self.num_train_unlabeled:
            FinishEpoch = True
            return FinishEpoch, train_data

        ## Slice labeled/unlabeled train sample
        # slice labeled train set
        train_labeled_data = np.stack(
            [self.all_data[self.train_labeled_names_active[i]]['img'] for i in index_labeled_train])
        train_labeled_gt = np.stack(
            [self.all_data[self.train_labeled_names_active[i]]['gt'] for i in index_labeled_train])
        train_labeled_gt_thin = np.stack(
            [self.all_data[self.train_labeled_names_active[i]]['gt_thin'] for i in index_labeled_train])
        train_labeled_name = np.stack(
            [self.all_data[self.train_labeled_names_active[i]]['name'] for i in index_labeled_train])

        # slice unlabeled train set
        if self.batch_size_train_unlabeled > 0:
            train_unlabeled_data = np.stack(
                [self.all_data[self.train_unlabeled_names_active[i]]['img'] for i in index_unlabeled_train])
            train_unlabeled_gt = np.stack(
                [self.all_data[self.train_unlabeled_names_active[i]]['gt'] for i in index_unlabeled_train])
            # train_unlabeled_gt_thin = np.stack([self.all_data[self.train_unlabeled_names_active[i]]['gt_thin'] for i in index_unlabeled_train])
            # train_unlabeled_gt = None
            train_unlabeled_gt_thin = None
            train_unlabeled_name = np.stack(
                [self.all_data[self.train_unlabeled_names_active[i]]['name'] for i in index_unlabeled_train])
        else:
            train_unlabeled_data = None
            train_unlabeled_gt = None
            train_unlabeled_gt_thin = None
            train_unlabeled_name = None

        ## Update data sample pointer
        self.train_labeled_ptr += self.batch_size_train_labeled
        self.train_unlabeled_ptr += self.batch_size_train_unlabeled

        train_data['labeled']['data'] = train_labeled_data
        train_data['labeled']['gt'] = train_labeled_gt
        train_data['labeled']['gt_thin'] = train_labeled_gt_thin
        train_data['labeled']['name'] = train_labeled_name
        train_data['unlabeled']['data'] = train_unlabeled_data
        train_data['unlabeled']['gt'] = train_unlabeled_gt
        train_data['unlabeled']['gt_thin'] = train_unlabeled_gt_thin
        train_data['unlabeled']['name'] = train_unlabeled_name

        return FinishEpoch, train_data

    def NextValBatch(self):
        '''
        return the next batch validation samples
        :return:
        '''

        ## Initialize All return variables
        val_data = {'data': None, 'gt': None}
        data = None
        gt = None
        FinishEpoch = False

        ## Check reaching the end of dataset
        start_val = self.val_ptr  # start index for labeled train set
        end_val = self.val_ptr + self.batch_size  # end index for labeled train set
        if start_val >= self.num_val:
            # stop when reaching the end of training set
            FinishEpoch = True
            return FinishEpoch, val_data

        if end_val > self.num_val:
            # reached the end of dataset
            index_val = np.arange(start_val, self.num_val)
            # index_labeled_train = np.concatenate([index_labeled_train,
            #                                       np.random.choice(index_labeled_train,
            #                                                        end_labeled_train - self.num_train_labeled)], axis=0)
        else:
            # continue val with new batch of data
            index_val = np.arange(start_val, end_val)

        ## Slice val sample
        data = np.stack([self.all_data[self.val_names[i]]['img'] for i in index_val])
        gt = np.stack([self.all_data[self.val_names[i]]['gt'] for i in index_val])
        name = np.stack([self.all_data[self.val_names[i]]['name'] for i in index_val])

        # data = self.all_data['imgs'][self.val_index[index_val]]
        # gt = self.all_data['gt'][self.val_index[index_val]]
        # name = self.all_data['name'][self.val_index[index_val]]

        ## Update data sample pointer
        self.val_ptr += self.batch_size

        val_data['data'] = data
        val_data['gt'] = gt
        val_data['name'] = name

        return FinishEpoch, val_data

    def NextTestBatch(self):
        '''
        return the next batch testing samples
        :return:
        '''

        ## Initialize All return variables
        test_index = None
        test_data = None
        test_gt = None
        FinishEpoch = False

        test_data = {'data': None, 'gt': None, 'gt_org': None, 'gt_thin': None, 'name': None}

        ## Check reaching the end of dataset

        start_test = self.test_ptr  # start index for labeled train set
        end_test = self.test_ptr + self.batch_size  # end index for labeled train set
        if start_test >= self.num_test:
            # stop when reaching the end of training set
            FinishEpoch = True
            return FinishEpoch, test_data

        if end_test > self.num_test:
            # reached the end of dataset
            index_test = np.arange(start_test, self.num_test)
            # index_labeled_train = np.concatenate([index_labeled_train,
            #                                       np.random.choice(index_labeled_train,
            #                                                        end_labeled_train - self.num_train_labeled)], axis=0)
        else:
            # continue test with new batch of data
            index_test = np.arange(start_test, end_test)

        ## Slice test sample
        test_data['data'] = np.stack([self.all_data[self.test_names[i]]['img'] for i in index_test])
        test_data['gt'] = np.stack([self.all_data[self.test_names[i]]['gt'] for i in index_test])
        test_data['gt_thin'] = np.stack([self.all_data[self.test_names[i]]['gt_thin'] for i in index_test])
        test_data['name'] = np.stack([self.all_data[self.test_names[i]]['name'] for i in index_test])

        ## Update data sample pointer
        self.test_ptr += self.batch_size

        return FinishEpoch, test_data

    def LoadDataset_bk(self):
        '''
        load the entire dataset.
        :return:
        '''

        ## Load Images & GT
        self.all_data = {'imgs': [], 'gt': [], 'name': []}
        img_path = os.path.join(self.data_path, 'image')
        gt_path = os.path.join(self.data_path, 'groundTruth')
        for s_i in range(1, 119):
            # collect all 118 labeled samples
            samplename = '{:03d}'.format(s_i)
            img = plt.imread(os.path.join(img_path, '{:03d}.jpg'.format(s_i)))
            gt = scio.loadmat(os.path.join(gt_path, '{:03d}.mat'.format(s_i)))['groundTruth'][0][0][0]
            # shift all gt to be 0 or 1
            gt -= 1
            gt[gt > 1] = 1
            # collect all data
            self.all_data['imgs'].append(img)
            self.all_data['gt'].append(gt)
            self.all_data['name'].append(samplename)

        self.num_all_data = len(self.all_data['imgs'])
        self.all_data['imgs'] = np.array(self.all_data['imgs'])
        self.all_data['gt'] = np.array(self.all_data['gt'])
        self.all_data['name'] = np.array(self.all_data['name'])