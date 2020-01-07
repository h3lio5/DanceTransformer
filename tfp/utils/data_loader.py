import torch.utils.data as data
import torch
import json
import numpy as np
import math
from tfp.config.config import SPLIT_JSON_LOC


class PoseDataset(data.Dataset):
    """
    Dataset for seq2seq model
    """

    def __init__(self, args):
        """
        Parameter:
        frames :  collection of frames sequences
        shape  : (num_collections,sequence_length,dimension)
        """
        self.data = self._get_data(
            args.location, args.seq_len, args.overlap, args.split_size, args.split, args.num_joints)
        # function
        self.checkcomb = self.check_comb()
        self.sequence_length = args.seq_len
        self.source_length = args.source_length
        self.target_length = args.target_length
        assert self.source_length + \
            self.target_length != self.sequence_length, "Source length and target length don't sum upto sequence length"

    def ___len__(self):
        """
        returns: total number of collections of frame sequences
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            encoder_input: dance pose sequence input for encoder
            decoder_input: dance pose sequence input for decoder
            decoder_output: dance pose sequence used as target 
        """
        frame_seq = self.data[idx]
        encoder_input = frame_seq[:self.source_length, :]
        decoder_input = frame_seq[self.source_length:
                                  self.source_length+self.target_length-1, :]
        target = frame_seq[self.source_length +
                           1:self.source_length+self.target_length, :]
        return torch.FloatTensor(encoder_input), torch.FloatTensor(decoder_input), torch.FloatTensor(target)

    def _get_data(self, folder_location, sequence_length, overlap, split_size, split, num_joints):
        """
        Args:
            folder_location: location of the dataset folder
            sequence_length: total number of dance pose frames in each batch
            overlap: overlap between subsequent dance pose frames
            split_size: percentage of frames in test dataset
            split: train / test data
            num_joints: number of joints in the pose

        Returns:
            data: total number of frame sequences
        """
        # Unique identifier of the dataset
        split_string = 'seq_len_'+str(sequence_length) + "_overlap" + \
            str(overlap) + "_split_size" + str(split_size)
        # Strides
        strides = sequence_length - \
            math.ceil(overlap * sequence_length / 100)
        # All trial file locations
        file_locations = self._get_files_locations(folder_location)
        # combination present in the config file or not
        comb_found = self._check_comb()
        #======== Create train/test data =============#
        if split == 'train':
            if comb_found:
                with open(SPLIT_JSON_LOC, 'r') as jsonfile:
                    data = json.load(jsonfile)
                    train_splits = data[self.split_string]['train_splits']
            else:
                train_splits, test_splits = self._generate_split(
                    file_locations, split_size)
                data = None
                with open(SPLIT_JSON_LOC, "r") as jsonfile:
                    data = json.load(jsonfile)
                with open(SPLIT_JSON_LOC, "w") as jsonfile:
                    data[split_string] = {"train_splits": list(
                        train_splits), "test_splits": list(test_splits)}
                    json.dump(data, jsonfile)
            #
            train_data = []
            for file_name in train_splits:
                file_loc = os.path.join(folder_location, file_name)
                data = np.load(file_loc)
                if data.shape[0] < sequence_length:
                    break
                num_batches = (data.shape[0] - sequence_length)//(strides) + 1
                for i in range(num_batches):
                    train_data.append(
                        data[i * strides: sequence_length + i * strides])

            return np.asarray(train_data).reshape(-1, sequence_length, num_joints*3)

        else:
            if comb_found:
                with open(SPLIT_JSON_LOC, 'r') as jsonfile:
                    data = json.load(jsonfile)
                    test_splits = data[self.split_string]['test_splits']
            else:
                train_splits, test_splits = self._generate_split(
                    file_locations, split_size)
                data = None
                with open(SPLIT_JSON_LOC, "r") as jsonfile:
                    data = json.load(jsonfile)
                with open(SPLIT_JSON_LOC, "w") as jsonfile:
                    data[split_string] = {"train_splits": list(
                        train_splits), "test_splits": list(test_splits)}
                    json.dump(data, jsonfile)
            #
            test_data = []
            for file_name in test_splits:
                file_loc = os.path.join(folder_location, file_name)
                data = np.load(file_loc)
                if data.shape[0] < sequence_length:
                    break
                num_batches = (data.shape[0] - sequence_length)//(strides) + 1
                for i in range(num_batches):
                    test_data.append(
                        data[i * strides: sequence_length + i * strides])

            return np.asarray(test_data).reshape(-1, sequence_length, num_joints*3)

    def _generate_split(self, file_locations, split_size):
        """ function to divide trails into train trials and split trails"""

        num_test_trails = math.floor(len(file_locations) * (split_size)//100)
        np.random.shuffle(file_locations)
        train_split = file_locations[num_test_trails:]
        test_split = file_locations[:num_test_trails]
        return train_split, test_split

    def _get_files_locations(self, folder_location):
        """
        Returns the locations of all the trial data present in our main data folder
        """
        all_numpy_files_loc = [x for x in os.listdir(
            folder_location) if x[-3:] == "npy" or x[-3:] == "npz"]
        return np.asarray(all_numpy_files_loc)

    def _check_comb(self):
        """
        Checks whether json file is present in config folder
        """
        found = False
        file = SPLIT_JSON_LOC
        with open(file) as jsonfile:
            data = json.load(jsonfile)
            if self.split_string in data.keys():
                found = True
            else:
                found = False
        return found
