import torch.utils.data as data
import torch


class PoseDataset(data.Dataset):
    """

    """

    def __init__(self, args):
        """
        Parameter:
        frames :  collection of frames sequences
        shape  : (num_collections,sequence_length,dimension)
        """
        self.data = self._get_data(args.location, args.seq_len, args.overlap,)
        # function
        self._file_locations = self.get_files_locations()
        self.checkcomb = self.check_comb()
        self.data = frames

    def ___len__(self):
        """
        returns: total number of collections of frame sequences
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        returns: sequence of frames as FloatTensor object
        """
        frame_seq = self.data[idx]
        return torch.FloatTensor(frame_seq)

    def _get_data(self, args):
        """
        Returns
        """


class Split:
    r"""
            This class generate Training data using the folder created by getData class

            returns an array of shape (None, seg_len, num_joints, 3)
    """

    def __init__(self, location=None, sequence_length=100, overlap=0, split_size=20):
            # pramaters
        self.folder_location = location
        self.split_size = split_size
        self.overlap = overlap
        self.seq_len = sequence_length
        self.split_string = str(sequence_length) + "_" + \
            str(overlap) + "_" + str(split_size)
        self.strides = self.seq_len - \
            math.ceil(self.overlap * self.seq_len / 100)
        # function
        self.file_locations = self.get_files_locations()
        self.checkcomb = self.check_comb()

    def check_comb(self):
        # json file should be present in config folder
        found = False
        file = SPLIT_JSON_LOC
        with open(file) as jsonfile:
            data = json.load(jsonfile)
            if self.split_string in data.keys():
                found = True
            else:
                found = False
        return found

    def get_files_locations(self):
        all_numpy_files_loc = [x for x in os.listdir(
            self.folder_location) if x[-3:] == "npy" or x[-3:] == "npz"]
        return np.asarray(all_numpy_files_loc)

    def gen_split(self):
        """ function to divide trials into train trials and split trails"""

        files = self.file_locations

        num_test_trails = math.floor(len(files) * (self.split_size)//100)
        np.random.shuffle(files)
        train_trails = files[num_test_trails:]
        test_trails = files[:num_test_trails]

        return train_trails, test_trails

    def split_train(self):
        if self.checkcomb:
            with open(SPLIT_JSON_LOC, 'r') as jsonfile:
                data = json.load(jsonfile)
                train_splits = data[self.split_string]['train_splits']
                test_splits = data[self.split_string]['test_splits']
        else:
            train_splits, test_splits = self.gen_split()
            data = None
            with open(SPLIT_JSON_LOC, "r") as jsonfile:
                data = json.load(jsonfile)
            with open(SPLIT_JSON_LOC, "w") as jsonfile:
                data[self.split_string] = {"train_splits": list(
                    train_splits), "test_splits": list(test_splits)}
                json.dump(data, jsonfile)

        comp_data = []
        for _file in train_splits:
            loc = os.path.join(self.folder_location, _file)
            data = np.load(loc)
            if data.shape[0] < self.seq_len:
                break

            num_bat = (data.shape[0] - self.seq_len)//(self.strides) + 1
            print(data.shape)
            print(num_bat)
            for i in range(num_bat):
                comp_data.append(
                    data[i * self.strides: self.seq_len + i * self.strides])

        return np.asarray(comp_data)

    def split_test(self):
        if self.checkcomb:
            with open(SPLIT_JSON_LOC, 'r') as jsonfile:
                data = json.load(jsonfile)
                train_splits = data[self.split_string]['train_splits']
                test_splits = data[self.split_string]['test_splits']
        else:
            train_splits, test_splits = self.gen_split()
            data = None
            with open(SPLIT_JSON_LOC, "r") as jsonfile:
                data = json.load(jsonfile)
            with open(SPLIT_JSON_LOC, "w") as jsonfile:
                data[self.split_string] = {"train_splits": list(
                    train_splits), "test_splits": list(test_splits)}
                json.dump(data, jsonfile)

        comp_data = []
        for file in test_splits:
            loc = os.path.join(self.folder_location, file)
            data = np.load(loc)
            if data.shape[0] < self.seq_len:
                break

            num_bat = (data.shape[0] - self.seq_len)//(self.strides) + 1
            for i in range(num_bat):
                comp_data.append(
                    data[i * self.strides: self.seq_len + i * self.strides])

            return np.asarray(comp_data)
