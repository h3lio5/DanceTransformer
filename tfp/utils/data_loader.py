import torch.utils.data as data
import torch


class PoseDataset(data.Dataset):
    """

    """

    def __init__(self, split="train"):
        """
        Parameter:
        frames :  collection of frames sequences
        shape  : (num_collections,sequence_length,dimension)
        """
        if split == 'test':
            self.data =
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
