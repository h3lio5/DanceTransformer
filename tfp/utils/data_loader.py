import torch.utils.data as data
import torch



class MyDataset(data.Dataset):
    """

    """
    def __init__(self,frames):
        """
        Parameter:
        frames :  collection of frames sequences
        shape  : (num_collections,sequence_length,dimension)
        """
        self.data = frames

    def ___len__(self) :
        """
        returns: total number of collections of frame sequences
        """
        return len(self.data)

    def __getitem__(self,idx):
        """
        returns: sequence of frames as FloatTensor object
        """
        frame_seq = self.data[idx]
        return torch.FloatTensor(frame_seq)
