import os
import argparse
import tfp.config.config as config
import torch
from tfp.utils.transform_data import GetData
from tfp.utils.splitting import Split
from tfp.models.seq2seq import Seq2SeqModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Information")
    parser.add_argument("-category",
                    help="The catergory of data for which you have to train")
    parser.add_argument("-seq_len",
                    help="Sequence length for which you have to train")
    parser.add_argument("-overlap",
                    help="overlap for sequence length for which you have to train")
    args = parser.parse_args()


    data_loc = os.path.join(os.getcwd(),args.category) #transformed data location
    split = Split(location=data_loc, sequence_length = int(args.seq_len), overlap = int(args.overlap))
    test_data = torch.Tensor(split.split_test())
    test_data = test_data.reshape(-1, 100, 63)

    model = Seq2SeqModel(None, 128, num_layers=3, num_joints=21, residual_velocities=True, dropout=0.3, teacher_ratio=0.3)
    model.eval()
    model.load_state_dict(torch.load('saved_models/state_dict.pt'))

    MSE = torch.nn.MSELoss()

    n_warmup = 5
    n_frames = 10

    input1 = test_data[:, :n_warmup, :]
    input2 = test_data[:, n_warmup+n_frames:n_warmup+n_frames+n_warmup, :]
    target1 = test_data[:, n_warmup:n_warmup+n_frames, :]
    target2 = test_data[:, 2*n_warmup+n_frames:2*n_warmup+2*n_frames, :]

    frames1 = model.infer(input1, n_frames)
    frames2 = model.infer(input2, n_frames)

    loss1 = MSE(frames1, target1)
    loss2 = MSE(frames2, target2)

    print(loss1.item())
    print(loss2.item())


