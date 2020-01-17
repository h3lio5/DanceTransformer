import os
import argparse
import tfp.config.config as config
import torch
from torch.utils.data import DataLoader
from tfp.utils.data_loader import PoseDataset
from tfp.models.seq2seq import Seq2SeqModel
from tqdm import tqdm, trange


parser = argparse.ArgumentParser(description="Training Information")
# parser.add_argument("--category",
#                     help="The catergory of data for which you have to train")
parser.add_argument("--seq_len",
                    help="Sequence length for which you have to train", default=100, type=int)
parser.add_argument("--overlap",
                    help="overlap for sequence length for which you have to train", default=0, type=int)
parser.add_argument(
    "--location", help="location of data set", default="data/salsa")
parser.add_argument("--split_size", help="Test/Train ratio",
                    default=20, type=int)
parser.add_argument(
    "--overlap", help="overlap between subsequent pose sequences", default=0, type=int)
parser.add_argument("--num_joints", help="number of joints",
                    default=21, type=int)
parser.add_argument("--source_length", help="lengt", default=60, type=int)
parser.add_argument("--split",)
parser.add_argument("--transform", default=True, type=str,
                    help="Normalize bone lengths or not?")

args = parser.parse_args()


if __name__ == "__main__":

    # Spliting into train and testdata
    # transformed data location
    posedataset = PoseDataset(args)
    train_loader = DataLoader(posedataset, batch_size=16, num_workers=4)
    model = Seq2SeqModel(None, 128, num_layers=3, num_joints=21,
                         residual_velocities=True, dropout=0.3, teacher_ratio=0.3)
    model = model.cuda()

    EPOCHS = 100
    save_freq = 10
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    MSE = torch.nn.MSELoss()
    avg_losses = []
    for epoch in trange(EPOCHS):
        losses = []
        for encoder_inputs, decoder_inputs, target in tqdm(train_loader):
            opt.zero_grad()
            output = model(encoder_inputs.cuda(), decoder_inputs.cuda())
            loss = MSE(output, target)
            loss.backward()
            opt.step()

            losses.append(loss.item())
        avg_loss = sum(losses)/len(losses)
        avg_losses.append(avg_loss)
        print(f'Epoch: {_} Completed\nLoss: {avg_loss}\n')
        if epoch + 1 % save_freq == 0:
            print("Saving model ")
            torch.save(model.state_dict(),
                       f'saved_models/state_dict_{epoch}.pt')
