import os
import argparse
import tfp.config.config as config
import torch
from tfp.utils.transform_data import GetData
from tfp.utils.splitting import Split
from tfp.models.seq2seq import Seq2SeqModel


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser(description="Training Information")
parser.add_argument("-category",
                    help="The catergory of data for which you have to train")
parser.add_argument("-seq_len",
                    help="Sequence length for which you have to train")
parser.add_argument("-overlap",
                    help="overlap for sequence length for which you have to train")
parser.add_argument("--first_time",
                    action='store_true',
                    default=False,
                    help="transformed data available or not")

args = parser.parse_args()
def _prepareData(args):
    r"""
        Prepare Data for given catergory
        if --first_time = TRUE then create category folder at root of repo,
        copy transformed file in that folder
    """
    getdata = GetData(config.DATA_LOC,args.category)
    getdata.getdata()




if __name__ == "__main__":
    
    print(os.getcwd())
    if bool(args.first_time):
        _prepareData(args)
        print('Preparing data')

    ## Spliting into train and testdata
    data_loc = os.path.join(os.getcwd(),args.category) #transformed data location
    split = Split(location=data_loc, sequence_length = int(args.seq_len), overlap = int(args.overlap))
    train_data = torch.Tensor(split.split_train()).cuda()

    train_data = train_data.reshape(-1, 100, 63)
    batch_size = 8
    n_batches = train_data.shape[0]//batch_size
    batches = train_data[:n_batches*batch_size, :, :].reshape(n_batches, batch_size, 100, 63)

    model = Seq2SeqModel(None, 128, num_layers=3, num_joints=21, residual_velocities=True, dropout=0.3, teacher_ratio=0.3)
    model=model.cuda()

    EPOCHS = 100
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    MSE = torch.nn.MSELoss()
    avg_losses = []
    for _ in range(EPOCHS):
        losses = []
        for mini_batch in batches:
            opt.zero_grad()

            encoder_inputs = mini_batch[:, :60, :]
            decoder_inputs = mini_batch[:, 60:-1, :]
            target = mini_batch[:, 61:, :]
    
            output = model.forward(encoder_inputs, decoder_inputs)
            loss = MSE(output, target)
            loss.backward()
            opt.step()

            losses.append(loss.item())
        avg_loss = sum(losses)/len(losses)
        avg_losses.append(avg_loss)
        print(f'Epoch: {_} Completed\nLoss: {avg_loss}\n')
        torch.save(model.state_dict(), 'saved_models/state_dict.pt')
