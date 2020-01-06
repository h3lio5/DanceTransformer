import os
import argparse
import tfp.config.config as config
from tfp.utils.transform_data import GetData
from tfp.utils.splitting import Split


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


parser = argparse.ArgumentParser(description="Training Information")
parser.add_argument("category",
                    help="The catergory of data for which you have to train")
parser.add_argument("--seq_len",
                    help="Sequence length for which you have to train")
parser.add_argument("--overlap",
                    help="overlap for sequence length for which you have to train")
parser.add_argument("--first_time",
                    type=str_to_bool,
                    nargs='?',
                    default=True,
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

    ## Spliting into train and testdata
    data_loc = os.path.join(os.getcwd(),args.category) #transformed data location
    split = Split(location=data_loc, sequence_length = int(args.seq_len), overlap = int(args.overlap))
    train_data = split.split_test()
    print(train_data.shape)
