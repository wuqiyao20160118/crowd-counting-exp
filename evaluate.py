from siamese_model import CoattentionNet
from evaluate_model import evaluate_model
from torch.utils import data
from dataloader_scvd import TestDataLoader
import torch
import argparse
import os


try:
    from termcolor import cprint
except ImportError:
    cprint = None


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("testdata")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--dataset", default="scvd")
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--checkpoint",
                        help="The path to the model checkpoint", default="")
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


def get_dataloader(datapath, args, device,
                   train=False):
    b_size = args.val_batch_size
    datas = TestDataLoader(device=device, data_path=datapath, training=train)
    data_loader = data.DataLoader(datas, batch_size=b_size, shuffle=train, num_workers=args.workers, pin_memory=True)
    return data_loader


def get_testdata(args, device):
    test_loader = get_dataloader(args.testdata, args, train=False, device=device)
    return test_loader


def get_model(checkpoint):
    model = CoattentionNet()
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["model"])
    return model


def main():
    args = arguments()
    model_name = "siamese"

    # check for CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda:2')
    else:
        device = torch.device('cpu')

    test_dataloader = get_testdata(args, device=device)

    model_dir = args.checkpoint

    print("Start testing!")

    mae, mse = evaluate_model(model_dir, test_dataloader, device=device, training=False, debug=False, test=True, segmentation=False)
    log_text = 'mae: %4.2f, mse: %4.2f' % (mae, mse)
    log_print(log_text, color='green', attrs=['bold'])
    file_results = os.path.join('./', 'results_' + model_name + '.txt')
    f = open(file_results, 'w')
    f.write('MAE: %0.2f, MSE: %0.2f' % (mae, mse))
    f.close()


if __name__ == "__main__":
    main()
