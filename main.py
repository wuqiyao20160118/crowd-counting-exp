from siamese_model import CoattentionNet
from evaluate_model import evaluate_model
from loss import SiameseCriterion
import trainer
from torchvision import transforms
from torch.utils import data
from dataloader_scvd import TrainDataLoader, ValDataLoader
import torch
from torch import optim
import argparse
import os
import os.path as osp


try:
    from termcolor import cprint
except ImportError:
    cprint = None


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("traindata")
    parser.add_argument("valdata")
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--dataset", default="scvd")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.0005, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--save-every", default=1, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)


def get_dataloader(datapath, args, img_transforms=None,
                   train=True, split="train"):
    b_size = args.batch_size if split == "train" else args.val_batch_size
    if split == "train":
        datas = TrainDataLoader(img_transforms, img_transforms, root_dir=datapath, training=train)
    else:
        datas = ValDataLoader(img_transforms, img_transforms, root_dir=datapath, training=train)
    data_loader = data.DataLoader(datas, batch_size=b_size, shuffle=train, num_workers=args.workers, pin_memory=True)
    return data_loader


def val_dataloader(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    val_loader = get_dataloader(args.valdata, args, train=False, split="val", img_transforms=val_transforms)
    return val_loader


def get_model(checkpoint=None):
    model = CoattentionNet()
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["model"])
    return model


def main():
    args = arguments()

    # check for CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_loader = get_dataloader(args.traindata, args, img_transforms=img_transforms)

    model = CoattentionNet()
    loss_fn = SiameseCriterion(device=device)

    # directory where we'll store model weights
    weights_dir = "weight"
    if not osp.exists(weights_dir):
        os.mkdir(weights_dir)

    optimizer = optim.SGD(model.learnable_parameters(args.lr), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        checkpoint = torch.load(args.resume)
        model = model.to(device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Set the start epoch if it has not been
        if not args.start_epoch:
            args.start_epoch = checkpoint['epoch']

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=20,
                                          last_epoch=args.start_epoch - 1)
    print("Start training!")

    # train and evalute for `epochs`
    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        trainer.train(model, loss_fn, optimizer, train_loader, epoch, device=device)

        if (epoch + 1) % args.save_every == 0:
            trainer.save_checkpoint({
                'epoch': epoch + 1,
                'batch_size': train_loader.batch_size,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, filename="checkpoint_{0}.pth".format(epoch + 1), save_path=weights_dir)

        if epoch != 0 and epoch % 5 == 0:
            val_loader = val_dataloader(args)
            with torch.no_grad():
                mae, mse = evaluate_model(model, val_loader, device=device, training=True, debug=args.debug)
                log_text = 'epoch: %4d, mae: %4.2f, mse: %4.2f' % (epoch, mae, mse)
                log_print(log_text, color='green', attrs=['bold'])


if __name__ == '__main__':
    main()

