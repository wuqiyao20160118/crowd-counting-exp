import torch
from pathlib import Path


def print_state(idx, epoch, size, loss):
    if epoch >= 0:
        message = "Epoch: [{0}][{1}/{2}]\t".format(epoch, idx, size)
    else:
        message = "Val: [{0}/{1}]\t".format(idx, size)

    print(message + '\tloss_cls: {basic_loss:.6f}'.format(
        basic_loss=loss))


def save_checkpoint(state, filename="checkpoint.pth", save_path="weights"):
    # check if the save directory exists
    if not Path(save_path).exists():
        Path(save_path).mkdir()

    save_path = Path(save_path, filename)
    torch.save(state, str(save_path))


def train(model, loss_fn, optimizer, dataloader, epoch, device):
    model = model.to(device)
    model.train()
    for idx, (train_last_transforms, train_present_transforms, last_map, present_map, _, _) in enumerate(dataloader):
        last_img, present_img = train_last_transforms.float().to(device), train_present_transforms.float().to(device)
        last_map, present_map = last_map.float().to(device), present_map.float().to(device)

        prediction1, prediction2 = model(last_img, present_img)
        loss = loss_fn(prediction1, prediction2, last_map, present_map, model.linear_e.weight)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 100)
        optimizer.step()

        print_state(idx, epoch, len(dataloader),
                    loss_fn.MSEloss.average)
