import time
import os

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils import AvgMeter, dice_coeff, check_dir
from dataop import MySet, create_list
from init import InitParser
import UNet


def dice_loss(seg,gt):
    seg = seg.flatten()
    gt = gt.flatten()
    eps = 1e-5
    dice = (2 * (gt * seg).sum())/(gt.sum() + seg.sum()+eps)
    return dice


def test_epoch(net, loader):
    net.eval()
    test_dice_meter = AvgMeter()
    for batch_idx, (data, label) in enumerate(loader):
        data = Variable(data.cuda())
        output = net(data)

        output = output.squeeze().data.cpu().numpy()
        label = label.squeeze().cpu().numpy()

        test_dice_meter.update(dice_coeff(output, label))

        print("Test {} || Dice: {:.4f}".format(str(batch_idx).zfill(4), test_dice_meter.val))
    return test_dice_meter.avg


def train_epoch(net, loader, optimizer, cost, lam):
    net.train()
    
    batch_loss = AvgMeter()
    for batch_idx, (data, label) in enumerate(loader):
        data = Variable(data.cuda())
        label = Variable(label.cuda())

        output = net(data)

        loss = cost(output, label) + lam*dice_loss(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss.update(loss.item())
        print("Train Batch {} || Loss: {:.4f}".format(str(batch_idx).zfill(4), batch_loss.val))
    return batch_loss.avg


def main(args):
    ckpt_path = os.path.join(args.output_path, "Checkpoint")
    log_path = os.path.join(args.output_path, "Log")
    
    check_dir("../output/")
    check_dir(args.output_path)
    check_dir(log_path)
    check_dir(ckpt_path)

    torch.cuda.set_device(args.gpu_id)
    train_list, test_list = create_list(args.data_path, ratio=args.train_ratio)
    
    train_set = MySet(train_list, args.segment)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_set = MySet(test_list, args.segment)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    
    net = UNet.UNet().cuda()
    if args.is_load:
        net.load_state_dict(torch.load(args.load_path))
    
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=args.lr,
    )
    
    cost = torch.nn.BCELoss()
    best_dice = 0.
    for epoch in range(args.init_epoch, args.init_epoch+args.num_epoch):
        start_time = time.time()
        epoch_loss = train_epoch(net, train_loader, optimizer, cost, args.lam)
        epoch_time = time.time() - start_time
        epoch_dice = test_epoch(net, test_loader)

        info_line = "Epoch {} || Loss: {:.4f} | Time: {:.2f} | Test Dice: {:.4f} ".format(
            str(epoch).zfill(3), epoch_loss, epoch_time, epoch_dice
        )
        print(info_line)
        open(os.path.join(log_path, 'train_log.txt'), 'a').write(info_line+'\n')
        
        torch.save(net.state_dict(), os.path.join(ckpt_path, "Network_{}.pth.gz".format(epoch)))
        if epoch_dice > best_dice:
            best_dice = epoch_dice
            torch.save(net.state_dict(), os.path.join(ckpt_path, "Best_Dice.pth.gz"))


if __name__ == '__main__':
    parsers = InitParser()
    main(parsers)
