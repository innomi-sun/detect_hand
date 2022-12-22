from datetime import datetime
import argparse
import torch, os
import torch.nn as nn
import torch.nn.init as init
from torchvision import models, transforms

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datasets.datasets import LandmarkDiffDataset

# tensorboard --logdir F:\repositories\misc\tensorboard-logs\detect_hand\
# python detect/train.py --n_epochs 50 --pretrained

parser = argparse.ArgumentParser()
parser.add_argument("--num_class", type=int, default=21, help="number of epochs of training")
parser.add_argument("--input_size", type=int, default=21, help="number of epochs of training")
parser.add_argument("--pretrained", action='store_true', help="Determine whether to use a pre-trained model")
parser.add_argument("--train_root", type=str, default='E:/datasets/my_hand_pose_dataset/train', help="logs dir")
parser.add_argument("--val_root", type=str, default='E:/datasets/my_hand_pose_dataset/val', help="logs dir")

parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1080, help="size of the batches")
# parser.add_argument("--batch_size_val", type=int, default=4000, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--logs_dir", type=str, default='F:/repositories/misc/tensorboard-logs/detect_hand', help="logs dir")

opt = parser.parse_args()
print(opt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

def main(): 

    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_datasets = LandmarkDiffDataset(opt.train_root, num_class=opt.num_class, 
        input_width=opt.input_size, input_height=opt.input_size, transform=t)
    train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=opt.batch_size, shuffle=True)
    
    val_datasets = LandmarkDiffDataset(opt.val_root, num_class=opt.num_class, 
        input_width=opt.input_size, input_height=opt.input_size, transform=t)
    val_loader = torch.utils.data.DataLoader(dataset=val_datasets, batch_size=opt.batch_size, shuffle=False)

    save_tag = 'resnet18'

    train_data_len = train_datasets.__len__()
    val_total = val_datasets.__len__()
    steps = train_data_len // opt.batch_size + 1
    logs_dir = os.path.join(opt.logs_dir, datetime.now().strftime("%Y%m%d-%H%M%S") + save_tag)
    # if opt.resume:
    # init model
    model = models.resnet18(pretrained=opt.pretrained)
    #model = models.resnet50(pretrained=opt.pretrained)
    # model = models.resnet101(pretrained=opt.pretrained)
    if opt.pretrained:
        # 除了bn层，将所有的参数层进行冻结
        for name, param in model.named_parameters():
            if "bn" not in name:
                param.requires_grad = False
    else:
        init_weights(model.modules())

    # 输入channel,width,height不同, 需要重新定义conv1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=2, stride=1)

    # 定义一个新的FC层
    num_fc_ftr = model.fc.in_features
    model.fc = nn.Linear(num_fc_ftr, opt.num_class)
    model = model.to(device)

    # ----------
    #  Training
    # ----------
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    for epoch in range(opt.n_epochs):

        model.train()
        for i, (data, target) in enumerate(train_loader):

            data = data.to(device)
            target = target.to(device)

            # Forward pass
            target_hat = model(data)
            loss = criterion(target_hat, target)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print('Epoch [{:>3}/{:>3}]  Batch [{:>3}/{:>3}]  Loss: {:.10f}'.format(epoch + 1, opt.n_epochs, i + 1, steps, loss.item()), end='')
        draw_chart('01_loss', loss.item(), steps * epoch + i + 1, logs_dir)

        val_loss, correct = 0, 0
        model.eval()
        with torch.no_grad():
            
            for v_i, (v_data, v_target) in enumerate(val_loader):          

                v_data = v_data.to(device)
                v_target = v_target.to(device)
                
                # optimizer.zero_grad()
                v_target_hat = model(v_data)
                 # sum up batch loss
                val_loss += criterion(v_target_hat, v_target).item()

                pred = v_target_hat.max(1, keepdim=True)[1]
                labels = v_target.max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()

        acc_rate = round(100. * (correct / val_total), 1)
        # print(' +++++++++++++++++++++++++++++++++++ ')
        # print(' +++++  Accuracy  {:>6}/{}  +++++'.format(correct, val_total))
        # print(' +++++  Acc Rate       {:>5}%  +++++'.format(str(acc_rate)))
        # print(' +++++  Val loss      {:.5f}  +++++'.format((val_loss / steps)))
        # print(' +++++++++++++++++++++++++++++++++++ ')
        
        print('    -->  Accuracy [{:>6}/{}], Acc Rate [{:>5}%], Val loss: [{:.5f}]'.format(correct, val_total, str(acc_rate), val_loss / steps))
        draw_chart('02_acc_rate(%)', acc_rate, epoch + 1, logs_dir)
        # draw_chart('03_refund_rate(%)', refund_rate, epoch + 1, logs_dir)

    checkpoin_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'checkpoint')
    Path(checkpoin_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_filename = "ckp_{}_{}_{}.pth".format(datetime.now().strftime("%Y_%m_%d_%H%M"), save_tag, str(acc_rate))
    torch.save(model, os.path.join(checkpoin_dir, checkpoint_filename))

    print(" checkpoint file {} saved.".format(checkpoint_filename))

def draw_chart(name, value, steps, logs_dir):

    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=logs_dir, flush_secs=2, comment='train') as writer:
        
        # writer.add_histogram('his/loss', loss_value, epoch)
        writer.add_scalar(name, value, steps)

        #writer.add_histogram('his/y', y, epoch)
        #writer.add_scalar('data/y', y, epoch)
        #writer.add_scalar('data/loss', loss, epoch)
        #writer.add_scalars('data/data_group', {'x': x, 'y': y}, epoch)

if __name__ == '__main__':
    main()
