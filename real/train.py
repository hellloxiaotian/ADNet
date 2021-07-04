import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import  time #tcw20182159tcw
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
from torch.nn.modules.loss import _Loss #TCW20180913TCW
from models import ADNet
from dataset import prepare_data, Dataset
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="ADNet")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=15, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=15, help='noise level used on validation set')
'''
parser.add_argument("--clip",type=float,default=0.005,help='Clipping Gradients. Default=0.4') #tcw201809131446tcw
parser.add_argument("--momentum",default=0.9,type='float',help = 'Momentum, Default:0.9') #tcw201809131447tcw
parser.add_argument("--weight-decay","-wd",default=1e-3,type=float,help='Weight decay, Default:1e-4') #tcw20180913347tcw
'''
opt = parser.parse_args()
class sum_squared_error(_Loss):  # PyTorch 0.4.1
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """
    def __init__(self, size_average=None, reduce=None, reduction='sum'):
        super(sum_squared_error, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        # return torch.sum(torch.pow(input-target,2), (0,1,2,3)).div_(2)
        return torch.nn.functional.mse_loss(input, target, size_average=None, reduce=None, reduction='sum').div_(2)

def main():
    # Load dataset
    t1 = time.clock()
    global data_label
    save_dir = opt.outf + 'sigma' + str(opt.noiseL) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=False)
    loader_val = DataLoader(dataset=dataset_val, num_workers=4, batch_size=opt.batchSize, shuffle=False)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = ADNet(channels=3, num_of_layers=opt.num_of_layers)
    #net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    #criterion1 = nn.SmoothL1Loss(size_average=False) #tcw201810192202tcw 
    #criterion = sum_squared_error() #tcw20180913211tcw
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda() #tcw201810192202tcw 
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    #writer = SummaryWriter(opt.outf)
    #step = 0
    noiseL_B=[0,55] # ingnored when opt.mode=='S'
    psnr_list = [] #201809062254tcw. it is used to save the psnr value of each epoch
    for epoch in range(opt.epochs):
        if epoch <= opt.milestone:
            current_lr = opt.lr
        if epoch > opt.milestone and  epoch <=60:
            current_lr  =  opt.lr/10. 
        if epoch > 60  and  epoch <=90:
            current_lr = opt.lr/100.
        if epoch > 90:
            current_lr = opt.lr/1000.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            #print 'a'
            for j, data1 in enumerate(loader_val,0):
                 if i== j:
                    data_label = data1
                    break
            #print 'b'         
            # training step
            model.train()
            #print 'c'
            #model.zero_grad()
            #model.zero_grad()
            #optimizer.zero_grad()
            #criterion = criterion2
            img_train = data
            img_label = data_label
            #print img_train.size()
            #print img_label.size()
            '''
            if opt.mode == 'S':
                 noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
                 #noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=1)*opt.noiseL/255.
            if opt.mode == 'B':
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.) #tcw20180913tcw
		    #noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0,std=1)*stdN[n]/255.
            imgn_train = img_train + noise
            '''
            '''
            img_train, imgn_train = Variable(img_train), Variable(imgn_train) #tcw201809131428tcw
	    noise = Variable(noise,requires_grad=False) #tcw201809131430tcw
	    img_train = img_train.cuda() #tcw201809131432
	    imgn_train = imgn_train.cuda() #tcw201809131432
            '''
            img_label, img_train = Variable(img_label.cuda()), Variable(img_train.cuda()) #tcw201809131425tcw
            #noise = Variable(noise.cuda())  #tcw201809131425tcw
 	    #img_train = img_train.cuda()
            #imgn_train = imgn_train.cuda()
            #noise = noise.cuda()
            out_train = model(img_train)
            #loss = criterion(out_train, noise) / (imgn_train.size()[0]*2) #tcw201809182256tcw
            loss =  criterion(out_train, img_label) / (img_train.size()[0]*2)
            optimizer.zero_grad() #tcw201809112015tcw
            loss.backward()
            optimizer.step()
            model.eval()
            # results
            #model.eval() #tcw20180915tcw
            #out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.) #tcw201809182304tcw
            out_train = torch.clamp(model(img_train), 0., 1.) 
            psnr_train = batch_PSNR(out_train, img_label, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            '''            
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
            '''
        ## the end of each epoch
        model.eval() #tcw20180915tcw
        model_name = 'model'+ '_' + str(epoch+1) + '.pth' #tcw201809071117tcw
        torch.save(model.state_dict(), os.path.join(save_dir, model_name)) #tcw201809062210tcw
        t2 = time.clock()
        t3 = t2-t1
        print t3
        '''
        for param in model.parameters():
            param.requires_grad = False
        '''
        # validate
        '''
        psnr_val = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            torch.manual_seed(0) #set the seed,tcw201809030915 
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
	    #noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=1)*opt.val_noiseL/255.
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda(),requires_grad=False)
            #print 'a'
            #out_val = torch.clamp(imgn_val-model(imgn_val), 0., 1.) #tcw201809182305tcw
            out_val = torch.clamp(model(imgn_val), 0., 1.)
            #print 'b'
            psnr_val += batch_PSNR(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        psnr_val1 = str(psnr_val) #tcw201809071251tcw
        psnr_list.append(psnr_val1) #tcw201809071103tcw
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        #writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        # log the images
        #out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
        # save model
        #torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))
        model_name = 'model'+ '_' + str(epoch+1) + '.pth' #tcw201809071117tcw
        torch.save(model.state_dict(), os.path.join(save_dir, model_name)) #tcw201809062210tcw
    filename = save_dir + 'psnr.txt' #tcw201809071117tcw
    f = open(filename,'w') #201809071117tcw
    for line in psnr_list:  #201809071117tcw
        f.write(line+'\n') #2018090711117tcw
    f.close()
    '''

if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=50, stride=40, aug_times=1) #tcw201810102244
            #prepare_data(data_path='data', patch_size=50, stride=40, aug_times=1) #tcw201810102244
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
    main()
