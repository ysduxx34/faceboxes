import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable

from networks import PedestrainBox
from multibox_loss import MultiBoxLoss
from dataset import ListDataset

import visdom
import numpy as np
import torch.backends.cudnn as cudnn
import math

use_gpu = torch.cuda.is_available()
file_root = ''

learning_rate = 0.001
num_epochs = 300
batch_size = 64

net = PedestrainBox()
# if use_gpu:
#     net.cuda()

# TODO: use two gpu parallel
device = 'cuda'
if device == 'cuda':
    net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

print('load model...')
# net.load_state_dict(torch.load('weight/pedestrainboxes_2.76.pt'))

criterion = MultiBoxLoss()

# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0003)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)

train_dataset = ListDataset(root=file_root,list_file='label/box_label_train.txt',train=True,transform = [transforms.ToTensor()] )
train_loader  = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=5)
print('the dataset has %d images' % (len(train_dataset)))
print('the batch_size is %d' % (batch_size))

val_dataset = ListDataset(root=file_root,list_file='label/box_label_val.txt',train=False,transform = [transforms.ToTensor()] )
val_loader  = DataLoader(val_dataset,batch_size=batch_size,shuffle=True,num_workers=5)
print('the dataset has %d images' % (len(val_dataset)))
print('the batch_size is %d' % (batch_size))

num_iter = 0
vis = visdom.Visdom()
win = vis.line(Y=np.array([0]), X=np.array([0]))
min_loss = 10.

for epoch in range(num_epochs):
    if epoch == 100 or epoch == 200:
        learning_rate *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    
    print('\n\nStarting epoch %d / %d' % (epoch + 1, num_epochs))
    print('Learning Rate for this epoch: {}'.format(learning_rate))
    
    total_loss = 0.

    #train
    net.train()
    for i,(images,loc_targets,conf_targets) in enumerate(train_loader):
        images = Variable(images)
        loc_targets  = Variable(loc_targets)
        conf_targets = Variable(conf_targets)
        if use_gpu:
            images,loc_targets,conf_targets = images.cuda(),loc_targets.cuda(),conf_targets.cuda()
        
        loc_preds, conf_preds = net(images)
        loss = criterion(loc_preds,loc_targets,conf_preds,conf_targets)
        total_loss += loss.data[0]
        
        # print(('total_loss:  %.4f' %total_loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 5 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, average_loss: %.4f' 
            %(epoch+1, num_epochs, i+1, len(train_loader), loss.data[0], total_loss / (i+1)))
            num_iter = num_iter + 1
            vis.line(Y=np.array([total_loss / (i+1)]), X=np.array([num_iter]), 
                    win=win,
                    update='append')
    #val
    net.eval()
    total_loss_val = 0.
    count = 0
    for i,(images,loc_targets,conf_targets) in enumerate(val_loader):
        images = Variable(images)
        loc_targets  = Variable(loc_targets)
        conf_targets = Variable(conf_targets)
        if use_gpu:
            images,loc_targets,conf_targets = images.cuda(),loc_targets.cuda(),conf_targets.cuda()
    
        loc_preds, conf_preds = net(images)
        loss = criterion(loc_preds,loc_targets,conf_preds,conf_targets)

        if math.isinf(loss.data[0]) or math.isnan(loss.data[0]) :
            print ("loss data is inf or nan")
        else :
            total_loss_val += loss.data[0]
            count = count + 1
            print ('loss.data: %.4f'  %(loss.data[0]))

    print ('count of val:  '+str(count))
    average_loss = total_loss_val / count
    print ('average_loss: %.4f' 
         %(average_loss))
    if average_loss < min_loss :
        min_loss = average_loss
        if not os.path.exists('weight/'):
            os.mkdir('weight')  
        print('saving model ...')  
        torch.save(net.state_dict(),'weight/pedestrainboxes.pt')
    

