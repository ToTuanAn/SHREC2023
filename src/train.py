from model.model import SiameseNetwork
from loss import ContrastiveLoss
import torch

# Declare Siamese Network
net = SiameseNetwork().cuda()
# Decalre Loss Function
criterion = ContrastiveLoss()
# Declare Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=0.0005)
#train the model

def train():
    loss=[]
    counter=[]
    iteration_number = 0
    for epoch in range(1,config.epochs):
        for i, data in enumerate(train_dataloader,0):
            img0, img1, label = data
            img0, img1, label = img0.cuda(), img1.cuda() , label.cuda()
            optimizer.zero_grad()
            output1, output2 = net(img0,img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
        print("Epoch {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
        iteration_number += 10
        counter.append(iteration_number)
        loss.append(loss_contrastive.item())
    #show_plot(counter, loss)
    return net
