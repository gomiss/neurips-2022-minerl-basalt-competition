import torch as th
from tqdm import tqdm
from torchvision import models
import torch.nn as nn
from torch.nn.modules.loss import BCEWithLogitsLoss
import os,sys
sys.path.append(os.path.dirname(__file__))
os.environ['TORCH_HOME'] = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

from dataloader import PicsDataLoader


def train_flatten_area():
    trainloader, testloader = PicsDataLoader(labels=[0,1], size=4000, batch_size=32)


    def make_train_step(model, optimizer, loss_fn):
        def train_step(x,y):
            #make prediction
            yhat = model(x)
            #enter train mode
            model.train()
            #compute loss
            loss = loss_fn(yhat,y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #optimizer.cleargrads()

            return loss
        return train_step

    device = "cuda" if th.cuda.is_available() else "cpu"
    model = th.load(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'hub', 'checkpoints', 'resnet50-0676ba61.pth'))

    #freeze all params
    for params in model.parameters():
        params.requires_grad_ = False


    # #add a new final layer
    # nr_filters = model.fc.in_features  #number of input features of last layer
    # model.fc = nn.Linear(nr_filters, 1)
    #add a new final layer
    nr_filters = model.classifier[1].in_features  #number of input features of last layer
    model.classifier[1] = nn.Linear(nr_filters, 1)

    model = model.to(device)


    #loss
    loss_fn = BCEWithLogitsLoss() #binary cross entropy with sigmoid, so no need to use sigmoid in the model

    #optimizer
    optimizer = th.optim.Adam(model.classifier[1].parameters()) 

    #train step
    train_step = make_train_step(model, optimizer, loss_fn)



    losses = []
    val_losses = []

    epoch_train_losses = []
    epoch_test_losses = []

    n_epochs = 100
    early_stopping_tolerance = 3
    early_stopping_threshold = 0.03



    for epoch in range(n_epochs):
        epoch_loss = 0
        for i ,data in tqdm(enumerate(trainloader), total = len(trainloader)): #iterate ove batches
            x_batch , y_batch = data
            x_batch = x_batch.to(device) #move to gpu
            y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
            y_batch = y_batch.to(device) #move to gpu


            loss = train_step(x_batch, y_batch)
            epoch_loss += loss/len(trainloader)
            losses.append(loss)
        
        epoch_train_losses.append(epoch_loss)
        print('\nEpoch : {}, train loss : {}'.format(epoch+1,epoch_loss))

        #validation doesnt requires gradient
        with th.no_grad():
            cum_loss = 0
            for x_batch, y_batch in testloader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
                y_batch = y_batch.to(device)

                #model to eval mode
                model.eval()

                yhat = model(x_batch)
                val_loss = loss_fn(yhat,y_batch)
                cum_loss += loss/len(testloader)
                val_losses.append(val_loss.item())


            epoch_test_losses.append(cum_loss)
            print('Epoch : {}, val loss : {}'.format(epoch+1,cum_loss))  
            
            best_loss = min(epoch_test_losses)
            
            #save best model
            if cum_loss <= best_loss:
                best_model_wts = model.state_dict()
            
            #early stopping
            early_stopping_counter = 0
            if cum_loss > best_loss:
                early_stopping_counter +=1

            if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
                print("/nTerminating: early stopping")
                break #terminate training
        
    #load best model
    model.load_state_dict(best_model_wts)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'train', 'flattenareaV5mobilenet.pt')
    th.save(model, output_dir)

if __name__ == '__main__':
    train_flatten_area()