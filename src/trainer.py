import torch
import argparse
import torchaudio
import os
import numpy as np

from tensorboardX import SummaryWriter

from model.Model import Model
from Dataset import Dataset

from utils.hparams import HParam
from utils.writer import MyWriter

def spec_to_wav(complex_ri, window, length):
    audio = torch.istft(input= complex_ri, n_fft=int(1024), hop_length=int(256), win_length=int(1024), window=window, center=True, normalized=False, onesided=True, length=length)
    return audio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--step','-s',type=int,required=False,default=0)
    args = parser.parse_args()

    hp = HParam(args.config)
    print("NOTE::Loading configuration : "+args.config)

    device = hp.gpu
    torch.cuda.set_device(device)

    batch_size = hp.train.batch_size
    block = hp.model.Model.block
    num_epochs = hp.train.epoch
    num_workers = hp.train.num_workers

    window = torch.hann_window(window_length=hp.audio.frame, periodic=True,
                               dtype=None, layout=torch.strided, device=None,
                               requires_grad=False).to(device)

    best_loss = 10

    ## load

    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + args.version_name
    log_dir = hp.log.root+'/'+'log'+'/'+args.version_name

    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)

    writer = MyWriter(hp, log_dir)

    ## target

    ## TODO
    list_train= ['','']
    list_test= ['','']

    # TODO
    train_dataset = DatasetModel(hp.data.root+'/STFT',list_train,'*.npy',block=block)
    test_dataset= DatasetModel(hp.data.root+'/STFT',list_test,'*.npy',block=block)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    # TODO
    model = ModelModel(hp).to(device)

    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        model.load_state_dict(torch.load(args.chkpt, map_location=device))

    # TODO
    criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.train.adam)

    if hp.scheduler.type == 'Plateau': 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            mode=hp.scheduler.Plateau.mode,
            factor=hp.scheduler.Plateau.factor,
            patience=hp.scheduler.Plateau.patience,
            min_lr=hp.scheduler.Plateau.min_lr)
    elif hp.scheduler.type == 'oneCycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                max_lr = hp.scheduler.oneCycle.max_lr,
                epochs=hp.train.epoch,
                steps_per_epoch = len(train_loader)
                )
    else :
        raise Exception("Unsupported sceduler type")

    step = args.step

    for epoch in range(num_epochs):
        ### TRAIN ####
        model.train()
        train_loss=0
        for i, (batch_data) in enumerate(train_loader):
            step +=1
            
            # TODO
            input = batch_data[''].to(device)
            target = batch_data[''].to(device)
            output = model(input)

            loss = criterion(output,target).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('TRAIN::Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
            train_loss+=loss.item()

            if step %  hp.train.summary_interval == 0:
                writer.log_value(loss,step,'train loss : '+hp.loss.type)

        train_loss = train_loss/len(train_loader)
        torch.save(model.state_dict(), str(modelsave_path)+'/lastmodel.pt')
            
        #### EVAL ####
        model.eval()
        with torch.no_grad():
            test_loss =0.
            for j, (batch_data) in enumerate(test_loader):
                # TODO
                input = batch_data['input'].to(device)
                target = batch_data['target'].to(device)
                output = model(input)

                loss = criterion(output,target).to(device)

                print('TEST::Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, j+1, len(test_loader), loss.item()))
                test_loss +=loss.item()

            test_loss = test_loss/len(test_loader)
            if hp.scheduler.type == 'Plateau':
                scheduler.step(test_loss)
            else :
                scheduler.step(test_loss)
            
            writer.log_value(test_loss,step,'test lost : ' + hp.loss.type)

            if best_loss > test_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = test_loss

