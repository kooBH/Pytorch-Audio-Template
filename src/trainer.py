import torch
import argparse
import torchaudio
import os
import numpy as np

from tensorboardX import SummaryWriter
from utils.hparams import HParam
from utils.writer import MyWriter

## import model & dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelsave_path', '-m', type=str, required=True)
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

    ## get parameters

    batch_size = hp.train.batch_size
    num_epochs = hp.train.epoch
    num_workers = hp.train.num_workers

    ## dirs
    modelsave_path = args.modelsave_path +'/'+ args.version_name
    log_dir = hp.log.root+args.version_name
    
    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    writer = MyWriter(hp, log_dir)

    train_dataset = MyDataset(some_path,'train',hp)
    val_dataset = MyDataset(some_path,'test', hp)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    model = Model().to(device)
    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        model.load_state_dict(torch.load(args.chkpt, map_location=device))

    criterion = loss()
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

    step = args.step

    for epoch in range(num_epochs):
        ### TRAIN ####
        model.train()
        train_loss=0
        for i, (batch_data) in enumerate(train_loader):
            step +=1

            input_data = batch_data['input'].to(device)
            target_data = batch_data['target'].to(device)
           
            output_data = model(input_data)

            loss = criterion(output_data,target_data).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('TRAIN::Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
            train_loss+=loss.item()

            if step %  hp.train.summary_interval == 0:
                writer.log_training(loss,step)

        train_loss = train_loss/len(train_loader)
        torch.save(model.state_dict(), str(modelsave_path)+'/lastmodel.pth')
            
        #### EVAL ####
        model.eval()
        with torch.no_grad():
            val_loss =0.
            for j, (batch_data) in enumerate(val_loader):
                input_data = batch_data['input'].to(device)
                target_data = batch_data['target'].to(device)
            
                output_data = model(input_data)
               
                loss = criterion(input_audio,target_audio,audio_me_pe,eps=1e-8).to(device)
                print('TEST::Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, j+1, len(val_loader), loss.item()))
                val_loss +=loss.item()

            val_loss = val_loss/len(val_loader)
            scheduler.step(val_loss)

            input_audio = input_audio[0].cpu().numpy()
            target_audio= target_audio[0].cpu().numpy()
            audio_me_pe= audio_me_pe[0].cpu().numpy()

            writer.log_evaluation_scalar(val_loss,                                step)

            if best_loss > val_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = val_loss
               
