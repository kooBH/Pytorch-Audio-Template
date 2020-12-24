import torch
import argparse
import torchaudio
import os
import numpy as np

from fairseq import utils
from tensorboardX import SummaryWriter
from datasets.pickleDataset import pickleDataset

from model.model import Model

from utils.hparams import HParam
from utils.writer import MyWriter

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

    SNR = hp.train.SNR
    batch_size = hp.train.batch_size
    frame_num = hp.train.frame_num
    num_epochs = hp.train.epoch
    num_workers = hp.train.num_workers

    modelsave_path = args.modelsave_path +'/'+ args.version_name
    if not os.path.exists(modelsave_path):
        os.makedirs(modelsave_path)

    log_dir = hp.log.root+args.version_name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    pickle_path = hp.data.pkl

    writer = MyWriter(hp, log_dir)

    train_dataset = pickleDataset(pickle_path,'train',hp)
    val_dataset = pickleDataset(pickle_path,'test', hp)

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

            audio_real = batch_data["audio_data_Real"][0].to(device)
            audio_imagine = batch_data["audio_data_Imagine"][0].to(device)
            target_audio = batch_data["audio_wav"][1].squeeze(1).to(device)
            input_audio = batch_data["audio_wav"][0].squeeze(1).to(device)
            
            mask_r, mask_i = model(audio_real, audio_imagine)

            # Respectively
            if hp.train.type =='R':
                enhance_r = audio_real * mask_r
                enhance_i = audio_imagine * mask_i
            # Complex operation : CRM
            else :
                enhance_r = audio_real * mask_r - audio_imagine * mask_i
                enhance_i = audio_real * mask_i + audio_imagine * mask_r

            enhance_r = enhance_r.unsqueeze(3)
            enhance_i = enhance_i.unsqueeze(3)
            enhance_spec = torch.cat((enhance_r,enhance_i),3)
            audio_me_pe = complex_demand_audio(enhance_spec,window,audio_maxlen,fs)

            #print(audio_me_pe.shape)

            loss = criterion(input_audio,target_audio,audio_me_pe,eps=1e-8).to(device)
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
                audio_real = batch_data["audio_data_Real"][0].to(device)
                audio_imagine = batch_data["audio_data_Imagine"][0].to(device)
                target_audio = batch_data["audio_wav"][1].squeeze(1).to(device)
                input_audio = batch_data["audio_wav"][0].squeeze(1).to(device)
            
                mask_r, mask_i = model(audio_real, audio_imagine)
                # Respectively
                if hp.train.type =='R':
                    enhance_r = audio_real * mask_r
                    enhance_i = audio_imagine * mask_i
                # Complex operation : CRM
                else :
                    enhance_r = audio_real * mask_r - audio_imagine * mask_i
                    enhance_i = audio_real * mask_i + audio_imagine * mask_r

                enhance_r = enhance_r.unsqueeze(3)
                enhance_i = enhance_i.unsqueeze(3)
                enhance_spec = torch.cat((enhance_r,enhance_i),3)
                audio_me_pe = complex_demand_audio(enhance_spec,window,audio_maxlen,fs).to(device)
                
                loss = criterion(input_audio,target_audio,audio_me_pe,eps=1e-8).to(device)
                print('TEST::Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, j+1, len(val_loader), loss.item()))
                val_loss +=loss.item()

            val_loss = val_loss/len(val_loader)
            scheduler.step(val_loss)

            input_audio = input_audio[0].cpu().numpy()
            target_audio= target_audio[0].cpu().numpy()
            audio_me_pe= audio_me_pe[0].cpu().numpy()

            writer.log_evaluation(val_loss,
                                  input_audio,target_audio,audio_me_pe,
                                  #input_spec, target_spec,enhance_spec,
                                  step)

            if best_loss > val_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = val_loss
               
