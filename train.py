import torch
import argparse
import torchaudio
import os
import numpy as np

from tensorboardX import SummaryWriter

from Dataset.DatasetDNS import DatasetDNS

from utils.hparams import HParam
from utils.writer import MyWriter

from common import run,get_model, evaluate, set_seed
from ptflops import get_model_complexity_info


def get_criterion(hp):
    if hp.loss.type == "MSELoss":
        criterion = torch.nn.MSELoss()
    elif hp.loss.type == "ListLoss" :
        from utils.Loss import ListLoss
        criterion = ListLoss(
            hp.loss,
            hp.loss.ListLoss.list,
            hp.loss.ListLoss.weight
        )
    else : 
        raise NotImplementedError("Loss type {} is not implemented".format(hp.loss.type))
    
    return criterion

def get_optimzer(hp):
    if hp.train.optimizer == 'Adam' :
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.train.Adam)
    elif hp.train.optimizer == 'AdamW' :
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp.train.AdamW.lr)
    elif hp.train.optimizer == "AdamP" : 
        from utils.optimizer import AdamP
        optimizer = AdamP(model.parameters(), lr=hp.train.AdamP.lr, weight_decay=hp.train.AdamP.weight_decay, betas = hp.train.AdamP.betas, wd_ratio = hp.train.AdamP.wd_ratio,projection = hp.train.AdamP.projection)
    else :
        raise Exception("ERROR::Unknown optimizer : {}".format(hp.train.optimizer))
    return optimizer

def get_scheduler(hp,optimizer,train_loader = None) :
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
    elif hp.scheduler.type == "LinearPerEpoch" :
        from utils.schedule import LinearPerEpochScheduler
        scheduler = LinearPerEpochScheduler(optimizer, len(train_loader))
    elif hp.scheduler.type == "CosineAnnealingLR" : 
       scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.scheduler.CosineAnnealingLR.T_max, eta_min=hp.scheduler.CosineAnnealingLR.eta_min) 
    elif hp.scheduler.type == "StepLR" :
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hp.scheduler.StepLR.step_size, gamma=hp.scheduler.StepLR.gamma)
    elif hp.scheduler.type == "CosineAnnealingWarmRestarts" :
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=hp.scheduler.CosineAnnealingWarmRestarts.T_0, T_mult=hp.scheduler.CosineAnnealingWarmRestarts.T_mult, eta_min=hp.scheduler.CosineAnnealingWarmRestarts.eta_min)
    elif hp.scheduler.type == "Fixed" : 
        from torch.optim.lr_scheduler import LambdaLR
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    else :
        raise Exception("Unsupported sceduler type : {}".format(hp.scheduler.type))
    
    warmup = None
    if hp.scheduler.use_warmup : 
        from utils.schedule import WarmUpScheduler
        warmup = WarmUpScheduler(optimizer, len(train_loader))
    return scheduler, warmup

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', type=str, default=None,
                        help="default configuration")
    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--step','-s',type=int,required=False,default=0)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    parser.add_argument('--epoch','-e',type=int,required=False,default=None)
    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)

    #hp = HParam(args.config,args.default,merge_except=["architecture"])
    hp = HParam(args.config,args.default)
    print("NOTE::Loading configuration : "+args.config)

    device = args.device
    version = args.version_name
    torch.cuda.set_device(device)

    batch_size = hp.train.batch_size
    if args.batch_size is not None :
        batch_size = args.batch_size

    if args.epoch is None : 
        num_epochs = hp.train.epoch
    else :
        num_epochs = args.epoch
    num_workers = hp.train.num_workers

    best_loss = 1e7

    ## load
    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + version
    log_dir = hp.log.root+'/'+'log'+'/'+version

    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)

    writer = MyWriter(hp, log_dir)

    # TODO
    train_dataset = DatasetDNS(hp.data.root_train)
    test_dataset= DatasetDNS(hp.data.root_test)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    model = get_model(hp,device=device)
    # or model = get_model(hp).to(device)

    macs_ptflos, params_ptflops = get_model_complexity_info(model, (16000,), as_strings=False,print_per_layer_stat=False,verbose=False)   
    print("ptflops : MACS {}M |  PARAM {}K".format(macs_ptflos/1e6,params_ptflops/1e3))

    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        model.load_state_dict(torch.load(args.chkpt, map_location=device))

    criterion = get_criterion(hp).to(device)
    optimizer = get_optimzer(hp)
    scheduler, warmup = get_scheduler(hp,optimizer,train_loader)

    step = args.step
    log_train_cnt = 0
    log_dev_cnt = 0

    for epoch in range(num_epochs):
        ### TRAIN ####
        model.train()
        train_loss=0
        for i, (batch_data) in enumerate(train_loader):
            step +=batch_size
            log_train_cnt +=batch_size
            
            # TODO
            loss = run(batch_data,model,criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
           

            if log_train_cnt > hp.train.train_interval :
                print('TRAIN::{} : Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(version,epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
                writer.log_value(loss,step,'train loss : '+hp.loss.type)
                log_train_cnt = 0

        train_loss = train_loss/len(train_loader)
        torch.save(model.state_dict(), str(modelsave_path)+'/lastmodel.pt')
            
        #### EVAL ####
        model.eval()
        with torch.no_grad():
            test_loss =0.
            for j, (batch_data) in enumerate(test_loader):
                loss = run(batch_data,model,criterion)
                test_loss += loss.item()
                log_dev_cnt += batch_size

                if log_dev_cnt > hp.train.dev_interval :
                    print('TEST::{} :  Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(version, epoch+1, num_epochs, j+1, len(test_loader), loss.item()))
                    test_loss +=loss.item()
                    log_dev_cnt = 0

            test_loss = test_loss/len(test_loader)
            scheduler.step(test_loss)
            
            writer.log_value(test_loss,step,'test los : ' + hp.loss.type)

            # metric = evaluate(hp,model,list_eval,device=device)
            # for m in hp.log.eval : 
            #     writer.log_value(metric[m],step,m+"_VD")

            if epoch % hp.train.eval_interval == 0 or epoch == num_epochs-1:
                """
                idx = np.random.randint(0,len(test_dataset))
                batch_data = test_dataset[idx]
                noisy,estim,clean = run_sample(batch_data,idx)
                noisy = noisy[0][0].cpu().numpy()
                estim = estim[0].cpu().numpy()
                clean = clean[0].cpu().numpy()

                writer.log_spec(noisy,"noisy_s",step)
                writer.log_spec(estim,"estim_s",step)
                writer.log_spec(clean,"clean_s",step)

                writer.log_audio(noisy,"noisy_a",step)
                writer.log_audio(estim,"estim_a",step)
                writer.log_audio(clean,"clean_a",step)

                metric = evaluate(hp,model,list_eval,device=device)
                for m in hp.log.eval : 
                    writer.log_value(metric[m],step,m+"_custom")
                    print("METRIC::{} : {} : {:.4f}".format(version,m,metric[m]))   
                """
                pass

            if best_loss > test_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = test_loss

    writer.close()

