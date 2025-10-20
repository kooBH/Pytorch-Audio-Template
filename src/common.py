import torch
import torch.nn as nn
from utils.metric import run_metric
import librosa as rs


def get_model(hp):

    return

def run(data,model,hp,criterion=None,device="cuda:0",ret_output=False): 
    noisy = data['noisy'].to(device)
    clean = data['clean'].to(device)
    estim = model(noisy)

    if criterion is None : 
        return estim

    loss = criterion(estim,clean).to(device)
    if hp.loss.type == "MSELoss" : 
        loss = criterion(estim,clean).to(device)
    elif hp.loss.type == "wSDRLoss" : 
        loss = criterion(estim,noisy,clean, alpha=hp.loss.wSDRLoss.alpha)

    if loss.isinf().any() : 
        print("Warning::There is inf in loss, nan_to_num(1e-7)")
        loss = torch.tensor(0.0).to(loss.device)
        loss.requires_grad_()

    if loss.isnan().any() : 
        print("Warning::There is nan in loss, nan_to_num(1e-7)")
        loss = torch.tensor(0.0).to(loss.device)
        loss.requires_grad_()
    
    if ret_output:
        return estim, loss
    else : 
        return loss


def evaluate(hp, model,list_data,device="cuda:0"):
    #### EVAL ####
    model.eval()
    with torch.no_grad():
        ## Metric
        metric = {}
        for m in hp.log.eval : 
            metric["{}".format(m)] = 0.0

        for pair_data in list_data : 
            path_noisy = pair_data[0]
            path_clean = pair_data[1]
            noisy = rs.load(path_noisy,sr=hp.data.sr)[0]
            noisy = torch.unsqueeze(torch.from_numpy(noisy),0).to(device)
            estim = model(noisy).cpu().detach().numpy()[0]
            clean = rs.load(path_clean,sr=hp.data.sr)[0]

            if len(clean) > len(estim) :
                clean = clean[:len(estim)]
            else :
                estim = estim[:len(clean)]
            for m in hp.log.eval : 
                val= run_metric(estim,clean,m) 
                metric["{}".format(m)] += val
            
        for m in hp.log.eval : 
            key = "{}".format(m)
            metric[key] /= len(list_data)
    return metric