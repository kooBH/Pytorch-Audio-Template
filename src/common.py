import torch
import torch.nn as nn


def get_model(hp):

    return

def run(data,model,criterion,hp,device="cuda:0",ret_output=False): 
    input = data['input'].to(device)
    target = data['target'].to(device)
    output = model(input)

    loss = criterion(output,target).to(device)
    if hp.loss.type == "MSELoss" : 
        loss = criterion(output,target).to(device)
    elif hp.loss.type == "wSDRLoss" : 
        loss = criterion(estim,noisy,target, alpha=hp.loss.wSDRLoss.alpha)
    
    if ret_output :
        return output, loss
    else : 
        return loss
