import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from os.path import join
import pandas as pd
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import json
import datetime
from tqdm import tqdm
import os

def mkdir(p):
    if os.path.isdir(p):
        return
    else:
        os.makedirs(p)

def get_datetime():
    time1 = datetime.datetime.now()
    time2 = datetime.datetime.strftime(time1,'%Y-%m-%d %H:%M:%S')
    return str(time2)

class Logger():
    def __init__(self,log_file_path) -> None:
        self.path = log_file_path
        with open(self.path,'w') as f:
            f.write(get_datetime() + "\n")
            print(get_datetime())
        return
    
    def log(self,content):
        with open(self.path,'a') as f:
            f.write(content + "\n")
            print(content)
        return


def save_json(obj,path):
    with open(path,'w') as f:
        json.dump(obj,f,indent=4)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='electricity')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='result-debug')
    parser.add_argument('--record_path', type=str, default='record-IMS.csv')
    parser.add_argument('--inference_method', type=str, default='fixed_len', choices=['fixed_len','dynamic_decoding'])
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--context_len', type=int, default=96)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--num_channel', type=int, default=321)
    parser.add_argument('--gamma', type=float, default=0.95)
    args = parser.parse_args()
    return args


class MyDataset(Dataset):
    def __init__(self,dataset_name,context_len,step,pred_len,mode,data_root='data',eps=1e-6) -> None:
        super().__init__()
        data = pd.read_csv(join(data_root,dataset_name+'.csv')).values[:,1:].astype(np.float32)
        
        train_len = int(data.shape[0] * 0.7)
        val_len = int(data.shape[0] * 0.1)
        test_len = int(data.shape[0] * 0.2)
        
        i = 0
        train = data[i: i+train_len]
        i += train_len
        val = data[i: i+val_len]
        i += val_len
        test = data[i: i+test_len]

        train = torch.from_numpy(train)
        val = torch.from_numpy(val)
        test = torch.from_numpy(test)

        mean = train.mean(dim=0,keepdim=True)
        std = train.std(dim=0,keepdim=True)

        train = (train - mean) / (std + eps)
        val = (val - mean) / (std + eps)
        test = (test - mean) / (std + eps)

        context = []
        target = []
        if mode == 'train':
            i = 0
            while i + step + context_len <= train.shape[0]:
                context.append(train[i:i+context_len])
                target.append(train[i+step:i+step+context_len])
                i += 1
        elif mode == 'val':
            i = 0
            while i+context_len+pred_len <= val.shape[0]:
                context.append(val[i:i+context_len])
                target.append(val[i+context_len:i+context_len+pred_len])
                i += 1
        elif mode == 'test':
            i = 0
            while i+context_len+pred_len <= test.shape[0]:
                context.append(test[i:i+context_len])
                target.append(test[i+context_len:i+context_len+pred_len])
                i += 1
        
        self.context = context
        self.target = target
        self.length = len(target)

    def __getitem__(self, index):
        return self.context[index], self.target[index]
    
    def __len__(self):
        return self.length
        

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [L,C] -> [1,L,C] -> [L,1,C]
        pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
          

class TransAm(nn.Module):
    def __init__(self,d_model=512,num_layers=1,nhead=8,dropout=0.1,num_channel=-1):
        '''input is [T,B,D]'''
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(d_model,num_channel)
        self.up = nn.Linear(num_channel,d_model)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        # src.shape == [L,B,C]
        src = self.up(src)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def compute_loss(y,y_hat):
    '''mse'''
    return torch.nn.functional.mse_loss(y,y_hat,reduction='mean')

def compute_metric(y,y_hat):
    '''mae,mse'''
    mae = torch.nn.functional.l1_loss(y,y_hat,reduction='mean')
    mse = torch.nn.functional.mse_loss(y,y_hat,reduction='mean')
    return mae, mse

def main(args):
    torch.manual_seed(0)
    np.random.seed(0)

    train_dataset = MyDataset(args.dataset_name, args.context_len, args.step, args.pred_len, 'train')
    test_dataset = MyDataset(args.dataset_name, args.context_len, args.step, args.pred_len, 'test')
    val_dataset = MyDataset(args.dataset_name, args.context_len, args.step, args.pred_len, 'val')

    train_loader = DataLoader(train_dataset,args.batch_size,shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset,args.batch_size,shuffle=False,drop_last=True)
    val_loader = DataLoader(val_dataset,args.batch_size,shuffle=False,drop_last=True)

    model = TransAm(args.d_model,args.num_layers,args.nhead,args.dropout,args.num_channel).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=args.gamma,verbose=True)

    summary_writer = SummaryWriter(args.output_dir)
    train_logger = Logger(join(args.output_dir,'train.txt'))
    test_logger = Logger(join(args.output_dir,'test.txt'))
    val_logger = Logger(join(args.output_dir,'val.txt'))
    save_json(vars(args),join(args.output_dir,'args.json'))

    best_val_mse = best_test_mae = best_test_mse = np.inf
    for epoch in tqdm(range(args.num_epoch)):
        train(train_loader, model, optimizer, scheduler, epoch, summary_writer, args, train_logger)
        val_mae, val_mse   = test_or_val(val_loader, model, epoch, summary_writer, 'val', args, val_logger)
        test_mae, test_mse = test_or_val(test_loader, model, epoch, summary_writer, 'test', args, test_logger)

        if best_val_mse > val_mse:
            best_val_mse = val_mse
            best_test_mae = test_mae
            best_test_mse = test_mse
    
    args.mae = best_test_mae
    args.mse = best_test_mse

    try:
        df = pd.read_csv(args.record_path)
    except:
        df = pd.DataFrame()
    df = pd.concat([df,pd.DataFrame([vars(args)])])
    df.to_csv(args.record_path, index=False)
    



def test_or_val(dataloader, model, epoch, summary_writer, mode, args, logger):
    with torch.no_grad():
        model.eval()
        epoch_mae = epoch_mse = num_batch = 0
        for context, target in dataloader:
            context = rearrange(context,'b t d -> t b d').to(args.device)
            target = rearrange(target,'b t d -> t b d').to(args.device)
            
            prediction = []
            for _ in range(target.shape[0] // args.step):
                current_step_prediction = model(context)
                prediction.append(current_step_prediction[-args.step:])
                if args.inference_method == 'dynamic_decoding':
                    context = torch.cat([context,current_step_prediction[-args.step:]],dim=0)
                elif args.inference_method == 'fixed_len':
                    context = torch.cat([context[args.step:],current_step_prediction[-args.step:]],dim=0)
                else:
                    raise NotImplementedError(f'Inference method = {args.inference_method} is not implemented')
            prediction = torch.cat(prediction,dim=0)

            mae, mse = compute_metric(target,prediction)

            epoch_mae += mae.item()
            epoch_mse += mse.item()
            num_batch += 1

        epoch_mae /= num_batch            
        epoch_mse /= num_batch      

        summary_writer.add_scalar(f'MAE/{mode}', epoch_mae, epoch)      
        summary_writer.add_scalar(f'MSE/{mode}', epoch_mse, epoch)     
        logger.log("Epoch={:<2}\tMode={:<5}\tMAE={:<8.6f}\tMSE={:8.6f}".format(epoch,mode,epoch_mae,epoch_mse)) 

        return epoch_mae, epoch_mse 


def train(dataloader, model, optimizer, scheduler, epoch, summary_writer ,args, logger):
    model.train()
    epoch_loss = epoch_mae = epoch_mse = num_batch = 0
    for context, target in dataloader:
        context = rearrange(context,'b t d -> t b d').to(args.device)
        target = rearrange(target,'b t d -> t b d').to(args.device)
        prediction = model(context)
        loss = compute_loss(target,prediction)
        loss.backward()
        optimizer.step()

        mae, mse = compute_metric(target,prediction)
        epoch_loss += loss.item()
        epoch_mae += mae.item()
        epoch_mse += mse.item()
        optimizer.zero_grad()
        num_batch += 1

    scheduler.step()
    epoch_loss /= num_batch
    epoch_mae /= num_batch
    epoch_mse /= num_batch

    summary_writer.add_scalar('Loss/train', epoch_loss, epoch)
    summary_writer.add_scalar('MAE/train', epoch_mae, epoch)
    summary_writer.add_scalar('MSE/train', epoch_mse, epoch)
    logger.log("Epoch={:<2}\tMode={:<5}\tMAE={:<8.6f}\tMSE={:8.6f}".format(epoch,'train',epoch_mae,epoch_mse)) 

    return epoch_loss, epoch_mae, epoch_mse
    






if __name__ == '__main__':
    args = get_args()
    main(args)