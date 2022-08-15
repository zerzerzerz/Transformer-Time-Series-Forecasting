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

def load_json(path):
    with open(path,'r') as f:
        res = json.load(f)
    return res

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='national_illness')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='result-debug')
    parser.add_argument('--record_path', type=str, default='record-IMS-smaller_model.csv')
    parser.add_argument('--inference_method', type=str, default='fixed_len', choices=['fixed_len','dynamic_decoding'])
    parser.add_argument('--num_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=8)
    parser.add_argument('--nhead', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--num_encoder_layers', type=int, default=2)
    parser.add_argument('--num_decoder_layers', type=int, default=1)
    parser.add_argument('--dim_feedforward', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--context_len', type=int, default=36)
    parser.add_argument('--pred_len', type=int, default=24)
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--num_channel', type=int, default=7)
    parser.add_argument('--gamma', type=float, default=0.95)
    args = parser.parse_args()
    return args


class MyDataset(Dataset):
    def __init__(self,dataset_name,context_len,step,pred_len,mode,data_root='data',eps=1e-6) -> None:
        super().__init__()
        data = pd.read_csv(join(data_root,dataset_name+'.csv')).values[:,1:].astype(np.float32)
        
        if dataset_name == 'exchange_rate':
            train_len = int(data.shape[0] * 0.70)
            val_len   = int(data.shape[0] * 0.15)
            test_len  = int(data.shape[0] * 0.15)
        else:
            train_len = int(data.shape[0] * 0.7)
            val_len   = int(data.shape[0] * 0.1)
            test_len  = int(data.shape[0] * 0.2)
        
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
        else:
            raise NotImplementedError(f'mode={mode} is invalid')
        
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
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
          

class TransAm(nn.Module):
    def __init__(self,d_model=512,num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=2048, nhead=8,dropout=0.1,num_channel=-1):
        '''input is [T,B,D]'''
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.tgt_mask = None
        self.pos_encoder = PositionalEncoding(d_model)

        self.model = nn.Transformer(d_model,nhead,num_encoder_layers,num_decoder_layers,dim_feedforward=dim_feedforward,dropout=dropout)

        self.up = nn.Linear(num_channel,d_model)
        self.down = nn.Linear(d_model,num_channel)

    def forward(self,src,tgt):
        # src.shape == [L,B,C]
        src = self.up(src)
        tgt = self.up(tgt)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        if self.tgt_mask is None or self.tgt_mask.size(0) != len(tgt):
            device = tgt.device
            mask = self._generate_square_subsequent_mask(len(tgt)).to(device)
            self.tgt_mask = mask

        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        output = self.down(self.model(src, tgt, self.src_mask, self.tgt_mask))
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

    print('{:<20}\ttrain.num_seq = {:<6}'.format(args.dataset_name, len(train_dataset)))
    print('{:<20}\tval.num_seq   = {:<6}'.format(args.dataset_name, len(val_dataset)))
    print('{:<20}\ttest.num_seq  = {:<6}'.format(args.dataset_name, len(test_dataset)))

    train_loader = DataLoader(train_dataset,args.batch_size,shuffle=True,drop_last=False,num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset,args.batch_size,shuffle=False,drop_last=False,num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset,args.batch_size,shuffle=False,drop_last=False,num_workers=args.num_workers)

    model = TransAm(
        args.d_model,
        args.num_encoder_layers,
        args.num_decoder_layers,
        args.dim_feedforward,
        args.nhead,
        args.dropout,
        args.num_channel
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=args.gamma,verbose=True)

    summary_writer = SummaryWriter(args.output_dir)
    train_logger = Logger(join(args.output_dir,'train.txt'))
    test_logger = Logger(join(args.output_dir,'test.txt'))
    val_logger = Logger(join(args.output_dir,'val.txt'))
    save_json(vars(args),join(args.output_dir,'args.json'))

    ck_dir = join(args.output_dir,'checkpoint')
    mkdir(ck_dir)

    best_val_mse = best_test_mae = best_test_mse = np.inf
    for epoch in tqdm(range(args.num_epoch)):
        train(train_loader, model, optimizer, scheduler, epoch, summary_writer, args, train_logger)
        val_mae, val_mse   = test_or_val(val_loader, model, epoch, summary_writer, 'val', args, val_logger)
        test_mae, test_mse = test_or_val(test_loader, model, epoch, summary_writer, 'test', args, test_logger)

        if best_val_mse > val_mse:
            best_val_mse = val_mse
            best_test_mae = test_mae
            best_test_mse = test_mse
            torch.save(model.cpu(),join(ck_dir,'best.pt'))
            model.to(args.device)
    
    args.mae = best_test_mae
    args.mse = best_test_mse

    try:
        df = pd.read_csv(args.record_path)
    except:
        df = pd.DataFrame()
    df = pd.concat([df,pd.DataFrame([vars(args)])])
    df.to_csv(args.record_path, index=False)
    
    torch.save(model.cpu(),join(ck_dir,'final.pt'))



def test_or_val(dataloader, model, epoch, summary_writer, mode, args, logger):
    with torch.no_grad():
        model.eval()
        epoch_mae = epoch_mse = num_batch = 0
        for context, target in dataloader:
            context = rearrange(context,'b t d -> t b d').to(args.device)
            target = rearrange(target,'b t d -> t b d').to(args.device)
            src = context.clone()
            
            prediction = []
            for _ in range(target.shape[0] // args.step):
                current_step_prediction = model(src,context)
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
        prediction = model(context,context)
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
    # args = get_args()
    # main(args)


    info = load_json('dataset-info.json')
    args = get_args()
    result_dir = 'result-vanilla-Transformer-smaller_model'
    inference_method = 'fixed_len'
    # inference_method = 'dynamic_decoding'
    device = 'cuda:0'

    for d in info:
        dataset_name = d['dataset_name']

        # if dataset_name not in ['exchange_rate']:
        if dataset_name in ['ETTh1','ETTh2','ETTm1','national_illness',\
            'ETTm2', 'electricity', 'exchange_rate']:
            continue

        context_len = d['context_len']
        num_channel = d['num_channel']
        for pred_len in d['pred_lens']:
            args.d_model = d['d_model']
            args.dim_feedforward = d['dim_feedforward']
            args.nhead = d['nhead']
            args.dataset_name = dataset_name
            args.context_len = context_len
            args.num_channel = num_channel
            args.pred_len = pred_len
            args.inference_method = inference_method
            args.device = device
            args.output_dir = f'{result_dir}/{dataset_name}/pred_len={pred_len}/{inference_method}'
            args.record_path = 'record-IMS-smaller_model.csv'

            main(args)