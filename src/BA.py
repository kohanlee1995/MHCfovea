import os, sys, copy
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics as metrics
from tqdm import tqdm


class BA():
    def __init__(self, mhc_dict, model, criterion, optimizer, scheduler_milestones, device, y_value_index, logdir):
        self.mhc_dict = mhc_dict
        self.model = model
        self.best_model = None
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler_milestones = scheduler_milestones
        self.device = device
        self.y_value_index = y_value_index
        self.writer = SummaryWriter(logdir)
        self.step = 0
        self.epoch = 0
        
        self.model.to(self.device)
        
        self.modeldir = "%s/model"%logdir
        if not os.path.isdir(self.modeldir):
            os.mkdir(self.modeldir)
        else:
            if os.path.isfile("%s/model_best.tar"%self.modeldir):
                self.best_model = copy.deepcopy(self.model)
                model_state_dict = torch.load("%s/model_best.tar"%self.modeldir, map_location=self.device)
                self.best_model.load_state_dict(model_state_dict["model_state_dict"])
                print("load best model")


    def train(self, train_mhc_idx, train_dataloader, valid_mhc_idx, valid_dataloader, num_epochs):
        if self.scheduler_milestones != [0]:
            scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.scheduler_milestones, gamma=0.1, last_epoch=-1)
        best_val_loss = np.inf

        # initial validation
        val_loss = self.validation(valid_mhc_idx, valid_dataloader, self.model)
        
        for epoch in tqdm(range(1, num_epochs+1), desc="epoch", leave=False, position=0):
            self.epoch += 1
            # epoch training
            self._train_epoch(train_mhc_idx, train_dataloader)
            torch.save({'model_state_dict': self.model.state_dict()}, "%s/model_epoch_%d.tar"%(self.modeldir, epoch))
            # validation
            val_loss = self.validation(valid_mhc_idx, valid_dataloader, self.model)
            # scheduler
            if self.scheduler_milestones != [0]:
                scheduler.step()
            # record best model
            if (epoch >= np.floor(num_epochs*0.8)):
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_model = copy.deepcopy(self.model)
        
        torch.save({'model_state_dict': self.best_model.state_dict()}, "%s/model_best.tar"%self.modeldir)
    
    
    # only for non-shuffle dataloader
    def validation(self, mhc_idx, dataloader, model):
        y, pred, loss = self._pred(mhc_idx, dataloader, model)
        metrics = self._metrics(y, pred)
        self.writer.add_scalar("Loss/Validation", loss, self.epoch)
        self.writer.add_scalar("AUC/Validation", metrics["AUC"], self.epoch)
        self.writer.add_scalar("AUC0.1/Validation", metrics["AUC0.1"], self.epoch)
        self.writer.add_scalar("AP/Validation", metrics["AP"], self.epoch)
        self.writer.add_scalar("PPV/Validation", metrics["PPV"], self.epoch)
        return loss.item()

    
    def _train_epoch(self, mhc_idx, dataloader):
        batch_num = len(dataloader)
        record_num = 20
        record_interval = int(np.floor(batch_num/record_num))
        record_loss = list()

        for j, (x, y) in enumerate(tqdm(dataloader, desc="batch", leave=False, position=1), 1):
            self.step += 1
            self.model.train()
            self.optimizer.zero_grad()

            mhc = torch.FloatTensor([self.mhc_dict[mhc_idx[int(y[i,0])]] for i in range(y.shape[0])]).to(self.device)
            epitope = x.to(self.device).float()
            pred = self.model(mhc, epitope)
            y = y[:, self.y_value_index].to(self.device).float().view(-1, 1)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            
            record_loss.append(loss.item())
            if (j % record_interval == 0) or (j == batch_num):
                self.writer.add_scalar("Loss/Train", np.mean(record_loss), self.step)
                record_loss = list()
    

    def _pred(self, mhc_idx, dataloader, model):
        model.eval()
        loss_list = list()
        ys = torch.tensor([])
        preds = torch.tensor([])
        
        for j, (x, y) in enumerate(dataloader, 0):
            with torch.no_grad():
                mhc = torch.FloatTensor([self.mhc_dict[mhc_idx[int(y[i,0])]] for i in range(y.shape[0])]).to(self.device)
                epitope = x.to(self.device).float()
                pred = model(mhc, epitope).to("cpu")
                y = y[:, self.y_value_index].float().view(-1, 1)
                loss = self.criterion(pred, y)
                pred = pred.view(-1,).numpy()
                
            if j == 0:
                preds = pred
                ys = y.view(-1,).numpy()
            else:
                preds = np.append(preds, pred, axis=0)
                ys = np.append(ys, y.view(-1,).numpy(), axis=0)
            
            loss_list.append(loss.item())
        
        return ys, preds, np.mean(loss_list)


    # AUC, AUC0.1, AP, PPV
    def _metrics(self, y, pred):
        # AUC
        fpr, tpr, _ = metrics.roc_curve(y, pred)
        auc_score = np.around(metrics.auc(fpr, tpr), decimals=3)
        # AUC0.1
        idx = np.where(fpr <= 0.1)[0]
        auc01_score = np.around(np.trapz(tpr[idx], fpr[idx]) * 10, decimals=3)
        # AP
        avg_precision_score = np.around(metrics.average_precision_score(y, pred), decimals=3)
        # PPV: method from netMHCpan4.1
        num = int(sum(y==1) * 0.95)
        idx = pred.argsort()[::-1][:num]
        ppv = np.around(y[idx].sum() / num, decimals=3)

        return {"AUC": auc_score, "AUC0.1": auc01_score, "AP": avg_precision_score, "PPV": ppv}