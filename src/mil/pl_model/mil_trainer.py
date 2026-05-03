import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import pytorch_lightning as pl

from sklearn.metrics import confusion_matrix
from pytorch_optimizer import Lookahead


class MILTrainerModule(pl.LightningModule):
    def __init__(self, mil_model, accumulate_grad_batches, seed, classifier, loss, opt, lr, weight_decay, metrics, batch_size, num_epochs, num_classes, forward_func="general"):
        super(MILTrainerModule, self).__init__()


        self.seed = seed
        self.mil_model = mil_model
        self.batch_size = batch_size
        self.opt = opt
        self.lr = lr
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.test_preds = []
        self.test_labels = []
        
        # Lookahead optimizer (TransMIL) can only work with automatic optimization
        if self.mil_model.lower() != "TransMIL":
            self.automatic_optimization = False # default automatic optimization is True

        self.classifier = classifier
        self.loss = loss
        self.metrics = metrics

        self.num_classes = num_classes
        
        self.forward_func = forward_func

        self.accumulate_grad_batches = accumulate_grad_batches

        self.train_metrics = metrics.clone(postfix="/train")
        self.val_metrics = nn.ModuleList([metrics.clone(postfix="/val"), metrics.clone(postfix="/test")])
        self.test_metrics = metrics.clone(prefix="final_test/")
        
        self.y_prob_list = []
        self.label_list = []

        self.save_hyperparameters("")
    
    def classifier_forward(self, data, caption=None, label=None):
        
        return self.forward_func(data, self.classifier, self.loss, self.num_classes, caption=caption, label=label)
    
    def forward(self, feats, caption, label=None, train=False):
        bag_prediction, loss, Y_prob = self.classifier_forward(feats, caption, label) # depends on the mil aggregator type
      
        if train and self.mil_model != "TransMIL":
            self.manual_backward(loss) # manual backward classifier

        return bag_prediction, loss, Y_prob
    
    def training_step(self, batch, batch_idx):
        feats, caption, label = batch
        feats = feats.squeeze(0) # remove the batch size (1)
      
        y, loss, y_prob = self.forward(feats, caption, label, train=True)

        opt = self.optimizers()
        
        if self.mil_model != "TransMIL":

           if (batch_idx + 1) % self.accumulate_grad_batches == 0:
               opt.step()
               opt.zero_grad()
        
        self.log("Loss/train", loss, on_step=True, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

        # https://github.com/Lightning-AI/pytorch-lightning/issues/2210
        self.train_metrics.update(y_prob, label) # metrics are calculated per epoch, not per step (batch) -> solution: use update
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        feats, caption, label = batch
        feats = feats.squeeze(0) # remove the batch size (1)
     
        y, loss, y_prob = self.forward(feats, caption, label, train=False)

        if not self.trainer.sanity_checking:
            prefix = get_prefix_from_val_id(dataloader_idx) # val/test
            metrics_idx = dataloader_idx if dataloader_idx is not None else 0
            self.log("Loss/%s" % prefix, loss, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False, batch_size=self.batch_size)
            self.val_metrics[metrics_idx].update(y_prob, label)
            self.log_dict(self.val_metrics[metrics_idx], on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False, batch_size=self.batch_size)

        return loss

    def test_step(self, batch, batch_idx):
        feats, caption, label = batch
        feats = feats.squeeze(0) # remove the batch size (1)
       
        y, loss, y_prob = self.forward(feats, caption, label, train=False)

        self.log("Loss/final_test", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)
        self.test_metrics.update(y_prob, label)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.batch_size)

        self.test_preds.append(np.argmax(y_prob.cpu().numpy(), axis=1))
        self.test_labels.append(label.cpu().numpy())

        self.y_prob_list.append(y_prob)
        self.label_list.append(label)
        
        return self.test_metrics

    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        if sch is not None and self.mil_model != "TransMIL":
            sch.step()
    
    # def on_test_epoch_end(self):
    #     all_preds = np.concatenate(self.test_preds)
    #     all_labels = np.concatenate(self.test_labels)
    #     cm = confusion_matrix(all_labels, all_preds)
    #     cm_prob = confusion_matrix(all_labels, all_preds)
    #     cm_prob = cm_prob/cm_prob.sum(axis=1)[:,np.newaxis]

    #     class_names = ["HP", "IP", "LP", "SSL", "TA", "TSA", "TVA+VA"]
    #     # save cm
    #     plt.figure(figsize=(10, 7))
    #     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    #     plt.xlabel("Predicted")
    #     plt.ylabel("True")
    #     plt.title(f"{self..dataset_name}/{self..magnification}/{self..patch_size}/{self..mil_model}/{self..feature_extractor}/seed_{self.seed}")

    #     plt.savefig(f"{self..output_dir}/{self..dataset_name}/{self..magnification}/{self..patch_size}/{self..mil_model}/{self..feature_extractor}/seed_{self.seed}/confusion_matrix.jpg", format="jpg")

    #     # sace cm with probability
    #     plt.figure(figsize=(10, 7))
    #     sns.heatmap(cm_prob, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    #     plt.xlabel("Predicted")
    #     plt.ylabel("True")
    #     plt.title(f"{self..dataset_name}/{self..magnification}/{self..patch_size}/{self..mil_model}/{self..feature_extractor}/seed_{self.seed}")

    #     plt.savefig(f"{self..output_dir}/{self..dataset_name}/{self..magnification}/{self..patch_size}/{self..mil_model}/{self..feature_extractor}/seed_{self.seed}/confusion_matrix_prob.jpg", format="jpg")

    #     self.test_preds = []
    #     self.test_labels = []
    
    def configure_optimizers(self):
        print(self.lr, self.num_epochs)
        params = [{"params": filter(lambda p: p.requires_grad, self.classifier.parameters())}]
        
        if self.opt == "adam":
            cus_optimizer = torch.optim.Adam(params, lr=self.lr, betas=(0.5, 0.9), weight_decay=self.weight_decay)
        elif self.opt == "adamw":
            cus_optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.opt == "lookahead_radam" and self.mil_model == "TransMIL":
            base_optimizer = torch.optim.RAdam(params, lr=self.lr, weight_decay=self.weight_decay)
            cus_optimizer = Lookahead(base_optimizer)

        cus_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cus_optimizer, T_max=self.num_epochs, eta_min=5e-6)

        return {
            "optimizer": cus_optimizer,
            "lr_scheduler": cus_scheduler,
        }
    
def get_prefix_from_val_id(dataloader_idx):
    if dataloader_idx is None or dataloader_idx == 0:
        return "val"
    elif dataloader_idx == 1:
        return "test"
    else:
        return NotImplementedError
