import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import pytorch_lightning as pl

from sklearn.metrics import confusion_matrix

from src.mil.pl_model.forward_fn import dtfdmil_forward_1st_tier, dtfdmil_forward_2nd_tier


class DTFDTrainerModule(pl.LightningModule):
    def __init__(self, 
            seed, 
            classifier_list, 
            loss_list, 
            accumulate_grad_batches, 
            grad_clipping, 
            metrics, 
            total_instance, 
            lr,
            weight_decay,
            lr_decay_ratio,
            numGroup, 
            epochs,
            distill):
        super(DTFDTrainerModule, self).__init__()

        self.seed = seed
        self.test_preds = []
        self.test_labels = []

        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay_ratio = lr_decay_ratio

        self.automatic_optimization = False

        self.classifier = classifier_list[0]
        self.attention = classifier_list[1]
        self.dimReduction = classifier_list[2]
        self.UClassifier = classifier_list[3]

        self.total_instance = total_instance
        self.numGroup = numGroup
        self.distill = distill     

        self.loss0 = loss_list[0]
        self.loss1 = loss_list[1]

        self.metrics = metrics

        self.accumulate_grad_batches = accumulate_grad_batches
        self.grad_clipping = grad_clipping

        self.train_metrics = metrics.clone(postfix='/train')
        self.val_metrics = nn.ModuleList([metrics.clone(postfix='/val'), metrics.clone(postfix='/test')])
        self.test_metrics = metrics.clone(prefix='final_test/')
        
        self.y_prob_list = []
        self.label_list = []

        # self.save_hyperparameters("args")
    
    def forward(self, feats, caption=None, label=None, train=False):

        loss0, slide_pseudo_feat = dtfdmil_forward_1st_tier(
             data=feats,
             classifier=self.classifier,
             attention=self.attention,
             dimReduction=self.dimReduction,
             loss0=self.loss0,
             caption=caption,
             label=label,
             total_instance=self.total_instance,
             numGroup=self.numGroup,
             distill=self.distill             
        )

        
        if train:
            self.manual_backward(loss0, retain_graph=True) # retain_graph=True -> used when want to backward through the graph a second time
            torch.nn.utils.clip_grad_norm_(self.dimReduction.parameters(), self.grad_clipping)
            torch.nn.utils.clip_grad_norm_(self.attention.parameters(), self.grad_clipping)
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), self.grad_clipping)

        loss1, gSlidePred = dtfdmil_forward_2nd_tier(slide_pseudo_feat, self.UClassifier, self.loss1, caption, label)

        if train:
            self.manual_backward(loss1)
            torch.nn.utils.clip_grad_norm_(self.UClassifier.parameters(), self.grad_clipping)

        return loss0, loss1, gSlidePred
    
    def training_step(self, batch, batch_idx):
        feats, caption, label = batch
        feats = feats.squeeze(0) # remove the batch size (1)

        loss0, loss1, Y_prob = self.forward(feats, caption, label, train=True) # Y_prob not being softmax?

        optimizer0, optimizer1 = self.optimizers()

        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            optimizer0.step()
            optimizer1.step()
            optimizer0.zero_grad()
            optimizer1.zero_grad()
        
        total_loss = loss0 + loss1
        self.log("Loss/train", total_loss, on_step=True, on_epoch=True, sync_dist=True)

        # https://github.com/Lightning-AI/pytorch-lightning/issues/2210
        self.train_metrics.update(Y_prob, label) # metrics are calculated per epoch, not per step (batch) -> solution: use update
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        feats, caption, label = batch
        feats = feats.squeeze(0) # remove the batch size (1)
     
        loss0, loss1, Y_prob = self.forward(feats, caption, label, train=False)

        if not self.trainer.sanity_checking:
            prefix = get_prefix_from_val_id(dataloader_idx) # val/test
            metrics_idx = dataloader_idx if dataloader_idx is not None else 0
            total_loss = loss0 + loss1
            self.log("Loss/%s" % prefix, total_loss, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
            self.val_metrics[metrics_idx].update(Y_prob, label)
            self.log_dict(self.val_metrics[metrics_idx], on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)

        return total_loss

    def test_step(self, batch, batch_idx):
        feats, caption, label = batch
        feats = feats.squeeze(0) # remove the batch size (1)
       
        loss0, loss1, Y_prob = self.forward(feats, caption, label, train=False)

        total_loss = loss0 + loss1
        self.log("Loss/final_test", total_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.test_metrics.update(Y_prob, label)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, sync_dist=True)

        self.test_preds.append(np.argmax(Y_prob.cpu().numpy(), axis=1))
        self.test_labels.append(label.cpu().numpy())

        self.y_prob_list.append(Y_prob)
        self.label_list.append(label)
        
        return self.test_metrics

    def on_train_epoch_end(self):
        scheduler0, scheduler1 = self.lr_schedulers()
        if scheduler0 is not None and scheduler1 is not None:
            scheduler0.step()
            scheduler1.step()
    
    # def on_test_epoch_end(self):
    #     all_preds = np.concatenate(self.test_preds)
    #     all_labels = np.concatenate(self.test_labels)
    #     cm = confusion_matrix(all_labels, all_preds)

    #     class_names = ['HP', 'IP', 'LP', 'SSL', 'TA', 'TSA', 'TVA+VA']
    #     plt.figure(figsize=(10, 7))
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    #     plt.xlabel('Predicted')
    #     plt.ylabel('True')
    #     plt.title(f"{self.dataset_name}/{self.magnification}/{self.patch_size}/{self.mil_model}-{self.distill}/{self.feature_extractor}/seed_{self.seed}")

    #     plt.savefig(f"{self.output_dir}/{self.dataset_name}/{self.magnification}/{self.patch_size}/{self.mil_model}-{self.distill}/{self.feature_extractor}/seed_{self.seed}/confusion_matrix.jpg", format="jpg")

    #     self.test_preds = []
    #     self.test_labels = []
    
    def configure_optimizers(self):
        trainable_parameters = []
        trainable_parameters += list(self.classifier.parameters())
        trainable_parameters += list(self.attention.parameters())
        trainable_parameters += list(self.dimReduction.parameters())

        params_optimizer_0 = [{"params": filter(lambda p: p.requires_grad, trainable_parameters)}]
        params_optimizer_1 = [{"params": filter(lambda p: p.requires_grad, self.UClassifier.parameters())}]

        optimizer0 = torch.optim.Adam(params_optimizer_0, lr=self.lr, weight_decay=self.weight_decay)
        optimizer1 = torch.optim.Adam(params_optimizer_1, lr=self.lr, weight_decay=self.weight_decay)

        scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer0, [int(self.epochs/2)], gamma=self.lr_decay_ratio)
        scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [int(self.epochs/2)], gamma=self.lr_decay_ratio)

        return [
            {"optimizer": optimizer0, "lr_scheduler": scheduler0},
            {"optimizer": optimizer1, "lr_scheduler": scheduler1}
        ]

def get_prefix_from_val_id(dataloader_idx):
    if dataloader_idx is None or dataloader_idx == 0:
        return "val"
    elif dataloader_idx == 1:
        return "test"
    else:
        return NotImplementedError
