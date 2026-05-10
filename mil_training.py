from src.mil.model import get_model_module
from pytorch_lightning.callbacks import RichProgressBar
from pathlib import Path
import os
import random
import torch
import numpy as np
import pytorch_lightning as pl



SLIDES_PER_CLASS = [200, 150, 100, 50, 25]
RANDOM_SEED = 2024
BACKBONE_NAME = 'resnet50'
DATADIR = f"patch_latents/baseline/{BACKBONE_NAME}"
BATCH_SIZE = 1
MIL_MODELS = ['meanpooling', 'maxpooling','ABMIL', 'GABMIL', 'DSMIL', 'DTFD-MIL', 'TransMIL']
# MIL_MODELS = ['TransMIL']
NUM_EPOCHS = 50
DEVICE = 'cuda:0'
CLASSES = ['D', 'M', 'N']
NUM_CLASSES = 3
# Feature dim depends on the patch encoder used to build the latents:
#   resnet50 (layer4 stripped) = 1024, vit_tiny = 192, ctranspath = 768
NUM_FEATS = {'resnet50': 1024, 'vit_tiny': 192, 'ctranspath': 768}[BACKBONE_NAME]
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
ACCUMULATE_GRAD_BATCHES = 1
GRAD_CLIPPING = 5
TOTAL_INSTANCE = 4
NUMGROUP = 4
LR_DECAY_RATIO = 0.2


def main(organ, slides_per_class, mil_model):

    try:
        torch.set_float32_matmul_precision('medium')
    except Exception as e:
        print("Unable to activate TensorCore")
        print(e)


    data_module = PatchFeaturesModule(
        datadir=f"{DATADIR}/{organ}/{slides_per_class}",
        classes=CLASSES,
        batch_size=BATCH_SIZE)
    # data_module.setup()
    
    # one_batch = next(iter(data_module.train_dataloader()))

    # print(one_batch[0][0].size())
    
    # exit()

    trainer_model = get_model_module(
        seed=RANDOM_SEED,
        mil_model=mil_model,
        num_feats=NUM_FEATS,
        num_classes=NUM_CLASSES,
        opt='adam',
        lr=LEARNING_RATE,
        lr_decay_ratio=LR_DECAY_RATIO,
        batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        num_epochs=NUM_EPOCHS,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        grad_clipping=GRAD_CLIPPING,
        total_instance=TOTAL_INSTANCE,
        numGroup=NUMGROUP,
        distill='MaxMinS')

    
    
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        log_every_n_steps=50,
        num_sanity_val_steps=0,
        precision=16,
        accelerator='gpu',
        devices=[0, 1],
        callbacks=[RichProgressBar(leave=False)],
        strategy='ddp' if mil_model is not 'DTFD-MIL' else 'ddp_find_unused_parameters_true',

    )



    trainer.fit(trainer_model, data_module)

    test_results = trainer.test(trainer_model, data_module)

    del trainer_model        
    del trainer        
    del data_module    
    torch.cuda.empty_cache()

    # this is also stuck
    pl.utilities.memory.garbage_collection_cuda()

    test_acc = round(test_results[0]["final_test/ACC"] * 100, 4)
    test_auc = round(test_results[0]["final_test/AUROC"] * 100, 4)
    test_f1 = round(test_results[0]["final_test/F1"] * 100, 4)

    # print(f"Test acc: {test_acc}. Test AUC: {test_auc}. Test F1Score: {test_f1}\n\n")

    return test_acc, test_auc


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data_paths = [x[0] for x in data]
        self.labels = torch.tensor([x[1] for x in data])
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        latent = torch.load(data_path)
        label = self.labels[idx]
        return latent, np.nan, label
    

class PatchFeaturesModule(pl.LightningDataModule):
    def __init__(self, datadir, classes, batch_size):
        super().__init__()
        self.datadir = datadir
        self.classes = classes
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.data = {}
        datadir_path = Path(self.datadir)
        
        for subset in ['train', 'val', 'test']:
            subset_path = datadir_path / subset
            self.data[subset] = [
                (str(latent), label)
                for label, class_ in enumerate(sorted(self.classes))
                for latent in (subset_path / class_).iterdir()
            ]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=LatentDataset(self.data['train']), 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=8)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=LatentDataset(self.data['val']), 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=8)
        
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=LatentDataset(self.data['test']), 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=8)  


if __name__=='__main__':

    for organ in ['stomach']:    
        logdir = f"logs/slide_level/{organ}/mil/"
        Path(logdir).mkdir(exist_ok=True, parents=True)
        for mil_model in MIL_MODELS:
            writer_file = f"{logdir}/{mil_model}.txt"
            with open(writer_file, 'w') as writer:
                for slides_per_class in SLIDES_PER_CLASS:
                    print('='*50)
                    print('='*50, file=writer)
                    print(f"Organ: {organ}. Slides per class: {slides_per_class}. MIL Method: {mil_model}") 
                    print(f"Organ: {organ}. Slides per class: {slides_per_class}. MIL Method: {mil_model}", file=writer) 

                    seed_everything(RANDOM_SEED)
                    test_acc_list, test_auc_list = [], []
                    for _ in range(5):
                        acc, auc = main(organ=organ, slides_per_class=slides_per_class, mil_model=mil_model)
                        test_acc_list.append(acc)
                        test_auc_list.append(auc)

                    acc_mean = np.array(test_acc_list).mean()
                    acc_std = np.array(test_acc_list).std()
                    auc_mean = np.array(test_auc_list).mean()
                    auc_std = np.array(test_auc_list).std()
                    print(f"{organ}. ACCURACY: {acc_mean:.4f} +/- {acc_std:.4f}. AUC: {auc_mean:.4f} +/- {auc_std:.4f}\n\n")
                    print(f"{organ}. ACCURACY: {acc_mean:.4f} +/- {acc_std:.4f}. AUC: {auc_mean:.4f} +/- {auc_std:.4f}\n", file=writer)
    



