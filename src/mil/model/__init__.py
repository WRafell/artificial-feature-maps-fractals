import torch
import torch.nn as nn
from torchmetrics import MetricCollection, Accuracy, AUROC, Precision, Recall, F1Score as F1
from src.mil.pl_model.mil_trainer import MILTrainerModule

# def apply_sparse_init(m):
#     if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
#         nn.init.orthogonal_(m.weight)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)

class switch_dim(nn.Module):
    def forward(self, x):
        x = x.unsqueeze(0)
        x = torch.transpose(x, 2, 1)
        return x
    

def get_metrics(num_classes, task):
    metrics = MetricCollection({
        "ACC": Accuracy(num_classes=num_classes, task=task), # default: average="micro" -> might be misleading if classes are imbalanced
        # "Balanced_ACC": Accuracy(num_classes=num_classes, average="macro", task=task), # does not take label imbalance -> helping in situations where each class's prediction is equally important.
        "AUROC": AUROC(num_classes=num_classes, task=task),
        "Precision": Precision(num_classes=num_classes, task=task),
        "Recall" : Recall(num_classes=num_classes, task=task),
        "F1": F1(num_classes=num_classes, task=task)
    })

    return metrics
    


def get_mil_model(mil_model, num_feats, num_classes, loss_weight=None):
    if mil_model in ["meanpooling", "maxpooling", "ABMIL", "GABMIL"]:
        if mil_model == "meanpooling":
            pooling_layer = nn.AdaptiveAvgPool1d(1)
        elif mil_model == "maxpooling":
            pooling_layer = nn.AdaptiveMaxPool1d(1)
        elif mil_model == "ABMIL":
            from src.mil.model.abmil import AttentionPooling
            pooling_layer = AttentionPooling(num_feats, mid_dim=128, out_dim=1, flatten=True, dropout=0.)
        elif mil_model == "GABMIL":
            from src.mil.model.gabmil import GatedAttentionPooling
            pooling_layer = GatedAttentionPooling(num_feats, mid_dim=128, out_dim=1, flatten=True, dropout=0.)
        
        classifier_model = nn.Sequential(
            switch_dim(),
            pooling_layer,
            nn.Flatten(),
            nn.Linear(num_feats, num_classes)
        )

        loss = nn.CrossEntropyLoss(weight=loss_weight)

    elif mil_model == "DSMIL":
        from src.mil.model.dsmil import FCLayer, BClassifier, MILNet
        i_classifier = FCLayer(in_size=num_feats, out_size=num_classes) # consider all classes as positive (out_size=num_classes)
        b_classifier = BClassifier(input_size=num_feats, output_class=num_classes, dropout_v=0.)
        classifier_model = MILNet(i_classifier, b_classifier)
        # classifier_model.apply(lambda m: apply_sparse_init(m))

        loss = nn.BCEWithLogitsLoss(pos_weight=loss_weight)
    
    elif mil_model in ["CLAM-SB", "CLAM-MB"]:
        from src.mil.model.clam import CLAM_SB, CLAM_MB
        CLAM = CLAM_SB if mil_model == "CLAM-SB" else CLAM_MB
        clam_model_dict = {"dropout": True, "n_classes": num_classes, "subtyping": True, "size_arg": "small",
                           "k_sample": 4, "bag_weight": 0.7, "embed_dim": num_feats}
        classifier_model = CLAM(**clam_model_dict, instance_loss_fn="svm")
        
        loss = nn.CrossEntropyLoss(weight=loss_weight)
    
    elif mil_model == "TransMIL":
        from src.mil.model.transmil import TransMIL
        classifier_model = TransMIL(n_classes=num_classes, input_size=num_feats)

        loss = nn.CrossEntropyLoss(weight=loss_weight)
    
    elif mil_model == "DTFD-MIL":
        from src.mil.model.dtfdmil.network import DimReduction
        from src.mil.model.dtfdmil.attention import Attention_Gated as Attention
        from src.mil.model.dtfdmil.attention import Attention_with_Classifier, Classifier_1fc

        mDim = num_feats // 2

        DTFDclassifier = Classifier_1fc(mDim, num_classes, 0.0)
        DTFDattention = Attention(mDim)
        DTFDdimReduction = DimReduction(num_feats, mDim, numLayer_Res=0)
        DTFDattCls = Attention_with_Classifier(L=mDim, num_cls=num_classes, droprate=0.0)
        classifier_model = [DTFDclassifier, DTFDattention, DTFDdimReduction, DTFDattCls]

        loss0 = nn.CrossEntropyLoss(reduction="none", weight=loss_weight)
        loss1 = nn.CrossEntropyLoss(reduction="none", weight=loss_weight)
        loss = [loss0, loss1]
    else:
        print("Good luck")
        raise NotImplementedError

    return classifier_model, loss


def get_model_module(
        seed, 
        mil_model, 
        num_feats, 
        num_classes, 
        batch_size,
        opt=None,
        lr=None,
        lr_decay_ratio=None,
        weight_decay=None,
        num_epochs=None,
        accumulate_grad_batches=None,
        grad_clipping=None,
        total_instance=None,
        numGroup=None,
        distill=None,
        loss_weight=None):
    from src.mil.pl_model.forward_fn import get_forward_func
    task = "multiclass"
    classifier_model, loss = get_mil_model(mil_model, num_feats, num_classes, loss_weight=loss_weight)
    forward_func = get_forward_func(mil_model)

    if mil_model == "DTFD-MIL":
        from src.mil.pl_model.mil_trainer_dtfdmil import DTFDTrainerModule
        assert accumulate_grad_batches is not None
        assert grad_clipping is not None
        assert total_instance is not None
        assert numGroup is not None
        assert distill is not None
        assert lr_decay_ratio is not None
        trainer_module = DTFDTrainerModule(
            seed=seed,
            classifier_list=classifier_model,
            loss_list=loss,
            epochs=num_epochs,
            lr=lr,
            weight_decay=weight_decay,
            lr_decay_ratio=lr_decay_ratio,
            accumulate_grad_batches=accumulate_grad_batches,
            grad_clipping=grad_clipping,
            metrics=get_metrics(num_classes, task),
            total_instance=total_instance,
            numGroup=numGroup,
            distill=distill

        )

    

    else:
        assert opt is not None
        assert lr is not None
        assert weight_decay is not None
        trainer_module = MILTrainerModule(
            mil_model=mil_model,
            accumulate_grad_batches=accumulate_grad_batches,
            seed=seed,
            classifier=classifier_model,
            loss=loss,
            opt=opt,
            lr=lr,
            batch_size=batch_size,
            weight_decay=weight_decay,
            metrics=get_metrics(num_classes, task),
            num_classes=num_classes,
            num_epochs=num_epochs,
            forward_func=forward_func
        )
                  
    return trainer_module