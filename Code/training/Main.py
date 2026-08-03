import os
import random
import numpy as np
import torch
import torch as t
import time
import torch.nn.functional as F
import Code.training.Utils.TimeLogger as logger
from Code.training.Utils.TimeLogger import log
from Code.training.Params import args
from Code.training.Model_sparse import Model
from Code.training.DataHandler import DataHandler
from Code.training.DatasetConfig import get_checkpoint_dir, get_default_checkpoint
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    average_precision_score, confusion_matrix,
    roc_auc_score, f1_score
)

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(TRAINING_DIR)


def set_seed(seed):
    print("Set seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
        t.backends.cudnn.benchmark = False
        t.backends.cudnn.deterministic = True


class Coach:
    def __init__(self, handler):
        self.handler = handler
        drug_cfg = {'d_atom': handler.drug_batch.x.shape[1], 'd_model': args.latdim, 'dropout': 0.1,
                    'num_total_atoms': handler.drug_batch.x.shape[0]}
        omics_cfg = {
            'omics_dims': [handler.omics_inputs_all[i].shape[1] for i in range(len(handler.omics_inputs_all))],
            'num_cells': handler.omics_inputs_all[0].shape[0],
            'hidden_dim': 128,
            'proj_dim': args.latdim
        }
        base_model = Model(
            drug_encoder_cfg=drug_cfg,
            omics_encoder_cfg=omics_cfg,
        )
        self.model = base_model
        self.best_auc = 0.0

    def makePrint(self, name, ep, reses):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        stloc = 0
        log('Model Initialized')

        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses))
            if tstFlag:
                test_reses = self.testEpoch()
                log(self.makePrint('Test', ep, test_reses))
                auc = test_reses.get('AUC', None)
                if auc is not None and auc > self.best_auc:
                    self.best_auc = auc
                    self.save_model('best_auc')
                    log(f'New best AUC {auc:.4f} at epoch {ep}, model saved.')
        final_res = self.testEpoch()
        log(self.makePrint('Test', args.epoch, final_res))
        final_auc = final_res.get('AUC', 0.0)
        if final_auc > self.best_auc:
            self.best_auc = final_auc
            self.save_model('best_auc')
            log(f'New best AUC {final_auc:.4f} at final test, model saved.')
        iter_id = time.strftime("%Y%m%d-%H%M%S")
        self.save_model(iter_id)

    def prepareModel(self):
        self.model = self.model.to(args.device)
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.reg)

    def trainEpoch(self):
        self.model.train()

        drugs = self.handler.train_d
        cells = self.handler.train_c
        labels = self.handler.train_y

        ceLoss, sslLoss, globalLoss, localLoss = self.model.calcLosses(
            drugs, cells, labels,
            self.handler.torchBiAdj, args.keepRate,
            drug_batch=self.handler.drug_batch,
            omics_inputs=self.handler.omics_inputs_all
        )
        weighted_ssl = sslLoss * args.ssl_reg
        weighted_global = globalLoss * args.global_cl_reg
        weighted_local = localLoss * args.local_cl_reg
        loss = ceLoss + weighted_ssl + weighted_global + weighted_local

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        ret = {'Loss': loss.item(), 'CELoss': ceLoss.item(),
               'SSLoss': weighted_ssl.item(), 'Global': weighted_global.item(), 'Local': weighted_local.item()}
        return ret

    def testEpoch(self):
        self.model.eval()
        with t.no_grad():
            drugs = self.handler.test_d
            cells = self.handler.test_c
            labels = self.handler.test_y

            logits = self.model.predict(
                self.handler.torchBiAdj, drugs, cells,
                drug_batch=self.handler.drug_batch,
                omics_inputs=self.handler.omics_inputs_all
            )
            probs = F.softmax(logits, dim=1)[:, 1]

            all_labels = labels.cpu().numpy()
            all_probs = probs.cpu().numpy()

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        best_thr = 0.5
        preds = (all_probs > best_thr).astype(int)
        acc = accuracy_score(all_labels, preds)
        precision = precision_score(all_labels, preds, pos_label=1, zero_division=0)
        recall = recall_score(all_labels, preds, pos_label=1, zero_division=0)
        pr_auc = average_precision_score(all_labels, all_probs)
        cm = confusion_matrix(all_labels, preds)
        roc_auc = roc_auc_score(all_labels, all_probs)
        f1 = f1_score(all_labels, preds, pos_label=1, zero_division=0)
        ret = {'Acc': acc, 'precision': precision, 'recall': recall, 'AUPR': pr_auc, 'AUC': roc_auc, 'F1': f1}
        for k, v in ret.items():
            print(f"{k}: {v:.4f}")
        print("\nConfusion Matrix (TN FP / FN TP):")
        print(cm)
        return ret

    def loadModel(self):
        map_loc = args.device if isinstance(args.device, torch.device) else torch.device(args.device)
        save_path = get_default_checkpoint(
            CODE_ROOT, args.dataset, args.checkpoint_dir
        )
        state = torch.load(save_path, map_location=map_loc)
        self.model.load_state_dict(state)
        log('Model Loaded')

    def save_model(self, model_path):
        model_parent_path = get_checkpoint_dir(
            CODE_ROOT, args.dataset, args.checkpoint_dir
        )
        os.makedirs(model_parent_path, exist_ok=True)
        save_path = f'{model_parent_path}/{model_path}_model.pkl'
        torch.save(self.model.state_dict(), save_path)


if __name__ == '__main__':
    use_cuda = args.gpu >= 0 and t.cuda.is_available()
    device = t.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
    args.device = device

    logger.saveDefault = True
    log(f'Start dataset={args.dataset}, device={args.device}')

    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    set_seed(args.seed)
    coach = Coach(handler)
    coach.run()
