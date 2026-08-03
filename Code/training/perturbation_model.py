import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
import copy
from tqdm import tqdm  # 引入进度条库

from Code.training.DataHandler import DataHandler
from Code.training.DatasetConfig import get_checkpoint_dir, get_default_checkpoint
from Code.training.Model_sparse import Model
from Code.training.Params import args
from Code.training.Utils.TimeLogger import log
from sklearn.metrics import roc_auc_score


TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.dirname(TRAINING_DIR)
NOISE_STD = 0.0001
EPSILON_DIFF = 0.05
IS_STRICT_LABELS = False 
TARGET_MODEL_COUNT = 1  
MODEL_DIR = get_checkpoint_dir(CODE_ROOT, args.dataset, args.checkpoint_dir)
REF_MODEL_PATH = get_default_checkpoint(CODE_ROOT, args.dataset, args.checkpoint_dir)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class StabilitySearcher:
    def __init__(self):
        print("Initializing DataHandler...")
        self.handler = DataHandler()
        self.handler.LoadData()

        self.current_drug_noise = None
        self.current_omics_noise = None

        self.ref_model = self.build_model()
        self.load_ref_model()
        print("Calculating reference predictions on clean data...")
        self.ref_probs, self.ref_labels = self.get_training_predictions(self.ref_model)

    def build_model(self):
        """构建一个新的模型实例"""
        drug_cfg = {'d_atom': self.handler.drug_batch.x.shape[1], 'd_model': args.latdim, 'dropout': 0.1,
                    'num_total_atoms': self.handler.drug_batch.x.shape[0]}
        omics_cfg = {
            'omics_dims': [self.handler.omics_inputs_all[i].shape[1] for i in
                           range(len(self.handler.omics_inputs_all))],
            'num_cells': self.handler.omics_inputs_all[0].shape[0],
            'hidden_dim': 128,
            'proj_dim': args.latdim
        }
        return Model(drug_encoder_cfg=drug_cfg, omics_encoder_cfg=omics_cfg).to(args.device)

    def load_ref_model(self):
        if not os.path.exists(REF_MODEL_PATH):
            raise FileNotFoundError(f"Reference model not found at {REF_MODEL_PATH}. Please run Main.py first.")
        print(f"Loading reference model from {REF_MODEL_PATH}")
        state = torch.load(REF_MODEL_PATH, map_location=args.device)
        self.ref_model.load_state_dict(state)
        self.ref_model.eval()

    def get_training_predictions(self, model):
        model.eval()
        with torch.no_grad():
            drugs = self.handler.train_d
            cells = self.handler.train_c

            logits = model.predict(
                self.handler.torchBiAdj, drugs, cells,
                drug_batch=self.handler.drug_batch,
                omics_inputs=self.handler.omics_inputs_all
            )
            probs = F.softmax(logits, dim=1)[:, 1]
            probs_np = probs.cpu().numpy()
            preds_np = (probs_np > 0.5).astype(int)

        return probs_np, preds_np

    def refresh_noise(self):

        num_drugs = self.handler.drug_batch.num_graphs
        num_cells = self.handler.omics_inputs_all[0].shape[0]

        self.current_drug_noise = torch.randn(num_drugs, args.latdim).to(args.device) * NOISE_STD
        self.current_omics_noise = torch.randn(num_cells, args.latdim).to(args.device) * NOISE_STD

    def apply_noise_hook(self, model):
        if self.current_drug_noise is None or self.current_omics_noise is None:
            return

        def drug_noise_hook(module, input, output):
            return output + self.current_drug_noise
        
        def omics_noise_hook(module, input, output):
            return output + self.current_omics_noise

        if model.drug_proj is not None:
            model.drug_proj.register_forward_hook(drug_noise_hook)
        elif model.drug_encoder is not None:
            model.drug_encoder.register_forward_hook(drug_noise_hook)

        # 注册到 Omics 分支
        if model.omics_proj is not None:
            model.omics_proj.register_forward_hook(omics_noise_hook)
        elif model.omics_encoder is not None:
            model.omics_encoder.register_forward_hook(omics_noise_hook)

    def run_training_round(self, round_idx):

        print(f"  > Start Training Round {round_idx}...")
        model = self.build_model()

        self.apply_noise_hook(model)
        
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)

        best_auc = 0.0
        best_state = None

        pbar = tqdm(range(args.epoch), desc=f"Round {round_idx}", unit="epoch")

        for ep in pbar:
            model.train()
            drugs = self.handler.train_d
            cells = self.handler.train_c
            labels = self.handler.train_y

            ceLoss, sslLoss, globalLoss, localLoss = model.calcLosses(
                drugs, cells, labels,
                self.handler.torchBiAdj, args.keepRate,
                drug_batch=self.handler.drug_batch,
                omics_inputs=self.handler.omics_inputs_all
            )

            loss = ceLoss + sslLoss * args.ssl_reg + globalLoss * args.global_cl_reg + localLoss * args.local_cl_reg

            opt.zero_grad()
            loss.backward()
            opt.step()

            if ep % args.tstEpoch == 0:
                cur_auc = self.evaluate_test_auc(model)
                if cur_auc > best_auc:
                    best_auc = cur_auc
                    best_state = copy.deepcopy(model.state_dict())


                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Best_AUC': f'{best_auc:.4f}'})

        pbar.close()
        return best_state

    def evaluate_test_auc(self, model):
        #
        model.eval()
        with torch.no_grad():
            drugs = self.handler.test_d
            cells = self.handler.test_c
            labels = self.handler.test_y.cpu().numpy()
            logits = model.predict(
                self.handler.torchBiAdj, drugs, cells,
                drug_batch=self.handler.drug_batch,
                omics_inputs=self.handler.omics_inputs_all
            )
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            try:
                if len(np.unique(labels)) < 2:
                    return 0.0
                return roc_auc_score(labels, probs)
            except:
                return 0.0

    def search(self):
        found_count = 0
        search_iter = 0

        print("\n=== Starting Stability Search Loop ===")
        print(f"Conditions: Max Diff < {EPSILON_DIFF}, Labels Identical = {IS_STRICT_LABELS}")

        while found_count < TARGET_MODEL_COUNT:
            search_iter += 1
            print(f"\n--- Iteration {search_iter} ---")

            self.refresh_noise()

            best_state = self.run_training_round(search_iter)

            if best_state is None:
                print("  > Warning: Training failed to produce a valid model.")
                continue

            temp_model = self.build_model()
            self.apply_noise_hook(temp_model)
            temp_model.load_state_dict(best_state)

            current_probs, current_labels = self.get_training_predictions(temp_model)

            diff = np.abs(current_probs - self.ref_probs)
            mean_diff = np.mean(diff)
            labels_match = np.all(current_labels == self.ref_labels)

            print(f"  > Validation Result:")
            print(f"    Mean Probability Diff: {mean_diff:.6f} (Limit: {EPSILON_DIFF})")
            print(f"    Labels Identical    : {'YES' if labels_match else 'NO'}")

            is_pass = (mean_diff < EPSILON_DIFF)

            if is_pass:
                found_count += 1
                save_name = f'stable_model_{found_count}.pkl'
                save_path = os.path.join(MODEL_DIR, save_name)
                torch.save(best_state, save_path)
                print(f">>> [SUCCESS] Found Stable Model {found_count}/{TARGET_MODEL_COUNT}. Saved to {save_path}")
            else:
                print("  > [FAILED] Model discarded. Retrying with new noise...")


if __name__ == '__main__':
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    args.device = torch.device(f'cuda:{args.gpu}' if use_cuda else 'cpu')
    set_seed(args.seed)
    os.makedirs(MODEL_DIR, exist_ok=True)

    searcher = StabilitySearcher()
    searcher.search()
