# trainer.py
import os
import torch
import numpy as np
from typing import Dict
from omegaconf import OmegaConf


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = self.cfg.device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # ✅ Build model
        self.model = self.build_model().to(self.device) if cfg.model.name != "KNN" else None

        # ✅ Build loss function
        self.criterion = self.build_loss().to(self.device) if cfg.model.name != "KNN" else None

        # ✅ Optimizer
        self.optimizer = None
        if cfg.model.name != "KNN":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=cfg.training.lr
            )

        # ✅ Learning rate scheduler
        self.scheduler = None
        if cfg.model.name != "KNN" and cfg.training.lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=cfg.training.patience_lr,
                factor=cfg.training.factor_lr,
                verbose=True
            )

        # Create directories for saving outputs
        os.makedirs(os.path.dirname(cfg.paths.save_path), exist_ok=True)
        os.makedirs(cfg.paths.log_dir, exist_ok=True)

        # ✅ Cache AUX and GCN flags
        self.use_aux = "AUX" in cfg.model.name or cfg.model.name in ["LSTM", "DGCN_LSTM"]
        self.use_gcn = cfg.model.name != "LSTM"  # LSTM does not use GCN; DGCN_LSTM uses GCN
        print(f"Using auxiliary input: {self.use_aux}, Using GCN: {self.use_gcn}")

    def build_model(self):
        from utils import build_model
        return build_model(self.cfg.model)

    def build_loss(self):
        from utils import build_loss
        return build_loss(self.cfg)

    def evaluate(self, y_true, m_mask, a_input, x_input, batch_size: int, x_aux_true=None):
        """
        Evaluate model performance (validation or test).
        Inputs depend on self.use_aux and self.use_gcn.
        Metrics are computed only at positions where m_mask equals 0.
        """
        if self.cfg.model.name == "KNN":
            from models import KNN
            x_input_tensor = torch.from_numpy(x_input).float().to(self.device)
            a_input_tensor = torch.from_numpy(a_input).float().to(self.device)
            m_mask_tensor = torch.from_numpy(m_mask).bool().to(self.device)
            out = KNN(x_input_tensor, a_input_tensor, m_mask_tensor, k=self.cfg.model.params.k, eps=self.cfg.model.params.eps, device=self.device)
            y_target = torch.from_numpy(y_true).float().to(self.device)[~torch.from_numpy(m_mask).bool().to(self.device)]
            out_target = out[~torch.from_numpy(m_mask).bool().to(self.device)]
            preds = [out_target.cpu().numpy()]
            targets = [y_target.cpu().numpy()]
        else:
            self.model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for i in range(0, len(y_true), batch_size):
                    end_idx = min(i + batch_size, len(y_true))
                    batch_x = torch.from_numpy(x_input[i:end_idx]).float().to(self.device)
                    batch_y = torch.from_numpy(y_true[i:end_idx]).float().to(self.device)
                    batch_m = torch.from_numpy(m_mask[i:end_idx]).float().to(self.device)

                    # Conditionally pass inputs
                    if self.use_aux and self.use_gcn:
                        batch_a = torch.from_numpy(a_input[i:end_idx]).float().to(self.device)
                        batch_x_aux = torch.from_numpy(x_aux_true[i:end_idx]).float().to(self.device)
                        out = self.model(batch_x, batch_a, batch_x_aux)
                    elif self.use_aux:
                        batch_x_aux = torch.from_numpy(x_aux_true[i:end_idx]).float().to(self.device)
                        out = self.model(batch_x, batch_x_aux)
                    else:
                        batch_a = torch.from_numpy(a_input[i:end_idx]).float().to(self.device)
                        out = self.model(batch_x, batch_a)

                    # Only compute metrics at positions where mask is 0
                    y_target = batch_y[~batch_m.bool()]
                    out_target = out[~batch_m.bool()]

                    preds.append(out_target.cpu().numpy())
                    targets.append(y_target.cpu().numpy())

        if len(preds) == 0:
            return float('inf'), float('inf'), float('inf')

        p = np.concatenate(preds)
        t = np.concatenate(targets)
        mae = np.mean(np.abs(p - t))
        mape = np.mean(np.abs((t - p) / (np.abs(t) + 1e-8))) * 100
        rmse = np.sqrt(np.mean((p - t) ** 2))
        return mae, mape, rmse

    def train_epoch(self, y_train, m_train, a_train, x_train, batch_size: int, x_aux_train=None):
        """
        Train for one epoch.
        Inputs depend on self.use_aux and self.use_gcn.
        Loss is computed only at positions where the mask equals 0.
        """
        self.model.train()
        total_loss = 0.0
        count = 0
        for i in range(0, len(y_train), batch_size):
            end_idx = min(i + batch_size, len(y_train))
            batch_x = torch.from_numpy(x_train[i:end_idx]).float().to(self.device)
            batch_y = torch.from_numpy(y_train[i:end_idx]).float().to(self.device)
            batch_m = torch.from_numpy(m_train[i:end_idx]).float().to(self.device)

            # Conditionally pass inputs
            if self.use_aux and self.use_gcn:
                batch_a = torch.from_numpy(a_train[i:end_idx]).float().to(self.device)
                batch_x_aux = torch.from_numpy(x_aux_train[i:end_idx]).float().to(self.device)
                out = self.model(batch_x, batch_a, batch_x_aux)
            elif self.use_aux:
                batch_x_aux = torch.from_numpy(x_aux_train[i:end_idx]).float().to(self.device)
                out = self.model(batch_x, batch_x_aux)
            else:
                batch_a = torch.from_numpy(a_train[i:end_idx]).float().to(self.device)
                out = self.model(batch_x, batch_a)

            # Compute loss only at positions where mask is 0
            loss = self.criterion(out[~batch_m.bool()], batch_y[~batch_m.bool()])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            count += 1
        return total_loss / count if count > 0 else float('inf')

    def fit(self, Y_train, Y_test, X_aux_train=None, X_aux_test=None, A=None) -> Dict:
        """
        Full training pipeline.
        Data handling depends on self.use_aux and self.use_gcn.
        """
        print(OmegaConf.to_yaml(self.cfg))
        print("Starting training...")

        from data.preprocess import data_partition, train_construction, test_construction

        # ✅ 1. Data partitioning
        partition_args = {
            'Y_train': Y_train,
            'Y_test': Y_test,
            'unobserved_proportion': self.cfg.data.unobserved_proportion,
            'seed': self.cfg.seed,
            'AUX': self.use_aux
        }
        if self.use_gcn:
            partition_args['A'] = A
        if self.use_aux:
            partition_args['X_aux_train'] = X_aux_train

        result = data_partition(**partition_args)
        if self.use_aux and self.use_gcn:
            Y_train_obs, Y_val_obs, X_aux_train_obs, X_aux_val_obs, X_test, A_obs, M_test, Indices = result
        elif self.use_aux:
            Y_train_obs, Y_val_obs, X_aux_train_obs, X_aux_val_obs, X_test, M_test, Indices = result
            A_obs = None
        else:
            Y_train_obs, Y_val_obs, X_test, A_obs, M_test, Indices = result
            X_aux_train_obs = X_aux_val_obs = X_aux_test = None

        # ✅ 2. Construct test set
        test_args = {
            'sample_size': self.cfg.data.test_sample_size,
            'Y_test': Y_test,
            'X_test': X_test,
            'Indices': Indices,
            'h': self.cfg.data.h,
            'N_sub': self.cfg.data.N_sub,
            'u': self.cfg.data.u,
            'seed': self.cfg.seed,
            'AUX': self.use_aux
        }
        if self.use_gcn:
            test_args['A'] = A
        if self.use_aux:
            test_args['X_aux_test'] = X_aux_test

        test_result = test_construction(**test_args)
        if self.use_aux and self.use_gcn:
            x_test, x_aux_test, m_test, a_test, y_test = test_result
        elif self.use_aux:
            x_test, x_aux_test, m_test, y_test = test_result
            a_test = None
        else:
            x_test, m_test, a_test, y_test = test_result
            x_aux_test = None

        # ✅ 3. Main training loop (KNN skips training)
        history = {'train_loss': [], 'val_mae': [], 'val_mape': [], 'val_rmse': []}
        if self.cfg.model.name != "KNN":
            best_val_mae = float('inf')
            patience = 0

            for epoch in range(self.cfg.training.epochs):
                # --- Resample training data ---
                train_args = {
                    'sample_size': self.cfg.data.train_sample_size,
                    'Y_train_obs': Y_train_obs,
                    'h': self.cfg.data.h,
                    'N_sub': self.cfg.data.N_sub,
                    'u': self.cfg.data.u,
                    'AUX': self.use_aux
                }
                if self.use_gcn:
                    train_args['A_obs'] = A_obs
                if self.use_aux:
                    train_args['X_aux_train_obs'] = X_aux_train_obs

                train_result = train_construction(**train_args)
                if self.use_aux and self.use_gcn:
                    y_train, x_aux_train, m_train, a_train, x_train = train_result
                elif self.use_aux:
                    y_train, x_aux_train, m_train, x_train = train_result
                    a_train = None
                else:
                    y_train, m_train, a_train, x_train = train_result
                    x_aux_train = None

                # --- Resample validation data ---
                val_args = {
                    'sample_size': self.cfg.data.val_sample_size,
                    'Y_train_obs': Y_val_obs,
                    'h': self.cfg.data.h,
                    'N_sub': self.cfg.data.N_sub,
                    'u': self.cfg.data.u,
                    'AUX': self.use_aux
                }
                if self.use_gcn:
                    val_args['A_obs'] = A_obs
                if self.use_aux:
                    val_args['X_aux_train_obs'] = X_aux_val_obs

                val_result = train_construction(**val_args)
                if self.use_aux and self.use_gcn:
                    y_val, x_aux_val, m_val, a_val, x_val = val_result
                elif self.use_aux:
                    y_val, x_aux_val, m_val, x_val = val_result
                    a_val = None
                else:
                    y_val, m_val, a_val, x_val = val_result
                    x_aux_val = None

                # --- Training and evaluation ---
                train_loss = self.train_epoch(
                    y_train, m_train, a_train, x_train,
                    batch_size=self.cfg.training.batch_size,
                    x_aux_train=x_aux_train
                )

                val_mae, val_mape, val_rmse = self.evaluate(
                    y_val, m_val, a_val, x_val,
                    batch_size=self.cfg.training.batch_size,
                    x_aux_true=x_aux_val
                )

                # --- Record history ---
                history['train_loss'].append(train_loss)
                history['val_mae'].append(val_mae)
                history['val_mape'].append(val_mape)
                history['val_rmse'].append(val_rmse)

                if self.scheduler is not None:
                    self.scheduler.step(val_mae)

                # --- Early stopping and checkpointing ---
                if val_mae < best_val_mae - self.cfg.training.min_delta:
                    best_val_mae = val_mae
                    patience = 0
                    torch.save(self.model.state_dict(), self.cfg.paths.save_path)
                    print(f"[Epoch {epoch+1}] ✅ Saved best model | Val MAE: {val_mae:.4f}")
                else:
                    patience += 1

                print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | "
                      f"Val MAE: {val_mae:.4f}, MAPE: {val_mape:.2f}%, RMSE: {val_rmse:.4f} | "
                      f"Patience: {patience}")

                if patience >= self.cfg.training.early_stopping_patience:
                    print("🛑 Early stopping triggered.")
                    break

            # ✅ 4. Load the best model
            self.model.load_state_dict(torch.load(self.cfg.paths.save_path, map_location=self.device))

        else:
            print("✅ KNN model detected, skipping training...")

        # ✅ 5. Final testing
        test_mae, test_mape, test_rmse = self.evaluate(
            y_test, m_test, a_test, x_test,
            batch_size=self.cfg.training.batch_size,
            x_aux_true=x_aux_test
        )
        history['test_mae'] = test_mae
        history['test_mape'] = test_mape
        history['test_rmse'] = test_rmse

        print(f"\n✅ Final Test | MAE: {test_mae:.4f}, MAPE: {test_mape:.2f}%, RMSE: {test_rmse:.4f}")
        return history
