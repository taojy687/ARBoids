import os
import pandas as pd
import yaml
import time
import torch
from datetime import datetime
from Utils.config import _namespace_to_dict

class ExperimentManager:
    def __init__(
            self, 
            config, 
            base_dir='experiments',
            run_id: str = None,
            repeat_idx: int = None,
            ):
        '''
            - save config + record metrics
            - support multiple repeats
        
            Args:
                config: dict-like (or Namespace that can be converted)
                base_dir: parent experiments folder
                run_id: optional string to identify this run group (if None, auto timestamp)
                repeat_idx: optional integer index for repeats of same config (1,2,..., under same folder)
        '''
        # normalize config dict
        if hasattr(config, "__dict__") or not isinstance(config, dict):
            try:
                config_dict = _namespace_to_dict(config)
            except Exception:
                # fallback: try yaml dump/load
                config_dict = yaml.safe_load(yaml.safe_dump(config))
        else:
            config_dict = config

        self.config = config_dict
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        self.repeat_idx = repeat_idx
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_id = run_id or timestamp

        self.exp_dir = os.path.join(self.base_dir, run_id)
        os.makedirs(self.exp_dir, exist_ok=True)
    
        # CSV/pandas path
        self.config_path = os.path.join(self.exp_dir, "config.yaml")
        self.csv_path = os.path.join(self.exp_dir, 'metrics' + str(repeat_idx) + '.csv')
        self.git_hash = None

        # write files
        print(f"[ExperimentManager] run_id =", run_id, "repeat_idx =", repeat_idx)
        self._save_config()

        # runtime attrs
        self.start_time = time.time()
        self.last_time = None

    # ------------------- IO helpers -------------------
    def _save_config(self,):
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, sort_keys=False)
        print(f"[INFO] config saved -> {self.config_path}")

    def save_model(self, model, filename=None, step=None):
        if filename is None:
            filename = f'model.pth'
        
        base, ext = os.path.splitext(filename)
        if not ext:
            ext = '.pth'
        if self.repeat_idx:
            base = f"{base}{self.repeat_idx}"
        if step is not None:
            base = f"{base}-{step}"
        filename = f"{base}{ext}"

        save_path = os.path.join(self.exp_dir, filename)
        # try model.save else Pytorch save
        try:
            if hasattr(model, "save"):
                model.save(save_path)
            elif isinstance(model, torch.nn.Module):
                torch.save(model.state_dict(), save_path)
            else:
                # fallback: try to pickle
                import pickle
                with open(save_path, "wb") as f:
                    pickle.dump(model, f)
            print(f"[INFO] saved model -> {save_path}")  

        except Exception as e:
            print(f"[WARN] failed to save model: {e}")
    
    def load_model(self, model, filename):
        load_path = os.path.join(self.exp_dir, filename)
        model.load(load_path)
    
    # ------------------- metrics recording -------------------
    def record_metrics(self, **metrics):
        '''
            Automatically save metrics to CSV.
            Usage Example:
                exp.record_metrics(step=step, reward=episode_reward, loss=critic_loss)
        '''

        row = {}
        # merge user metrics (overwrite if same keys)
        for k, v in metrics.items():
            try:
                if hasattr(v, "item"):
                    v = v.item()
            except Exception:
                pass
            row[k] = v

        t = time.time()
        elapsed = t - self.start_time if self.start_time else 0.0

        row['time'] = elapsed
        
        df_new = pd.DataFrame([row])
        if os.path.exists(self.csv_path):
            df_old = pd.read_csv(self.csv_path)
            df_concat = pd.concat([df_old, df_new], ignore_index=True, sort=False)
            df_concat.to_csv(self.csv_path, index=False)
        else:
            df_new.to_csv(self.csv_path, index=False)
        
        print(f"[INFO] record metrics -> {self.csv_path}") 


