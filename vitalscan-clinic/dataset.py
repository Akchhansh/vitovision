import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MultiTaskRPPGDataset(Dataset):
    def __init__(self, csv_path, data_dir, target_frames=600):
        """
        Advanced Multi-Task PyTorch Dataset for rPPG tracking.
        Extracts raw RGB traces (X_signal) + Context Metrics (X_context)
        to predict ALL 9 clinical targets simultaneously.
        """
        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir
        self.target_frames = target_frames
        
        # 1. Filter out rows where the heavy video file doesn't exist locally
        self.valid_rows = []
        for idx, row in self.df.iterrows():
            video_path = os.path.join(self.data_dir, str(row['video']))
            if os.path.exists(video_path):
                self.valid_rows.append(row)
                
        print(f"📦 Multi-Task Dataset initialized with {len(self.valid_rows)} valid training samples.")
        
        # 2. Establish Min/Max bounds for normalization from the master sheet
        # This prevents large numbers (like Weight=100) from overpowering small numbers (like HbA1c=5.5)
        self.age_min, self.age_max = self.df['age'].min(), self.df['age'].max()
        self.bmi_min, self.bmi_max = self.df['bmi'].min(), self.df['bmi'].max()

    def __len__(self):
        return len(self.valid_rows)

    def _extract_rgb_trace(self, video_path):
        """Extracts 1D color average curves from the center facial zone."""
        cap = cv2.VideoCapture(video_path)
        rgb_traces = []
        frame_count = 0
        
        while cap.isOpened() and frame_count < self.target_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame_rgb.shape
            
            # Crop center 40% region
            top, bottom = int(h * 0.3), int(h * 0.7)
            left, right = int(w * 0.3), int(w * 0.7)
            roi = frame_rgb[top:bottom, left:right]
            
            mean_val = cv2.mean(roi)[:3]
            rgb_traces.append(mean_val)
            frame_count += 1
            
        cap.release()
        
        if len(rgb_traces) == 0:
            rgb_traces = [[0.0, 0.0, 0.0]]
        while len(rgb_traces) < self.target_frames:
            rgb_traces.append(rgb_traces[-1])
            
        signal = np.array(rgb_traces, dtype=np.float32)
        return signal.T  # Shape: (3, 600)

    def __getitem__(self, idx):
        row = self.valid_rows[idx]
        
        # --- 1. FEATURE EXTRACTOR A: Video Signal (Channels, Time) ---
        video_full_path = os.path.join(self.data_dir, row['video'])
        x_signal = self._extract_rgb_trace(video_full_path)
        x_signal_tensor = torch.from_numpy(x_signal)
        
        # --- 2. FEATURE EXTRACTOR B: Patient Context Metadata ---
        # Normalize age and BMI between 0.0 and 1.0
        norm_age = (float(row['age']) - self.age_min) / (self.age_max - self.age_min + 1e-5)
        norm_bmi = (float(row['bmi']) - self.bmi_min) / (self.bmi_max - self.bmi_min + 1e-5)
        
        # Binary Encode Sex (Male = 1, Female = 0)
        is_male = 1.0 if str(row['sex']).strip().upper() == 'M' else 0.0
        
        # One-Hot Encode Exercise State Step ('before', 'after', 'rest')
        step_str = str(row['step']).strip().lower()
        is_before = 1.0 if step_str == 'before' else 0.0
        is_after  = 1.0 if step_str == 'after' else 0.0
        is_rest   = 1.0 if step_str == 'rest' else 0.0
        
        # Combine into a static 1D Context Vector
        x_context = [norm_age, norm_bmi, is_male, is_before, is_after, is_rest]
        x_context_tensor = torch.tensor(x_context, dtype=torch.float32)
        
        # --- 3. ALL 9 CLINICAL TARGET HEADS (Y) ---
        targets = {
            'pulse': torch.tensor(float(row['pulse']), dtype=torch.float32),
            'saturation': torch.tensor(float(row['saturation']), dtype=torch.float32),
            'upper_ap': torch.tensor(float(row['upper_ap']), dtype=torch.float32),
            'lower_ap': torch.tensor(float(row['lower_ap']), dtype=torch.float32),
            'glycated_hemoglobin': torch.tensor(float(row['glycated_hemoglobin']), dtype=torch.float32),
            'hemoglobin': torch.tensor(float(row['hemoglobin']), dtype=torch.float32),
            'cholesterol': torch.tensor(float(row['cholesterol']), dtype=torch.float32),
            'temperature': torch.tensor(float(row['temperature']), dtype=torch.float32),
            'respiratory': torch.tensor(float(row['respiratory']), dtype=torch.float32)
        }
        
        return x_signal_tensor, x_context_tensor, targets

# --- Verification Block ---
if __name__ == "__main__":
    csv_file = "./mcd_rppg_5_samples/db.csv"
    data_directory = "./mcd_rppg_5_samples"
    
    dataset = MultiTaskRPPGDataset(csv_path=csv_file, data_dir=data_directory, target_frames=600)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print("\n⏳ Testing multi-task pipeline output data flow...")
    for batch_signals, batch_contexts, batch_targets in dataloader:
        print("\n✅ Success! DataLoader compiled multi-task batch perfectly.")
        print(f"   • Video Trace Input Shape  : {batch_signals.shape} -> (Batch, RGB_Channels, Time_Frames)")
        print(f"   • Context Vector Input Shape: {batch_contexts.shape} -> (Batch, Metadata_Features)")
        print("\n🎯 Multi-Target Batch Targets Extracted:")
        for key, val in batch_targets.items():
            print(f"     ↳ {key.ljust(20)}: Shape {list(val.shape)} | Samples: {val.tolist()}")
        break