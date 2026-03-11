import pandas as pd
import math
import kagglehub
from kagglehub import KaggleDatasetAdapter
from pathlib import Path
from config.ga_config import CONFIGS
import streamlit as st

# Load dataset
# ==========================================
def load_dataset(mode):
    if mode == "Experiment 1 (Kaggle)":
        file_path = "2022_Q1_OR_Utilization.csv"
        df = kagglehub.load_dataset(
            kagglehub.KaggleDatasetAdapter.PANDAS,
            "thedevastator/optimizing-operating-room-utilization",
            file_path,
        )
        return df
    else:
        # สำหรับ Experiment 2 (Anesthesia)
        dataset_folder = Path("data")
        df_file = dataset_folder / "Exp2_Anesthesia_Processed.csv"
        if not df_file.exists():
            print(f"ไม่พบไฟล์: {df_file}")
            return pd.DataFrame() 
        return pd.read_csv(df_file)


def calculate_case_weights(df, service_col, time_col):
    service_avg = df.groupby(service_col)[time_col].mean()
    max_avg = service_avg.max() if not service_avg.empty else 1
    max_booked = df[time_col].max() if not df[time_col].empty else 1
    
    weights = {}
    for idx, row in df.iterrows():
        # 70% จากเวลาที่ใช้ + 30% จากความซับซ้อนเฉลี่ยของแผนกนั้นๆ
        time_score = row[time_col] / max_booked
        comp_score = service_avg.get(row[service_col], 0.5) / max_avg
        weights[idx] = round((0.7 * time_score) + (0.3 * comp_score), 4)
    return weights


def parse_surgeries(df, SLOT_DURATION_MIN, BUFFER_SLOTS, mode):
    """แปลง DataFrame เป็น List of Dict โดยรองรับโครงสร้างที่ Clean แล้ว"""
    if df.empty:
        return []

    df = df.reset_index(drop=True) 
    
    # ดึง Mapping จาก Config
    current_mapping = CONFIGS[mode]["SERVICE_TO_CLUSTER"]

    # เราจึงใช้ชื่อคอลัมน์ชุดเดียวกันได้เลยครับ
    if mode == "Experiment 1 (Kaggle)":
        id_col, service_col, time_col, date_col = 'Encounter ID', 'Service', 'Booked Time (min)', 'Date'
    else:
        id_col = 'Encounter ID'
        service_col = 'Service'
        time_col = 'Booked Time (min)'
        date_col = 'Date'       

    if 'Weight' in df.columns:  
        case_weights = df['Weight'].to_dict()
    else:
        case_weights = calculate_case_weights(df, service_col, time_col)    

    surgeries = []
    for idx, row in df.iterrows():
        try:
            booked = int(row[time_col])
            service_name = str(row[service_col]) if pd.notna(row[service_col]) else "Unknown"
            
            surgeries.append({
                'Index': idx,
                'Encounter ID': int(row[id_col]),
                'Service': service_name,
                'cluster': current_mapping.get(service_name, 'A'), 
                'booked_time': booked,
                'slots_needed': math.ceil(booked / SLOT_DURATION_MIN),
                'buffer_slots': BUFFER_SLOTS,
                'Weight': case_weights.get(idx, 0.4),
                'Original_Date': row.get(date_col, "Unknown")
            })
        except KeyError as e:
            st.error(f"ไม่พบคอลัมน์ {e} ในชุดข้อมูล {mode}")
            return []
            
    return surgeries
