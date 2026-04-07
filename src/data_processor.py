import pandas as pd
import math
import kagglehub
from kagglehub import KaggleDatasetAdapter
from pathlib import Path
from config.ga_config import CONFIGS
import streamlit as st

def load_dataset(mode):
    if mode == "Experiment 1 (Kaggle)":
        # สำหรับ Experiment 1 (Kaggle)
        file_path = "2022_Q1_OR_Utilization.csv"
        df = kagglehub.load_dataset(
            kagglehub.KaggleDatasetAdapter.PANDAS,
            "thedevastator/optimizing-operating-room-utilization",
            file_path,
        )
        return df
    elif mode == "Experiment 2 (Anesthesia)":
        # สำหรับ Experiment 2 (Anesthesia)
        dataset_folder = Path("data")
        df_file = dataset_folder / "Exp2_Anesthesia_Processed.csv"
        if not df_file.exists():
            print(f"ไม่พบไฟล์: {df_file}")
            return pd.DataFrame() 
        return pd.read_csv(df_file)
    else:
        # สำหรับ Experiment 3 (Simulated Data)        
        dataset_folder = Path("data")
        df_file = dataset_folder / "Exp3_Simulated_Data.csv"
        if not df_file.exists():
            print(f"ไม่พบไฟล์: {df_file}")
            return pd.DataFrame() 
        return pd.read_csv(df_file)


def calculate_case_weights(df, service_col='Service', time_col='Booked Time (min)'):
    if df.empty:
        return {}

    service_avg = df.groupby(service_col)[time_col].mean()
    max_avg = service_avg.max() if not service_avg.empty else 1
    max_booked = df[time_col].max() if not df.empty else 1
    
    weights = {}
    for idx, row in df.iterrows():
        time_score = row[time_col] / max_booked
        comp_score = service_avg.get(row[service_col], 0.5) / max_avg
        weights[idx] = round((0.7 * time_score) + (0.3 * comp_score), 4)
    return weights


def parse_surgeries(df, SLOT_DURATION_MIN, BUFFER_SLOTS, mode):
    if df.empty:
        return []
    df = df.reset_index(drop=True) 
    current_mapping = CONFIGS[mode]["SERVICE_TO_CLUSTER"]

    if 'Weight' in df.columns:  
        case_weights = df['Weight'].to_dict()
    else:
        case_weights = calculate_case_weights(df)    

    surgeries = []
    for idx, row in df.iterrows():
        try:
            booked = int(row['Booked Time (min)'])

            if mode == "Experiment 2 (Anesthesia)" or mode == "Experiment 3 (Simulated 1 Year)":
                logic_service = str(row['Technique']).strip()
                actual_dept = str(row['Service']).strip()
            else:
                logic_service = str(row['Service']).strip()
                actual_dept = logic_service

            surgeries.append({
                'Index': idx,
                'Encounter ID': int(row['Encounter ID']),
                'Service': logic_service,
                'Actual_Dept': actual_dept,
                'cluster': current_mapping.get(logic_service, 'A'), 
                'booked_time': booked,
                'slots_needed': math.ceil(booked / SLOT_DURATION_MIN),
                'buffer_slots': BUFFER_SLOTS,
                'Weight': case_weights.get(idx, 0.5),
                'Original_Date': row.get('Date', "Unknown")
            })
        except KeyError as e:
            st.error(f"ไม่พบคอลัมน์ {e} ในชุดข้อมูลที่ Clean มา (Mode: {mode})")
            return []
    return surgeries
