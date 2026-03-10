import pandas as pd
import math
import kagglehub
from kagglehub import KaggleDatasetAdapter
from config.ga_config import SERVICE_TO_CLUSTER
from pathlib import Path

# Load dataset
# ==========================================
def load_dataset():
    # การทดลองที่1: Kaggle Dataset
    # file_path = "2022_Q1_OR_Utilization.csv"
    # df = kagglehub.load_dataset(
    #     KaggleDatasetAdapter.PANDAS,
    #     "thedevastator/optimizing-operating-room-utilization",
    #     file_path,
    # )

    # การทดลองที่2: Thai Anesthesia Dataset
    dataset_folder = Path("dataset")
    df_file = dataset_folder / "Exp2_Anesthesia_Processed.csv"
    df = pd.read_csv(df_file)
    
    return df


def calculate_case_weights(df):
    service_avg = df.groupby('Service')['Booked Time (min)'].mean()
    max_avg = service_avg.max() if not service_avg.empty else 1
    max_booked = df['Booked Time (min)'].max()
    
    weights = {}
    for idx, row in df.iterrows():
        time_score = row['Booked Time (min)'] / max_booked
        comp_score = service_avg.get(row['Service'], 0.5) / max_avg
        weights[idx] = round((0.7 * time_score) + (0.3 * comp_score), 4)
    return weights


def parse_surgeries(df, SLOT_DURATION_MIN, BUFFER_SLOTS):
    df = df.reset_index(drop=True) 
    if 'Weight' in df.columns:  
      case_weights = df['Weight'].to_dict()
    else:
      case_weights = calculate_case_weights(df)    

    surgeries = []
    for idx, row in df.iterrows():
        booked = int(row['Booked Time (min)'])
        surgeries.append({
            'Index': idx,
            'Encounter ID': int(row['Encounter ID']),
            'Service': row['Service'],
            'cluster': SERVICE_TO_CLUSTER.get(row['Service']),
            'booked_time': booked,
            'slots_needed': math.ceil(booked / SLOT_DURATION_MIN),
            'buffer_slots': BUFFER_SLOTS,
            'Weight': case_weights.get(idx, 0.5),
            'Original_Date': row['Date']
        })
    return surgeries
