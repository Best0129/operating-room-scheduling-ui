import pandas as pd
import math
import kagglehub
from kagglehub import KaggleDatasetAdapter
from config.ga_config import SERVICE_TO_CLUSTER, BUFFER_SLOTS

def load_dataset_kagglehub():
    file_path = "2022_Q1_OR_Utilization.csv"
    try:
        # ดึงข้อมูลจริงจาก KaggleHub
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "thedevastator/optimizing-operating-room-utilization",
            file_path,
        )
    except Exception:
        print("Warning: Could not load data from KaggleHub. Ensure offline access or use data/ folder.")
        return pd.DataFrame() 

    # กรองข้อมูลสำหรับวันเดียว (01/03/22)
    df_days = df[df['Date'] == '01/03/22']
    return df_days


def parse_surgeries(df, slot_duration, buffer_slots):
    surgeries = []
    for _, row in df.iterrows():
        try:
            booked = int(row['Booked Time (min)'])
            slots_needed = math.ceil(booked / slot_duration)
            service = row['Service']
            cluster = SERVICE_TO_CLUSTER.get(service, None)
            
            if cluster is None:
                continue # ข้ามเคสที่ไม่มี Cluster

            surgeries.append({
                'Encounter ID': int(row['Encounter ID']),
                'service': service,
                'cluster': cluster,
                'or_suite': int(row.get('OR Suite', 0)),
                'booked_time': booked,
                'slots_needed': slots_needed,
                'buffer_slots': buffer_slots
            })
        except (ValueError, KeyError):
            continue # ข้ามแถวที่มีข้อมูลไม่ถูกต้อง
            
    return surgeries
