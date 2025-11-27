# src/data_processor.py
import pandas as pd
import math
import kagglehub
from kagglehub import KaggleDatasetAdapter
from config.ga_config import SERVICE_TO_CLUSTER, BUFFER_SLOTS

def load_dataset_kagglehub():
    file_path = "2022_Q1_OR_Utilization.csv"
    try:
        # โค้ดสำหรับดึงข้อมูลจริงจาก Kaggle (ถ้าจำเป็น)
        df = kagglehub.load_dataset(
             KaggleDatasetAdapter.PANDAS,
             "thedevastator/optimizing-operating-room-utilization",
             file_path,
        )
    except Exception:
        # Fallback: ใช้ DataFrame ว่างหากโหลดไม่ได้ หรือใช้ข้อมูลใน data/
        print("Warning: Could not load data from KaggleHub. Ensure offline access or use data/ folder.")
        # สมมติว่ามีการโหลดจาก data/ หรือมีการสร้าง DataFrame ขึ้นมา
        return pd.DataFrame() 

    # กรองข้อมูลสำหรับวันเดียว (01/03/22)
    df_days = df[df['Date'] == '01/03/22']
    return df_days


def parse_surgeries(df, slot_duration):
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
                'or_suite': int(row.get('OR Suite', 0)), # ใช้ .get() เพื่อความปลอดภัย
                'booked_time': booked,
                'slots_needed': slots_needed,
                'buffer_slots': BUFFER_SLOTS
            })
        except (ValueError, KeyError):
            continue # ข้ามแถวที่มีข้อมูลไม่สมบูรณ์
            
    return surgeries
