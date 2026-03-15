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


def calculate_case_weights(df, service_col='Service', time_col='Booked Time (min)'):
    """
    คำนวณน้ำหนักความสำคัญของเคส (Weight): อิงตามหลักการใน Jupyter Notebook
    ใช้สัดส่วน 70:30 ระหว่างเวลาที่จอง และความซับซ้อนเฉลี่ยของแผนกนั้นๆ
    """
    if df.empty:
        return {}

    # 1. คำนวณค่าเฉลี่ยความซับซ้อนแยกตามแผนก (เหมือน Notebook)
    service_avg = df.groupby(service_col)[time_col].mean()
    max_avg = service_avg.max() if not service_avg.empty else 1
    max_booked = df[time_col].max() if not df.empty else 1
    
    weights = {}
    for idx, row in df.iterrows():
        # 2. คำนวณคะแนน (Normalization) - เหมือน Notebook
        # time_score: สัดส่วนเวลาของเคสนี้เทียบกับเคสที่นานที่สุด
        time_score = row[time_col] / max_booked
        
        # comp_score: สัดส่วนความซับซ้อนเฉลี่ยของแผนกนี้เทียบกับแผนกที่ซับซ้อนที่สุด
        comp_score = service_avg.get(row[service_col], 0.5) / max_avg
        
        # 3. รวมคะแนน (Weighted Sum) ตามสูตรวิจัยของคุณ [cite: 2026-03-12]
        weights[idx] = round((0.7 * time_score) + (0.3 * comp_score), 4)
        
    return weights


def parse_surgeries(df, SLOT_DURATION_MIN, BUFFER_SLOTS, mode):
    """
    แปลง DataFrame ที่ผ่านการ Clean แล้วเป็น List of Dict 
    โดยอิงโครงสร้างข้อมูลตาม Jupyter Notebook [cite: 2026-03-12]
    """
    if df.empty:
        return []

    df = df.reset_index(drop=True) 
    
    # 1. ดึง Mapping จาก Config ตามโหมดการทดลอง
    # (ใน Notebook อาจจะเป็นตัวแปร Global แต่ใน UI เราเก็บแยกตามโหมดใน CONFIGS)
    current_mapping = CONFIGS[mode]["SERVICE_TO_CLUSTER"]

    # 2. จัดการเรื่อง Weight (อิงตามหลักการ Notebook)
    if 'Weight' in df.columns:  
        case_weights = df['Weight'].to_dict()
    else:
        # เรียกใช้ฟังก์ชันคำนวณ Weight ที่เราปรับจูนให้ตรงกับ Notebook แล้ว
        case_weights = calculate_case_weights(df)    

    surgeries = []
    for idx, row in df.iterrows():
        try:
            # 3. ดึงค่าจากคอลัมน์ที่ Clean มาแล้ว (ชื่อคอลัมน์ต้องตรงกับใน Notebook)
            booked = int(row['Booked Time (min)'])

            if mode == "Experiment 2 (Anesthesia)":
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
                # ค้นหา Cluster จาก Mapping ของแต่ละ Experiment
                'cluster': current_mapping.get(logic_service, 'A'), 
                'booked_time': booked,
                'slots_needed': math.ceil(booked / SLOT_DURATION_MIN),
                'buffer_slots': BUFFER_SLOTS,
                # ใช้ 0.5 เป็นค่า Default ตามใน Jupyter Notebook
                'Weight': case_weights.get(idx, 0.5),
                'Original_Date': row.get('Date', "Unknown")
            })
        except KeyError as e:
            # แจ้งเตือนผ่าน UI หากไฟล์ CSV ที่ Clean มามีชื่อคอลัมน์ไม่ตรง
            st.error(f"❌ ไม่พบคอลัมน์ {e} ในชุดข้อมูลที่ Clean มา (Mode: {mode})")
            return []
            
    return surgeries