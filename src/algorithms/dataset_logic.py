import pandas as pd
import random
from datetime import datetime, timedelta

def generate_or_utilization_data(num_days=30, num_suites=8):
    # 1. กำหนดข้อมูลพื้นฐาน (อ้างอิงจากรูปแบบไฟล์ต้นฉบับ)
    procedures = [
        {'Service': 'Podiatry', 'CPT': 28110, 'Desc': 'Partial ostectomy, 5th metatarsal', 'Time': 90},
        {'Service': 'Orthopedics', 'CPT': 27445, 'Desc': 'Arthroplasty, knee', 'Time': 120},
        {'Service': 'Ophthalmology', 'CPT': 66982, 'Desc': 'Extracapsular cataract removal', 'Time': 45},
        {'Service': 'OBGYN', 'CPT': 58562, 'Desc': 'Hysterectomy, surgical', 'Time': 120},
        {'Service': 'Urology', 'CPT': 52353, 'Desc': 'Cystourethroscopy', 'Time': 60},
        {'Service': 'General', 'CPT': 47562, 'Desc': 'Laparoscopic cholecystectomy', 'Time': 90},
        {'Service': 'Pediatrics', 'CPT': 69436, 'Desc': 'Tympanostomy', 'Time': 60}
    ]

    data = []
    encounter_id = 15000 # เริ่มต้น ID สำหรับข้อมูลชุดใหม่
    current_date = datetime(2023, 1, 2)  # กำหนดวันที่เริ่มต้น

    for day in range(num_days):
        # ข้ามวันเสาร์-อาทิตย์ เพื่อให้เหมือนการนัดผ่าตัดจริงในวันทำการ
        if current_date.weekday() < 5:
            for suite in range(1, num_suites + 1):
                # เริ่มเคสแรกของวันในแต่ละห้องตอน 7:00 น.
                schedule_time = current_date.replace(hour=7, minute=0)
                
                # จัดตารางผ่าตัดต่อเนื่องไปจนถึงเวลาประมาณ 16:00 น.
                while schedule_time.hour < 16:
                    proc = random.choice(procedures)
                    
                    # คำนวณเวลาต่างๆ โดยการสุ่มช่วงเวลาสั้นๆ เพื่อความสมจริง
                    # Wheels In (คนไข้เข้าห้อง): หลังเวลาจองประมาณ 0-10 นาที
                    wheels_in = schedule_time + timedelta(minutes=random.randint(0, 10))

                    # Start Time (เริ่มผ่าตัด): หลังเข้าห้องประมาณ 15-25 นาที (เตรียมตัว/ดมยา)
                    start_time = wheels_in + timedelta(minutes=random.randint(15, 25))
                    
                    # ระยะเวลาผ่าตัดจริง: สุ่มระหว่าง 80% ถึง 120% ของเวลาที่จองไว้
                    actual_duration = int(proc['Time'] * random.uniform(0.8, 1.2))

                    # End Time (เสร็จสิ้น): เวลาเริ่ม + ระยะเวลาผ่าตัดจริง
                    end_time = start_time + timedelta(minutes=actual_duration)

                    # Wheels Out (คนไข้ ออกจากห้อง): หลังเสร็จสิ้น 5-15 นาที
                    wheels_out = end_time + timedelta(minutes=random.randint(5, 15))
                    
                    # บันทึกข้อมูลลงใน List
                    data.append({
                        'Encounter ID': encounter_id,
                        'Date': current_date.strftime('%m/%d/%y'),
                        'OR Suite': suite,
                        'Service': proc['Service'],
                        'CPT Code': proc['CPT'],
                        'CPT Description': proc['Desc'],
                        'Booked Time (min)': proc['Time'],
                        'OR Schedule': schedule_time.strftime('%m/%d/%y %I:%M %p'),
                        'Wheels In': wheels_in.strftime('%m/%d/%y %I:%M %p'),
                        'Start Time': start_time.strftime('%m/%d/%y %I:%M %p'),
                        'End Time': end_time.strftime('%m/%d/%y %I:%M %p'),
                        'Wheels Out': wheels_out.strftime('%m/%d/%y %I:%M %p')
                    })
                    
                    encounter_id += 1
                    # เคสถัดไปจะเริ่มหลังจากคนเก่าออกไปแล้ว + เวลาทำความสะอาดห้อง (Turn-over) ประมาณ 20-40 นาที
                    next_start = wheels_out + timedelta(minutes=random.randint(20, 40))

                    # ปรับเวลาจองเคสถัดไปให้ลงล็อคทุก 15 นาที เพื่อความสวยงามของตาราง
                    schedule_time = next_start.replace(minute=(next_start.minute // 15) * 15)
        
        # เลื่อนไปวันถัดไป
        current_date += timedelta(days=1)

    return pd.DataFrame(data)

# Execute and save
df_synthetic = generate_or_utilization_data()
df_synthetic.to_csv('Generated_OR_Data.csv', index_label='index')