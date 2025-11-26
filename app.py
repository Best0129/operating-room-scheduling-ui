import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np 

from config.ga_config import *
from src.data_processor import load_dataset_kagglehub, parse_surgeries
from src.utils import slot_to_time
from src.algorithms.ga_scheduler import run_ga 


# --- 1. CACHING DATA LOAD (ต้องรับพารามิเตอร์เวลา) ---
@st.cache_data
def load_data(slot_duration):
    df = load_dataset_kagglehub()
    surgeries = parse_surgeries(df, slot_duration)
    return surgeries

# --- 2. INITIALIZE SESSION STATE ---
if 'ga_results' not in st.session_state:
    st.session_state.ga_results = None
    
# =================================================================
# โครงสร้างหน้าเว็บ STREAMLIT UI LAYOUT
# =================================================================

st.set_page_config(page_title="การจัดตารางเวลาห้องผ่าตัด", layout="wide")
st.title("🏥 การจำลองการจัดตารางเวลาห้องผ่าตัด (GA)")
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 320px !important;
        }
            padding-left: 20px; 
            padding-right: 20px;
        }

    </style>
    """,
    unsafe_allow_html=True,
)
# =================================================================
# --- SIDEBAR: ส่วนควบคุม (UI Controls) ---
with st.sidebar:
    st.header("ตั้งค่าเวลาการทำงานห้องผ่าตัด")
    
    # INPUT 1: SLOT DURATION
    input_slot_duration = st.number_input("เวลาต่อ 1 Slot (นาที)", 5, 30, value=SLOT_DURATION_MIN, step=5)
    
    # INPUT 2 & 3: OPERATING TIME
    st.markdown("##### เวลาการทำงาน (Operating Time)")
    col_start, col_end = st.columns(2)
    
    # แปลง OPERATING_TIME (tuple of floats) ให้เป็น datetime.time สำหรับ input
    default_start = datetime.strptime("07:00", "%H:%M").time()
    default_end = datetime.strptime("15:00", "%H:%M").time()
    
    start_time_input = col_start.time_input("เริ่มต้น:", value=default_start, key='start_time')
    end_time_input = col_end.time_input("สิ้นสุด:", value=default_end, key='end_time')
    
    st.header("GA Parameters")
    num_generations = st.number_input("จำนวนรอบ (Generations)", 10, 1000, value=NUM_GENERATIONS)
    pop_size = st.number_input("ขนาดประชากร (Population Size - N)", 10, 500, value=POP_SIZE)
    
    st.markdown(f"**อัตราการกลายพันธุ์ :** {MUTATION_RATE:.2f}")
    st.markdown(f"**น้ำหนัก Overtime :** {W_OVERTIME:.1f}")
    st.markdown(f"**น้ำหนัก Imbalance :** {W_IMBALANCE:.1f}")
    
    run_button = st.button("เริ่มจำลองการจัดตารางเวลาห้องผ่าตัด")

# แปลง input time เป็น total slots
if start_time_input and end_time_input:
    dt_start = datetime.combine(datetime.today(), start_time_input)
    dt_end = datetime.combine(datetime.today(), end_time_input)
    
    # กรณีข้ามวัน
    if dt_end <= dt_start:
        dt_end += timedelta(days=1)
        
    duration_minutes = (dt_end - dt_start).total_seconds() / 60
    
    # คำนวณ TOTAL_SLOTS ใหม่ และ OPERATING_TIME (ชั่วโมงทศนิยม) ใหม่
    total_slots = int(duration_minutes / input_slot_duration)
    # 7.0, 15.0 ถูกแทนที่ด้วยค่าชั่วโมงจริง
    operating_time = (dt_start.hour + dt_start.minute/60, dt_end.hour + dt_end.minute/60)
else:
    total_slots = TOTAL_SLOTS
    operating_time = OPERATING_TIME
    

# --- 3. DATA LOADING ---
surgeries_list = load_data(input_slot_duration)
st.info(f"เคสผ่าตัดทั้งหมดที่จะใช้ในการจำลอง: {len(surgeries_list)} เคส")


# --- 4. EXECUTION ---
if run_button:
    with st.spinner(f"กำลังรัน Genetic Algorithm รอบที่ {num_generations} ..."):
        
        final_individual, history, OR_schedules, total_used_slots = run_ga(
            surgeries_list, 
            num_generations, 
            pop_size, 
            total_slots,
            # operating_time,
            input_slot_duration
        )
        
    # 💾 เก็บผลลัพธ์ไว้ใน session state
    st.session_state.ga_results = {
        'final_individual': final_individual,
        'history': history,
        'OR_schedules': OR_schedules,
        'total_used_slots': total_used_slots,
        'TOTAL_SLOTS_FINAL': total_slots,
        'OPERATING_TIME_FINAL': operating_time,
        'SLOT_DURATION_FINAL': input_slot_duration
    }
    st.success(f"การจำลองเสร็จสมบูรณ์! Final Best Fitness: {final_individual['fitness']:.4f}")
    # st.experimental_rerun() # ไม่จำเป็นถ้าไม่มี UI ที่ซับซ้อนตามมา


# =================================================================
# 5. การแสดงผลลัพธ์ (RESULTS DISPLAY - แนวตั้ง)
# =================================================================

if st.session_state.ga_results:
    # ดึงค่า Dynamic Time ที่คำนวณและรันแล้วออกมา
    results = st.session_state.ga_results
    final_individual = results['final_individual']
    OR_schedules = results['OR_schedules']
    total_used_slots = results['total_used_slots']
    history = results['history']
    
    TOTAL_SLOTS_FINAL = results['TOTAL_SLOTS_FINAL']
    OPERATING_TIME_FINAL = results['OPERATING_TIME_FINAL']
    SLOT_DURATION_FINAL = results['SLOT_DURATION_FINAL']

    # คำนวณเวลาเริ่มต้นและสิ้นสุดของวันทำงานปกติ
    TOTAL_SLOTS_END_TIME = slot_to_time(TOTAL_SLOTS_FINAL, OPERATING_TIME_FINAL, SLOT_DURATION_FINAL)
    TOTAL_SLOTS_START_TIME = slot_to_time(0, OPERATING_TIME_FINAL, SLOT_DURATION_FINAL)

    st.header("📊 ผลลัพธ์การจำลองการจัดตารางเวลาห้องผ่าตัด")
    # st.markdown(f"**ระยะเวลาในการปฏิบัติงาน:** {TOTAL_SLOTS_START_TIME} ถึง {TOTAL_SLOTS_END_TIME} ({TOTAL_SLOTS_FINAL} slots)")

    # ----------------------------------------------------
    # 🟢 NEW LOGIC: สร้าง MASTER DATAFRAME สำหรับ Download
    # ----------------------------------------------------
    all_case_data = []
    sorted_or_ids = sorted(OR_schedules.keys()) # เรียงลำดับห้องเพื่อความเรียบร้อย

    for or_id in sorted_or_ids:
        schedule = OR_schedules[or_id]
        if schedule:
            for s in schedule:
                start_time = slot_to_time(s['start_slot'], OPERATING_TIME_FINAL, SLOT_DURATION_FINAL)
                end_time = slot_to_time(s['end_slot'], OPERATING_TIME_FINAL, SLOT_DURATION_FINAL)
                
                all_case_data.append({
                    "OR ID": or_id, # เพิ่ม OR ID เข้าไปในตารางดาวน์โหลด
                    "รหัสเคส": s['Encounter ID'],
                    "ประเภทการผ่าตัด": s['Service'],
                    "Start Slot": s['start_slot'],
                    "เวลาเริ่มต้น": start_time,
                    "End Slot": s['end_slot'],
                    "เวลาสิ้นสุด": end_time,
                    "Slots ที่ใช้": s['slots_used'],
                })

    df_master_schedule = pd.DataFrame(all_case_data)

    
    st.markdown("---")
    
    # ----------------------------------------------------
    # 5.1 DETAILED SCHEDULE SECTION (ตารางรายละเอียดเคสทีละ OR)
    # ----------------------------------------------------
    st.subheader("1. ตารางเวลาและรายละเอียดเคสแต่ละห้องผ่าตัด")
    
    # 🎯 การเรียงลำดับห้อง (ใช้ sorted_or_ids ที่กำหนดไว้แล้ว)

    for or_id in sorted_or_ids: 
        schedule = OR_schedules[or_id] 
        
        if schedule:
            final_used = total_used_slots.get(or_id, 0)
            overtime = max(0, final_used - TOTAL_SLOTS_FINAL)
            makespan_time = slot_to_time(final_used, OPERATING_TIME_FINAL, SLOT_DURATION_FINAL)
            
            # แสดงหัวข้อ OR และสรุปผล
            st.markdown(f"##### [ห้องผ่าตัดที่ {or_id}] เวลาเสร็จสิ้น: **{makespan_time}** | Overtime: {overtime} slots")
            
            # สร้าง DataFrame สำหรับตารางรายละเอียดเคส
            case_data = []
            for s in schedule:
                start_time = slot_to_time(s['start_slot'], OPERATING_TIME_FINAL, SLOT_DURATION_FINAL)
                end_time = slot_to_time(s['end_slot'], OPERATING_TIME_FINAL, SLOT_DURATION_FINAL)
                
                case_data.append({
                    "รหัสเคส": s['Encounter ID'],
                    "ประเภทการผ่าตัด": s['Service'],
                    "เวลาเริ่มต้น": f"{start_time} (Slot {s['start_slot']})",
                    "เวลาสิ้นสุด": f"{end_time} (Slot {s['end_slot']})",
                    "Slots ที่ใช้": s['slots_used'],
                })
                
            df_case_details = pd.DataFrame(case_data)
            st.dataframe(df_case_details.set_index('รหัสเคส'), use_container_width=True)
            
    st.markdown("---")

    # 5.2 SUMMARY SECTION (ตารางสรุป Utilization)
    st.subheader("2. สรุปประสิทธิภาพโดยรวมแต่ละห้องผ่าตัด (Utilization & Overtime)")
    
    summary_rows = []
    for or_id, schedule in OR_schedules.items():
        final_used = total_used_slots.get(or_id, 0)
        overtime = max(0, final_used - TOTAL_SLOTS_FINAL) 
        makespan_time = slot_to_time(final_used, OPERATING_TIME_FINAL, SLOT_DURATION_FINAL)
        
        summary_rows.append({
            "ห้องผ่าตัด (OR)": or_id,
            "slots ที่ใช้ทั้งหมด": f"{final_used} ({makespan_time})",
            "Overtime (slots)": overtime,
            "จำนวนเคส": len(schedule)
        })
        
    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(df_summary.set_index('ห้องผ่าตัด (OR)'), use_container_width=True)
    
    st.markdown("---")
    
    # 5.3 CONVERGENCE PLOT (กราฟลู่เข้า)
    st.subheader("3. กราฟแสดงการเปรียบเทียบระหว่างค่า Fitness Score กับคำตอบในแต่ละ Generations")
    df_history = pd.DataFrame(history, columns=['Best Fitness Score'])
    st.line_chart(df_history)


    # ----------------------------------------------------
    # 🟢 ดาวโหลดไฟล์
    # ----------------------------------------------------
    st.markdown("#### ⬇️ ดาวน์โหลดผลลัพธ์การจัดตารางเวลาห้องผ่าตัด")
    
    # แปลง DataFrame เป็น CSV (ไม่เอา index)
    csv = df_master_schedule.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button(
        label="Download Full Schedule as CSV",
        data=csv,
        file_name='optimized_or_schedule_full.csv',
        mime='text/csv',
        key='download_full_schedule_button'
    )
    st.markdown("---")
