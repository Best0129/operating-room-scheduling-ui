import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np 
import time
from collections import defaultdict

from config.ga_config import *
from src.data_processor import load_dataset, parse_surgeries
from src.utils import slot_to_time, calculate_metrics, evaluate_fitness
from src.algorithms.ga_scheduler import run_ga_standard, run_ga_hybrid_q 
from src.algorithms.st_scheduler import run_ST

# --- ฟังก์ชันหลักในการเตรียมข้อมูล ---
@st.cache_data
def load_data(mode, slot_duration):
    df = load_dataset(mode)
    if df.empty:
        return []
    # ส่งพารามิเตอร์ mode เข้าไปด้วยเพื่อให้จัดการ Weight และ Cluster ได้ถูกต้อง
    surgeries = parse_surgeries(df, slot_duration, BUFFER_SLOTS, mode) 
    return surgeries

# จัดการ Session State เพื่อเก็บผลลัพธ์
if 'results' not in st.session_state:
    st.session_state.results = None

# =================================================================
# STREAMLIT UI LAYOUT & CSS
# =================================================================

st.set_page_config(page_title="ระบบจัดตารางเวลาห้องผ่าตัด", layout="wide")

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] { width: 380px !important; }
        .main-title { font-size: 40px; font-weight: bold; color: #1E40AF; margin-bottom: 20px; }
        div.stButton > button {
            display: block; margin-left: auto; margin-right: auto;
            padding: 12px 30px; font-size: 18px; font-weight: bold;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🏥 ระบบจำลองการจัดตารางเวลาห้องผ่าตัด</div>', unsafe_allow_html=True)

# =================================================================
# SIDEBAR: ส่วนตั้งค่าแยกส่วน (Settings & Parameters)
# =================================================================

with st.sidebar:
    st.title("ตั้งค่าเวลาการทำงานห้องผ่าตัด & Algorithm Parameters")
    
    # --- ส่วนที่ 1: ตั้งค่าเวลาการทำงานห้องผ่าตัด ---
    st.header("⏰ ตั้งค่าเวลาการทำงาน")
    with st.container():
        input_slot_duration = st.number_input(
            "เวลาต่อ 1 Slot (นาที)", 
            min_value=5, max_value=60, value=SLOT_DURATION_MIN, step=5
        )
        
        st.write("ช่วงเวลาเปิดทำการ (Operating Hours)")
        col_t1, col_t2 = st.columns(2)
        default_start = datetime.strptime("07:00", "%H:%M").time()
        default_end = datetime.strptime("15:00", "%H:%M").time()
        
        start_time_input = col_t1.time_input("เริ่มต้น:", value=default_start)
        end_time_input = col_t2.time_input("สิ้นสุด:", value=default_end)

    st.markdown("---")

    # --- ส่วนที่ 2: Parameters และ Algorithm ---
    st.header("🧬 พารามิเตอร์ Algorithm")
    with st.container():
        # เลือกชุดข้อมูล
        exp_mode = st.selectbox(
            "เลือกการทดลอง (Dataset)",
            ["Experiment 1 (Kaggle)", "Experiment 2 (Anesthesia)"]
        )
        
        # เลือก Algorithm
        algorithm_selection = st.selectbox(
            "เลือก Algorithm ที่ใช้คำนวณ",
            ("Standard GA", "Hybrid GA-Q-learning", "ST Baseline (Heuristic)")
        )

        is_st_selected = (algorithm_selection == "ST Baseline (Heuristic)")
        
        # ตั้งค่า N และ Generations
        col_p1, col_p2 = st.columns(2)
        num_generations = col_p1.number_input(
          "Generations", 10, 1000, value=100, 
          disabled=is_st_selected,
          help="ST Baseline ไม่มีการใช้ Generations" if is_st_selected else None
        )
        
        pop_size = col_p2.number_input(
          "Population", 10, 500, value=100, 
          disabled=is_st_selected,
          help="ST Baseline ไม่มีการใช้ Population" if is_st_selected else None
        )
        
        # แสดงข้อมูล Weights
        st.caption(f"Weights Config: Overtime({W_OVERTIME}) | Imbalance({W_IMBALANCE}) | Makespan({W_MAKESPAN})")
        
        if algorithm_selection == "Hybrid GA-Q-learning":
            st.info("💡 โหมด Hybrid จะปรับ Mutation Rate อัตโนมัติด้วย Q-Learning")

    st.markdown("<br>", unsafe_allow_html=True)
    run_button = st.button("🚀 เริ่มประมวลผลการจัดตาราง")

# --- ตรรกะการคำนวณเวลา (Dynamic Slot Calculation) ---
if start_time_input and end_time_input:
    dt_start = datetime.combine(datetime.today(), start_time_input)
    dt_end = datetime.combine(datetime.today(), end_time_input)
    
    # กรณีเวลาสิ้นสุดอยู่อีกวัน
    if dt_end <= dt_start:
        dt_end += timedelta(days=1)
        
    duration_minutes = (dt_end - dt_start).total_seconds() / 60
    total_slots = int(duration_minutes / input_slot_duration)
    # เก็บค่า Operating Time เป็นชั่วโมงทศนิยม
    operating_time = (dt_start.hour + dt_start.minute/60, dt_end.hour + dt_end.minute/60)
else:
    total_slots = TOTAL_SLOTS_PER_DAY
    operating_time = OPERATING_TIME

# =================================================================
# ส่วนการรันประมวลผล (Main Logic Execution)
# =================================================================

# โหลดข้อมูล
surgeries_list = load_data(exp_mode, input_slot_duration)

if run_button:
    if not surgeries_list:
        st.error("ไม่พบข้อมูลสำหรับการรัน!")
    else:
        # เริ่มจับเวลา
        start_time_exec = time.time()
        
        progress_bar = st.progress(0)
        with st.spinner(f"กำลังประมวลผลด้วย {algorithm_selection}..."):
            all_or_ids = [or_id for ors in CONFIGS[exp_mode]["CLUSTER_TO_ORS"].values() for or_id in ors]
            
            if algorithm_selection == "ST Baseline (Greedy)":
                sched, status = run_ST(surgeries_list, total_slots, BUFFER_SLOTS, exp_mode)
                history = [evaluate_fitness(sched, status, total_slots, W_MAKESPAN, W_OVERTIME, W_IMBALANCE)]
                best_ind = {'fitness': history[0]}
                progress_bar.progress(100)
            elif algorithm_selection == "Standard GA":
                best_ind, history, sched, status = run_ga_standard(
                    surgeries_list, num_generations, pop_size, total_slots, exp_mode, progress_bar
                )
            else: # Hybrid GA-Q
                best_ind, history, sched, status = run_ga_hybrid_q(
                    surgeries_list, num_generations, pop_size, total_slots, exp_mode, progress_bar
                )
            
            # คำนวณ Runtime [cite: 2026-03-01]
            runtime_seconds = time.time() - start_time_exec
            
            # คำนวณ Metrics พร้อมค่า Lower Bound และ Gap
            metrics = calculate_metrics(sched, status, total_slots, input_slot_duration, all_or_ids, surgeries_list, exp_mode)
            metrics['Runtime_Sec'] = round(runtime_seconds, 2)
            
            st.session_state.results = {
                'sched': sched, 'status': status, 'history': history,
                'metrics': metrics, 'mode': exp_mode, 'algo': algorithm_selection,
                'op_time': operating_time, 'slot_dur': input_slot_duration, 'total_slots': total_slots
            }
        st.success(f"✅ คำนวณเสร็จสมบูรณ์! (ใช้เวลา: {metrics['Runtime_Sec']} วินาที)")

# =================================================================
# ส่วนแสดงผลลัพธ์ (Result Visualization)
# =================================================================

if st.session_state.results:
    res = st.session_state.results
    m = res['metrics']
    
    st.header(f"📊 รายงานผลการทดสอบ: {res['algo']}")
    
    # สร้างการแสดงผล 2 แถวเพื่อให้รองรับ Metrics ที่เพิ่มขึ้น
    row1_c1, row1_c2, row1_c3 = st.columns(3)
    row1_c1.metric("🗓️ จำนวนวันที่ใช้จริง", f"{m['Total_Days']} วัน", delta=f"Lower Bound: {m['Lower_Bound_Days']} วัน", delta_color="inverse")
    row1_c2.metric("🎯 Optimality Gap", f"{m['Optimality_Gap (%)']}%")
    row1_c3.metric("📊 Utilization", f"{m['Global_Util (%)']}%")
    
    row2_c1, row2_c2, row2_c3 = st.columns(3)
    row2_c1.metric("🕒 Total Overtime", f"{m['Total_OT_Min']} นาที")
    row2_c2.metric("⚡ Runtime", f"{m['Runtime_Sec']} s")
    row2_c3.metric("📉 Penalty Score", f"{m['Penalty_Score']}")

    # 2. Tabs สำหรับรายละเอียด
    tab_sched, tab_graph, tab_download = st.tabs(["📅 ตารางการจัดเวลา (Schedule)", "📈 กราฟการเรียนรู้", "📥 บันทึกข้อมูล"])
    
    with tab_sched:
        # แสดงผลแยกรายวัน
        sorted_days = sorted(res['sched'].keys())
        for day in sorted_days:
            with st.expander(f"🗓️ ตารางการผ่าตัด วันที่ {day + 1}", expanded=(day==0)):
                day_sched = res['sched'][day]
                # ใช้ key=lambda x: str(x) เพื่อรองรับชื่อห้องที่เป็นทั้งตัวเลขและข้อความ (เช่น 'จิตเวช')
                for or_id in sorted(day_sched.keys(), key=lambda x: str(x)):
                    st.markdown(f"**📍 ห้องผ่าตัด: {or_id}**")
                    cases = []
                    for c in day_sched[or_id]:
                        cases.append({
                            "รหัสเคส": c['Encounter ID'],
                            "แผนก/เทคนิค": c['Service'],
                            "เวลาเริ่ม": slot_to_time(c['start_slot'], res['op_time'], res['slot_dur']),
                            "เวลาสิ้นสุด": slot_to_time(c['end_slot'], res['op_time'], res['slot_dur']),
                            "ระยะเวลา (Slot)": c['end_slot'] - c['start_slot'],
                            "Weight": c['Weight']
                        })
                    st.table(pd.DataFrame(cases))

    with tab_graph:
        if len(res['history']) > 1:
            st.subheader(f"Convergence Graph: {res['algo']}")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(res['history'], label='Fitness (Penalty)', color='#1E40AF', linewidth=2)
            ax.set_xlabel("Generations")
            ax.set_ylabel("Penalty Value")
            ax.set_title(f"Optimization Progress ({res['mode']})")
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("💡 อัลกอริทึม ST Baseline เป็นวิธีแบบ Greedy (รันครั้งเดียว) จึงไม่มีกราฟประวัติการเรียนรู้")

    with tab_download:
        st.subheader("ดาวน์โหลดรายงานการจัดตาราง")
        all_case_rows = []
        for d in res['sched']:
            for r in res['sched'][d]:
                for c in res['sched'][d][r]:
                    all_case_rows.append({
                        "Day": d + 1,
                        "OR_Room": r,
                        "Encounter_ID": c['Encounter ID'],
                        "Service": c['Service'],
                        "Start_Time": slot_to_time(c['start_slot'], res['op_time'], res['slot_dur']),
                        "End_Time": slot_to_time(c['end_slot'], res['op_time'], res['slot_dur']),
                        "Duration_Slots": c['end_slot'] - c['start_slot'],
                        "Weight": c['Weight']
                    })
        
        df_export = pd.DataFrame(all_case_rows)
        csv_data = df_export.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
        
        st.download_button(
            label="📥 Download Schedule (CSV)",
            data=csv_data,
            file_name=f"or_schedule_{res['mode'].replace(' ', '_')}.csv",
            mime="text/csv"
        )
