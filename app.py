import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np 
import time
from collections import defaultdict
import copy

from config.ga_config import *
from src.data_processor import load_dataset, parse_surgeries
from src.utils import slot_to_time, calculate_metrics, evaluate_fitness
from src.algorithms.ga_scheduler import run_ga_standard, run_ga_hybrid_q 
from src.algorithms.st_scheduler import run_ST

# =================================================================
# 1. INITIAL SETUP & CSS
# =================================================================
st.set_page_config(page_title="ระบบจัดตารางเวลาห้องผ่าตัด", layout="wide")

@st.cache_data
def load_data(mode, slot_duration):
    df = load_dataset(mode)
    if df.empty:
        return []
    # ส่ง mode เข้าไปด้วยเพื่อให้จัดการ Weight และ Cluster ได้ถูกต้องตาม Exp
    surgeries = parse_surgeries(df, slot_duration, BUFFER_SLOTS, mode) 
    return surgeries

if 'results' not in st.session_state:
    st.session_state.results = None
    
if 'last_exp' not in st.session_state:
    st.session_state.last_exp = None

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] { width: 380px !important; }
        .main-title { font-size: 36px; font-weight: bold; color: #1E40AF; margin-bottom: 20px; }
        .metric-card { background-color: #F3F4F6; padding: 15px; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🏥 ระบบจำลองการจัดตารางเวลาห้องผ่าตัด</div>', unsafe_allow_html=True)

# =================================================================
# 2. SIDEBAR CONFIGURATION
# =================================================================

with st.sidebar:
    st.title("⚙️ ตั้งค่าระบบ & อัลกอริทึม")
    
    st.header("🧬 พารามิเตอร์ อัลกอริทึม")
    exp_mode = st.selectbox("เลือกการทดลอง", ["Experiment 1 (Kaggle)", "Experiment 2 (Anesthesia)"])
    
    if st.session_state.last_exp != exp_mode:
        st.session_state.results = None
        st.session_state.last_exp = exp_mode

    st.header("⏰ เวลาการทำงาน")
    input_slot_duration = st.number_input("เวลาต่อ 1 Slot (นาที)", 5, 60, value=SLOT_DURATION_MIN, step=5)
    
    col_t1, col_t2 = st.columns(2)
    default_start = datetime.strptime("07:00", "%H:%M").time()
    default_end = datetime.strptime("15:00", "%H:%M").time()
    start_time_input = col_t1.time_input("เริ่มต้น:", value=default_start)
    end_time_input = col_t2.time_input("สิ้นสุด:", value=default_end)

    st.markdown("---")

    
    
    algorithm_selection = st.selectbox(
        "เลือก อัลกอริทึม",
        ("Standard GA", "Hybrid GA-Q-learning", "ST Baseline (Heuristic)")
    )

    is_st_selected = (algorithm_selection == "ST Baseline (Heuristic)")
    
    col_p1, col_p2 = st.columns(2)
    num_generations = col_p1.number_input("Generations", 10, 2000, value=NUM_GENERATIONS, disabled=is_st_selected)
    pop_size = col_p2.number_input("Population", 10, 500, value=POP_SIZE, disabled=is_st_selected)
    
    # ดึงน้ำหนักจาก Config ของแต่ละโหมด
    cfg = CONFIGS[exp_mode]
    
    st.info(f"**Weights Config:**\n"
            f"- Makespan: {W_MAKESPAN}\n"
            f"- Overtime: {W_OVERTIME}\n"
            f"- Balance: {W_IMBALANCE}")
    
    run_button = st.button("เริ่มการประมวลผล", use_container_width=True, type="primary")

# คำนวณ Slot เวลา
dt_start = datetime.combine(datetime.today(), start_time_input)
dt_end = datetime.combine(datetime.today(), end_time_input)
if dt_end <= dt_start: dt_end += timedelta(days=1)
total_slots = int(((dt_end - dt_start).total_seconds() / 60) / input_slot_duration)
operating_time = (dt_start.hour + dt_start.minute/60, dt_end.hour + dt_end.minute/60)

# =================================================================
# 3. EXECUTION LOGIC
# =================================================================

surgeries_list = load_data(exp_mode, input_slot_duration)

if run_button:
    if not surgeries_list:
        st.error("ไม่พบข้อมูล!")
    else:
        start_time_exec = time.time()
        progress_bar = st.progress(0)
        
        with st.spinner(f"กำลังคำนวณด้วย {algorithm_selection}..."):
            all_or_ids = [or_id for ors in cfg["CLUSTER_TO_ORS"].values() for or_id in ors]
            
            # --- แยกรันตามเงื่อนไข (แก้ไขชื่อให้ตรงกับ selectbox) ---
            if algorithm_selection == "ST Baseline (Heuristic)":
                sched, status = run_ST(surgeries_list, total_slots, BUFFER_SLOTS, exp_mode)
                history = [evaluate_fitness(sched, status, total_slots, W_MAKESPAN, W_OVERTIME, W_IMBALANCE)]
                best_ind = {'fitness': history[0]}
                progress_bar.progress(100)
                
            elif algorithm_selection == "Standard GA":
                # เพิ่ม patience=50 ตามหลักการ Jupyter
                best_ind, history, sched, status = run_ga_standard(
                    surgeries_list, num_generations, pop_size, total_slots, exp_mode, 
                    patience=50, st_progress=progress_bar
                )
                
            else: # Hybrid GA-Q-learning
                # เพิ่ม patience=50 ตามหลักการ Jupyter
                best_ind, history, sched, status, stop_gen = run_ga_hybrid_q(
                    surgeries_list, num_generations, pop_size, total_slots, exp_mode, 
                    patience=50, st_progress=progress_bar
                )
            
            runtime = time.time() - start_time_exec
            metrics = calculate_metrics(sched, status, total_slots, input_slot_duration, all_or_ids, surgeries_list, exp_mode)
            metrics['Runtime_Sec'] = round(runtime, 2)
            
            st.session_state.results = {
                'sched': sched, 'history': history, 'metrics': metrics, 
                'algo': algorithm_selection, 'mode': exp_mode,
                'op_time': operating_time, 'slot_dur': input_slot_duration
            }
        st.success(f"✅ ประมวลผลสำเร็จ! ({metrics['Runtime_Sec']} วินาที)")

# =================================================================
# 4. RESULT VISUALIZATION
# =================================================================

if st.session_state.results:
    res = st.session_state.results
    m = res['metrics']
    
    st.subheader(f"📊 ผลการทดลอง: {res['mode']} ด้วยวิธีการ: {res['algo']}")
    
    # แดชบอร์ด Metrics
    row1_c1, row1_c2, row1_c3 = st.columns(3)
    day_diff = m['Total_Days'] - m['Lower_Bound_Days']
    row1_c1.metric("🗓️ จำนวนวันที่ใช้", f"{m['Total_Days']} วัน", delta=f"Lower Bound: {m['Lower_Bound_Days']} วัน", delta_color="inverse")
    row1_c2.metric("🎯 Optimality Gap", f"{m['Optimality_Gap (%)']}%")
    row1_c3.metric("📊 Utilization", f"{m['Global_Util (%)']}%")
    
    row2_c1, row2_c2, row2_c3 = st.columns(3)
    # row2_c1.metric("🕒 Total Overtime", f"{m['Total_OT_Min']} นาที")
    row2_c1.metric("⚡ Runtime", f"{m['Runtime_Sec']} s")
    row2_c2.metric("📉 Penalty Score", f"{m['Penalty_Score']}")

    # ส่วนรายละเอียด
    tab_sched, tab_graph, tab_download = st.tabs(["📅 ตารางการจัดเวลา (Schedule)", "📈 กราฟการเรียนรู้", "📥 บันทึกข้อมูล"])
    
    with tab_sched:
        for day in sorted(res['sched'].keys()):
            with st.expander(f"🗓️ วันที่ {day + 1}", expanded=(day==0)):
                for or_id in sorted(res['sched'][day].keys(), key=lambda x: str(x)):
                    st.markdown(f"**📍 ห้องผ่าตัด: {or_id}**")
                    cases = pd.DataFrame([{
                        "รหัสเคส": c['Encounter ID'], 
                        "แผนก": c['Actual_Dept'],
                        "เทคนิคที่ใช้": c['Service'],
                        "เวลาที่ใช่": c['booked_time'],
                        "เวลาเริ่ม": slot_to_time(c['start_slot'], res['op_time'], res['slot_dur']),
                        "เวลาจบ": slot_to_time(c['end_slot'], res['op_time'], res['slot_dur']),
                        "น้ำหนักเคส": c['Weight']
                    } for c in res['sched'][day][or_id]])
                    
                    st.table(cases)

    with tab_graph:
        if len(res['history']) > 1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(res['history'], label='Fitness (Penalty)', color='#1E40AF', linewidth=2)
            ax.set_title(f"Optimization Progress ({res['algo']})")
            ax.set_xlabel("Generation"); ax.set_ylabel("Penalty"); ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("ST Baseline (Heuristic) คำนวณรอบเดียวไม่มีกราฟ")

    with tab_download:
        export_list = []
        for d, rooms in res['sched'].items():
            for r, cases in rooms.items():
                for c in cases:
                    export_list.append({
                        "Day": d+1, "OR": r, "ID": c['Encounter ID'], "Service": c['Service'],
                        "Start": slot_to_time(c['start_slot'], res['op_time'], res['slot_dur']),
                        "End": slot_to_time(c['end_slot'], res['op_time'], res['slot_dur']),
                        "Weight": c['Weight']
                    })
        st.download_button("📥 Download CSV", pd.DataFrame(export_list).to_csv(index=False).encode('utf-8-sig'), "schedule_results.csv", "text/csv")
