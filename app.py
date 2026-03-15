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

# =================================================================
# 1. INITIAL SETUP & CSS
# =================================================================
st.set_page_config(page_title="ระบบจัดตารางเวลาห้องผ่าตัด", layout="wide")

@st.cache_data
def load_data(mode, slot_duration):
    df = load_dataset(mode)
    if df.empty:
        return []
    surgeries = parse_surgeries(df, slot_duration, BUFFER_SLOTS, mode) 
    return surgeries

if 'results' not in st.session_state:
    st.session_state.results = {} # เปลี่ยนเป็น dict เก็บ 3 อัลกอริทึม
    
if 'last_exp' not in st.session_state:
    st.session_state.last_exp = None

st.markdown(
    """
    <style>

    section[data-testid="stSidebar"] { 
        width: 410px !important; 
    }

    .main-title { 
        font-size: 36px; 
        font-weight: bold; 
        color: #1E40AF; 
        margin-bottom: 20px; 
    }

    .metric-card { 
        background-color: #F3F4F6; 
        padding: 15px; 
        border-radius: 10px; 
    }

    div.stButton > button[kind="primary"] {
        font-weight: 600;
        transition: all 0.2s ease;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🏥 ระบบจำลองการจัดตารางเวลาห้องผ่าตัด</div>', unsafe_allow_html=True)

# =================================================================
# 2. SIDEBAR CONFIGURATION
# =================================================================

with st.sidebar:
    st.title("ตั้งค่าระบบ & การทดลอง")
    
    st.header("เลือกชุดข้อมูลการทดลอง")
    exp_mode = st.selectbox("เลือกการทดลอง", [
        "Experiment 1 (Kaggle)", 
        "Experiment 2 (Anesthesia)",
        "Experiment 3 (Simulated 1 Year)"
    ])
    
    if st.session_state.last_exp != exp_mode:
        st.session_state.results = {}
        st.session_state.last_exp = exp_mode

    st.header("เวลาการทำงาน")
    input_slot_duration = st.number_input("เวลาต่อ 1 Slot (นาที)", 5, 60, value=SLOT_DURATION_MIN, step=5)
    
    col_t1, col_t2 = st.columns(2)
    default_start = datetime.strptime("07:00", "%H:%M").time()
    default_end = datetime.strptime("15:00", "%H:%M").time()
    start_time_input = col_t1.time_input("เวลาเริ่ม:", value=default_start)
    end_time_input = col_t2.time_input("เวลาสิ้นสุด:", value=default_end)

    st.markdown("---")

    st.header("พารามิเตอร์ GA & Q-Learning")
    col_p1, col_p2 = st.columns(2)
    num_generations = col_p1.number_input("จำนวนรอบการคำนวณ", 10, 2000, value=NUM_GENERATIONS)
    pop_size = col_p2.number_input("ขนาดของประชากร", 10, 500, value=POP_SIZE)
    
    cfg = CONFIGS.get(exp_mode, CONFIGS.get("Experiment 1 (Kaggle)", {}))
    st.info(
      f"**ค่าน้ำหนักของตัวชี้วัด**\n"
      f"- ระยะเวลาการใช้ห้องผ่าตัดรวม (Makespan): {W_MAKESPAN}\n"
      f"- เวลาทำงานล่วงเวลา (Overtime): {W_OVERTIME}\n"
      f"- ความไม่สมดุลของภาระงาน (Imbalance): {W_IMBALANCE}"
    )
    
    run_button = st.button("เริ่มการประมวลผล", use_container_width=True, type="primary")

# คำนวณ Slot เวลา
dt_start = datetime.combine(datetime.today(), start_time_input)
dt_end = datetime.combine(datetime.today(), end_time_input)
if dt_end <= dt_start: dt_end += timedelta(days=1)
total_slots = int(((dt_end - dt_start).total_seconds() / 60) / input_slot_duration)
operating_time = (dt_start.hour + dt_start.minute/60, dt_end.hour + dt_end.minute/60)
total_avail_minutes_per_day = total_slots * input_slot_duration

# =================================================================
# 3 & 4. EXECUTION & VISUALIZATION
# =================================================================

surgeries_list = load_data(exp_mode, input_slot_duration)

if run_button:
    if not surgeries_list:
        st.error("ไม่พบข้อมูล!")
        st.stop()
    else:
        # เคลียร์ผลลัพธ์เก่าทิ้งเมื่อกดรันใหม่
        st.session_state.results = {}

# หากมีการกดรัน หรือมีผลลัพธ์เก่าค้างอยู่ ให้แสดงหน้าต่างแสดงผล
if run_button or st.session_state.results:
    st.header(f"ผลการทดลอง: {exp_mode}")
    st.markdown("---")
    
    all_or_ids = list(set(str(or_id).strip() for ors in cfg.get("CLUSTER_TO_ORS", {}).values() for or_id in ors))

    # เตรียมกล่อง Container 3 กล่องสำหรับ 3 อัลกอริทึมรอไว้แต่แรก
    container_st = st.container()
    container_ga = st.container()
    container_q = st.container()

    # ---------------------------------------------------------
    # 1. ST Baseline (Heuristic)
    # ---------------------------------------------------------
    with container_st:
        st.subheader("1. ST Baseline (Heuristic)")
        
        # ถ้ารันอยู่ ให้ทำงาน
        if run_button:
            with st.spinner("กำลังจัดตารางด้วย ST Baseline..."):
                start_t = time.time()
                sched_st, status_st = run_ST(surgeries_list, total_slots, BUFFER_SLOTS, exp_mode)
                hist_st = [evaluate_fitness(sched_st, status_st, total_slots, W_MAKESPAN, W_OVERTIME, W_IMBALANCE)]
                met_st = calculate_metrics(sched_st, status_st, total_slots, input_slot_duration, all_or_ids, surgeries_list, exp_mode)
                met_st['Runtime_Sec'] = round(time.time() - start_t, 2)
                st.session_state.results["ST Baseline"] = {'sched': sched_st, 'history': hist_st, 'metrics': met_st}
        
        # แสดงผลทันทีเมื่อมีข้อมูลใน Session State
        if "ST Baseline" in st.session_state.results:
            m = st.session_state.results["ST Baseline"]['metrics']
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("🗓️ จำนวนวันที่ใช้", f"{m.get('Total_Days', m.get('Total_Days_Used', 0))} วัน", delta=f"Lower Bound: {m.get('Lower_Bound_Days', 0)} วัน", delta_color="inverse")
            c2.metric("🎯 Optimality Gap", f"{m.get('Optimality_Gap (%)', 0)}%")
            c3.metric("📊 Utilization", f"{m.get('Global_Util (%)', m.get('Global_Utilization (%)', 0))}%")
            c4.metric("⚡ Runtime", f"{m.get('Runtime_Sec', 0)} s")
            c5.metric("📉 Penalty Score", f"{m.get('Penalty_Score', 0)}")
            
            # st.info("ST Baseline คำนวณรอบเดียวไม่มีกราฟการเรียนรู้")
            st.caption("* ST Baseline (Heuristic) เป็นการคำนวณแบบรอบเดียว ไม่มีกราฟการเรียนรู้")
        st.markdown("---")

    # ---------------------------------------------------------
    # 2. Standard GA
    # ---------------------------------------------------------
    with container_ga:
        st.subheader("2. Standard GA")
        # สร้างพื้นที่เปล่ารอก่อน เพื่อให้ UI ไม่กระโดด
        metrics_ph_ga = st.empty()
        chart_ph_ga = st.empty()

        if run_button:
            with st.spinner("กำลังจัดตารางด้วย Standard GA..."):
                start_t = time.time()
                # วาดกราฟสดลงใน chart_ph_ga
                _, hist_ga, sched_ga, status_ga = run_ga_standard(
                    surgeries_list, num_generations, pop_size, total_slots, exp_mode, 
                    patience=50, st_progress=None, chart_placeholder=chart_ph_ga
                )
                met_ga = calculate_metrics(sched_ga, status_ga, total_slots, input_slot_duration, all_or_ids, surgeries_list, exp_mode)
                met_ga['Runtime_Sec'] = round(time.time() - start_t, 2)
                st.session_state.results["Standard GA"] = {'sched': sched_ga, 'history': hist_ga, 'metrics': met_ga}

        if "Standard GA" in st.session_state.results:
            m = st.session_state.results["Standard GA"]['metrics']
            hist = st.session_state.results["Standard GA"]['history']
            
            with metrics_ph_ga.container():
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("🗓️ จำนวนวันที่ใช้", f"{m.get('Total_Days', m.get('Total_Days_Used', 0))} วัน", delta=f"Lower Bound: {m.get('Lower_Bound_Days', 0)} วัน", delta_color="inverse")
                c2.metric("🎯 Optimality Gap", f"{m.get('Optimality_Gap (%)', 0)}%")
                c3.metric("📊 Utilization", f"{m.get('Global_Util (%)', m.get('Global_Utilization (%)', 0))}%")
                c4.metric("⚡ Runtime", f"{m.get('Runtime_Sec', 0)} s")
                c5.metric("📉 Penalty Score", f"{round(m.get('Penalty_Score', 0), 4)}")

            # แปลง Live chart (ที่วาดไว้ตอนรัน) ให้กลายเป็น Matplotlib ที่สวยงาม ทับลงไปที่เดิม
            if len(hist) > 1:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(hist, label='Fitness (Penalty)', color='#1E40AF', linewidth=2)
                ax.set_title("Optimization Progress (Standard GA)")
                ax.set_xlabel("Generation"); ax.set_ylabel("Penalty"); ax.grid(True, alpha=0.3)
                chart_ph_ga.pyplot(fig)
        st.markdown("---")

    # ---------------------------------------------------------
    # 3. Hybrid GA-Q
    # ---------------------------------------------------------
    with container_q:
        st.subheader("3. Hybrid GA-Q")
        metrics_ph_q = st.empty()
        chart_ph_q = st.empty()

        if run_button:
            with st.spinner("กำลังจัดตารางด้วย Hybrid GA-Q-learning..."):
                start_t = time.time()
                _, hist_q, sched_q, status_q, _ = run_ga_hybrid_q(
                    surgeries_list, num_generations, pop_size, total_slots, exp_mode, 
                    patience=50, st_progress=None, chart_placeholder=chart_ph_q
                )
                met_q = calculate_metrics(sched_q, status_q, total_slots, input_slot_duration, all_or_ids, surgeries_list, exp_mode)
                met_q['Runtime_Sec'] = round(time.time() - start_t, 2)
                st.session_state.results["Hybrid GA-Q"] = {'sched': sched_q, 'history': hist_q, 'metrics': met_q}

        if "Hybrid GA-Q" in st.session_state.results:
            m = st.session_state.results["Hybrid GA-Q"]['metrics']
            hist = st.session_state.results["Hybrid GA-Q"]['history']
            
            with metrics_ph_q.container():
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("🗓️ จำนวนวันที่ใช้", f"{m.get('Total_Days', m.get('Total_Days_Used', 0))} วัน", delta=f"Lower Bound: {m.get('Lower_Bound_Days', 0)} วัน", delta_color="inverse")
                c2.metric("🎯 Optimality Gap", f"{m.get('Optimality_Gap (%)', 0)}%")
                c3.metric("📊 Utilization", f"{m.get('Global_Util (%)', m.get('Global_Utilization (%)', 0))}%")
                c4.metric("⚡ Runtime", f"{m.get('Runtime_Sec', 0)} s")
                c5.metric("📉 Penalty Score", f"{round(m.get('Penalty_Score', 0), 4)}")

            if len(hist) > 1:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(hist, label='Fitness (Penalty)', color='#10B981', linewidth=2) # ใช้สีเขียวแยกให้ชัดเจน
                ax.set_title("Optimization Progress (Hybrid GA-Q)")
                ax.set_xlabel("Generation"); ax.set_ylabel("Penalty"); ax.grid(True, alpha=0.3)
                chart_ph_q.pyplot(fig)
        st.markdown("---")

    # ---------------------------------------------------------
    # สรุปผลการเปรียบเทียบรวมทั้ง 3 อัลกอริทึม
    # ---------------------------------------------------------
    if all(algo in st.session_state.results for algo in ["ST Baseline", "Standard GA", "Hybrid GA-Q"]):
        res = st.session_state.results
        
        st.subheader("สรุปผลการเปรียบเทียบรวมทั้ง 3 อัลกอริทึม")
        
        # --- ตารางเปรียบเทียบ ---
        compare_data = []
        for algo in ["ST Baseline", "Standard GA", "Hybrid GA-Q"]:
            m = res[algo]['metrics']
            compare_data.append({
                "Algorithm": algo,
                "วันที่ใช้ (Days)": m.get('Total_Days', m.get('Total_Days_Used', 0)),
                "Optimality Gap (%)": m.get('Optimality_Gap (%)', 0),
                "Utilization (%)": m.get('Global_Util (%)', m.get('Global_Utilization (%)', 0)),
                "Runtime (s)": m.get('Runtime_Sec', 0),
                "Penalty Score": round(m.get('Penalty_Score', 0), 4)
            })
        
        df_compare = pd.DataFrame(compare_data)
        st.dataframe(
            df_compare.style.highlight_min(subset=["วันที่ใช้ (Days)", "Optimality Gap (%)", "Runtime (s)", "Penalty Score"], color='#D1FAE5')
                            .highlight_max(subset=["Utilization (%)"], color='#D1FAE5'),
            use_container_width=True, hide_index=True
        )

        # --- กราฟเปรียบเทียบรวม ---
        # st.markdown("<br><b>กราฟเปรียบเทียบการลู่เข้าหาคำตอบที่ดีที่สุด (Combined Convergence)</b>", unsafe_allow_html=True)
        st.subheader("กราฟเปรียบเทียบการลู่เข้าหาคำตอบที่ดีที่สุด (Combined Convergence)")
        fig_combine, ax_combine = plt.subplots(figsize=(12, 5.5))
        ax_combine.axhline(y=res["ST Baseline"]['history'][0], color='r', linestyle='--', label=f'ST Baseline ({res["ST Baseline"]["history"][0]:.2f})')
        ax_combine.plot(res["Standard GA"]['history'], label='Standard GA', color='#1E40AF')
        ax_combine.plot(res["Hybrid GA-Q"]['history'], label='Hybrid GA-Q', color='#10B981')
        
        ax_combine.set_xlabel("Generation"); ax_combine.set_ylabel("Penalty Score")
        ax_combine.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.12),
            ncol=3,
            frameon=False,
            fontsize=11
        )
        ax_combine.grid(True, alpha=0.3)
        st.pyplot(fig_combine)

        st.markdown("---")

        # --- ตารางเวลา ---
        st.markdown("รายละเอียดตารางเวลา (Schedules)")
        tabs = st.tabs(["ST Baseline", "Standard GA", "Hybrid GA-Q"])
        
        for idx, algo in enumerate(["ST Baseline", "Standard GA", "Hybrid GA-Q"]):
            with tabs[idx]:
                sched_data = res[algo]['sched']
                for day in sorted(sched_data.keys()):
                    with st.expander(f"วันที่ {day + 1}", expanded=(day==0)):
                        for or_id in sorted(sched_data[day].keys(), key=lambda x: str(x)):
                            st.markdown(f"**ห้องผ่าตัด: {or_id}**")
                            
                            # คำนวณ Utilization ย่อยของแต่ละห้องในวันนั้น (ฟีเจอร์ใหม่เพื่อให้เห็นภาพชัดเจน)
                            total_booked_today = sum(c['booked_time'] for c in sched_data[day][or_id])
                            util_percent = (total_booked_today / total_avail_minutes_per_day) * 100
                            
                            st.markdown(f"**ห้องผ่าตัด (OR): {or_id}** &nbsp;&nbsp;|&nbsp;&nbsp; ใช้งาน: **{total_booked_today}/{total_avail_minutes_per_day}** นาที (**{util_percent:.1f}%**)")
                            
                            case_list = []
                            for c in sched_data[day][or_id]:
                                row_data = {
                                    "รหัสเคส": str(c['Encounter ID']),
                                    "แผนก": c.get('Actual_Dept', 'N/A'),
                                    "เทคนิคที่ใช้": c['Service'],
                                    "เวลาที่ใช้": c['booked_time'],
                                    "เวลาเริ่ม": slot_to_time(c['start_slot'], operating_time, input_slot_duration),
                                    "เวลาสิ้นสุด": slot_to_time(c['end_slot'], operating_time, input_slot_duration),
                                    "น้ำหนักเคส": round(c['Weight'], 2)
                                }
                                # ซ่อนเทคนิคดมยาสำหรับ Exp 1 โดยเฉพาะ
                                if exp_mode == "Experiment 1 (Kaggle)":
                                    del row_data["เทคนิคที่ใช้"]
                                case_list.append(row_data)

                            st.dataframe(
                                pd.DataFrame(case_list),
                                hide_index=True,
                                use_container_width=True,
                                column_config={
                                    "รหัสเคส": st.column_config.TextColumn("รหัสเคส", width="small"),
                                    "เวลาที่ใช้": st.column_config.NumberColumn("เวลาที่ใช้", format="%d"),
                                    # "เวลาที่ใช้": st.column_config.ProgressColumn("เวลาที่ใช้ (นาที)", format="%d min", min_value=0, max_value=300),
                                    "น้ำหนักเคส": st.column_config.NumberColumn("น้ำหนักเคส", format="%.2f")
                                }
                            )
