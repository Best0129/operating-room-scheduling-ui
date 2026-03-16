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
            font-size: 40px; 
            font-weight: bold; 
            color: #1E40AF; 
            margin-bottom: 20px; 
            text-align: center;
        }

        div.stButton > button[kind="primary"] {
            font-weight: 600;
            transition: all 0.2s ease;
        }

        /* ส่วนที่ทำให้ st.metric มีกรอบสวยๆ เหมือนแบบที่ 2 */
        [data-testid="stMetric"] {
            background-color: #F8FAFC; /* สีพื้นหลังเทาอ่อน */
            padding: 15px;             /* ระยะห่างจากขอบด้านใน */
            border-radius: 10px;       /* ความมนของมุม */
            border: 1px solid #E2E8F0; /* เส้นขอบบางๆ */
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); /* เงาบางๆ เพิ่มมิติ */
        }
        
        /* แก้ไขให้ตัวเลขและหัวข้อจัดวางสวยงามในกรอบ */
        [data-testid="stMetric"] > div {
            width: fit-content;
            margin: auto;
        }

        /* ตกแต่งสีของตัวหนังสือ Lower Bound (Delta) ให้ดูสะอาดตา */
        [data-testid="stMetricDelta"] > div {
            font-weight: 500;
        }

        /* สไตล์สำหรับข้อความที่อยู่นอกกล่อง (Lower Bound) */
        .lower-bound-text {
            font-size: 14px;
            color: #FF0000;
            text-align: center;
            margin-top: -15px; /* ดันขึ้นไปให้ใกล้กล่องมากขึ้น */
            margin-bottom: 15px;
        }

       /* --- เพิ่มเติมส่วน Download CSV --- */
        .download-center-container {
            # border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 24px;
            margin-top: 10px;
        }

        .download-card {
            # padding: 16px;
            text-align: center;
            margin-bottom: 10px;
        }

        .download-label {
            font-size: 16px;
            font-weight: 700;
            display: block;
            margin-bottom: 5px;
        }

        /* ปรับแต่งปุ่ม Download ของ Streamlit ให้เป็น Modern Outline */
        div.stDownloadButton > button {
            width: 100% !important;
            background-color: white !important;
            color: #1E40AF !important;
            border: 1.5px solid #1E40AF !important;
            border-radius: 8px !important;
            padding: 0.5rem 1rem !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }

        div.stDownloadButton > button:hover {
            background-color: #F8FAFC !important;
            border-color: #1E40AF !important;
            color: #1E40AF !important;
            box-shadow: 0 4px 6px -1px rgba(30, 64, 175, 0.1) !important;
        }

        /* เส้นแบ่งส่วนในหน้าดาวน์โหลด */
        .download-divider {
            border-top: 1px solid #E2E8F0;
            margin: 20px 0;
        }

          .overview-header {
              font-size: 18px;
              font-weight: 600;
              margin-bottom: 15px;
              display: flex;
              align-items: center;
              gap: 10px;
          }

        .overview-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
        }

        .overview-item {
            background-color: #F8FAFC;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }

        .ov-label { 
            font-size: 16px; 
            font-weight: 600; 
            display: block; 
            margin-bottom: 4px; 
        }

        .ov-value { 
            font-size: 16px; 
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
        st.session_state.results = {}      # ล้างผลลัพธ์เก่า
        st.session_state.last_exp = exp_mode # อัปเดตโหมดปัจจุบัน
        st.rerun() # บังคับให้แอปเริ่มรันใหม่ทันที เพื่อให้ส่วนแสดงผลหายไป

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

if not surgeries_list:
    st.warning("ไม่พบข้อมูลสำหรับชุดการทดลองนี้")
    st.stop()

# ดึงข้อมูลห้องผ่าตัด
all_or_ids = list(set(str(or_id).strip() for ors in cfg.get("CLUSTER_TO_ORS", {}).values() for or_id in ors))

# st.header(f"ข้อมูลของชุดการทดลอง: {exp_mode}")
st.markdown(
    f"<h3>ข้อมูลของชุดการทดลอง: {exp_mode}</h3>",
    unsafe_allow_html=True
)

# --- 1. คำนวณค่าพื้นฐาน ---
total_cases = len(surgeries_list)
total_ors = len(all_or_ids)
total_mins = sum(s['booked_time'] for s in surgeries_list)
unique_depts = len(set(s.get('Actual_Dept', s.get('Service', '')) for s in surgeries_list))

# --- 2. ตรรกะคำนวณช่วงวันที่แบบใหม่ (รองรับทุกโหมด) ---
all_dates = [s.get('Original_Date', s.get('Date')) for s in surgeries_list]
# กรองค่าว่างและ Unknown ออก
clean_dates = [d for d in all_dates if d and str(d).lower() != 'unknown']

if clean_dates:
    try:
        # ใช้ pd.to_datetime เพื่อความยืดหยุ่นในการอ่าน format วันที่
        date_series = pd.to_datetime(clean_dates)
        start_date = date_series.min()
        end_date = date_series.max()
        num_days = len(date_series.unique())
        
        # แปลงเป็นปี พ.ศ. (+543)
        start_str = f"{start_date.day}/{start_date.month}/{start_date.year + 543}"
        end_str = f"{end_date.day}/{end_date.month}/{end_date.year + 543}"
        dataset_days_display = f"{start_str} - {end_str} ({num_days} วัน)"
    except:
        # ถ้าแปลงวันที่ไม่ได้จริงๆ ให้แสดงแค่จำนวนวันที่นับได้
        num_days = len(set(clean_dates))
        dataset_days_display = f"{num_days} วัน"
else:
    dataset_days_display = "ไม่ระบุ"

# --- 3. แสดงผลหน้าจอ (HTML & CSS) ---
st.markdown(f"""
    <div class="overview-container">
        <div class="overview-header"><h4>ภาพรวมชุดข้อมูล (Dataset Overview)</h4></div>
        <div class="overview-grid">
            <div class="overview-item">
                <span class="ov-label">จำนวนเคสทั้งหมด</span>
                <span class="ov-value">{total_cases:,} เคส</span>
            </div>
            <div class="overview-item">
                <span class="ov-label">ระยะเวลาข้อมูล</span>
                <span class="ov-value">{dataset_days_display}</span> 
            </div>
            <div class="overview-item">
                <span class="ov-label">จำนวนแผนก</span>
                <span class="ov-value">{unique_depts} แผนก</span>
            </div>
            <div class="overview-item">
                <span class="ov-label">จำนวนห้องผ่าตัด</span>
                <span class="ov-value">{total_ors} ห้อง</span>
            </div>
            <div class="overview-item">
                <span class="ov-label">ปริมาณงานรวม</span>
                <span class="ov-value">{total_mins / 60:,.1f} ชั่วโมง</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

if run_button:
    if not surgeries_list:
        st.error("ไม่พบข้อมูล!")
        st.stop()
    else:
        # เคลียร์ผลลัพธ์เก่าทิ้งเมื่อกดรันใหม่
        st.session_state.results = {}

# หากมีการกดรัน หรือมีผลลัพธ์เก่าค้างอยู่ ให้แสดงหน้าต่างแสดงผล
if run_button or st.session_state.results:
    # st.header(f"ผลการทดลอง: {exp_mode}")
    st.markdown(
        f"<h3>ผลการทดลอง: {exp_mode}</h3>",
        unsafe_allow_html=True
    )
    # เตรียมกล่อง Container 3 กล่องสำหรับ 3 อัลกอริทึมรอไว้แต่แรก
    container_st = st.container()
    container_ga = st.container()
    container_q = st.container()

    def style_plot(ax, title, xlabel, ylabel):
        ax.set_title(title, pad=15, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    # ---------------------------------------------------------
    # 1. ST Baseline (Heuristic)
    # ---------------------------------------------------------
    with container_st:
        # st.subheader("1. ST Baseline (Heuristic)")

        st.markdown(
            f"<h4>1. ST Baseline (Heuristic)</h4>",
            unsafe_allow_html=True
        ) 
        
        
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
            # คอลัมน์ที่ 1 (ตัวอย่างการปรับ)
            c1.metric("🗓️ จำนวนวันที่ใช้", f"{m.get('Total_Days', 0)} วัน")
            # ใส่ข้อความไว้นอกกล่อง
            c1.markdown(f'<div class="lower-bound-text">Lower Bound: {m.get('Lower_Bound_Days', 0)} วัน</div>', unsafe_allow_html=True)

            # คอลัมน์อื่น ๆ ก็ทำเช่นเดียวกัน
            c2.metric("🎯 Optimality Gap", f"{m.get('Optimality_Gap (%)', 0)}%")
            c3.metric("📊 Utilization", f"{m.get('Global_Util (%)', 0)}%")
            c4.metric("⚡ Runtime", f"{m.get('Runtime_Sec', 0)} s")
            c5.metric("📉 Penalty Score", f"{m.get('Penalty_Score', 0)}")
            
            # st.info("ST Baseline คำนวณรอบเดียวไม่มีกราฟการเรียนรู้")
            st.caption("* ST Baseline (Heuristic) เป็นการคำนวณแบบรอบเดียว ไม่มีกราฟการเรียนรู้")
        st.markdown("---")

    # ---------------------------------------------------------
    # 2. Standard GA
    # ---------------------------------------------------------
    with container_ga:
        # st.subheader("2. Standard GA")

        st.markdown(
            f"<h4>2. Standard GA</h4>",
            unsafe_allow_html=True
        ) 
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
                # คอลัมน์ที่ 1 (ตัวอย่างการปรับ)
                c1.metric("🗓️ จำนวนวันที่ใช้", f"{m.get('Total_Days', 0)} วัน")
                # ใส่ข้อความไว้นอกกล่อง
                c1.markdown(f'<div class="lower-bound-text">Lower Bound: {m.get('Lower_Bound_Days', 0)} วัน</div>', unsafe_allow_html=True)

                # คอลัมน์อื่น ๆ ก็ทำเช่นเดียวกัน
                c2.metric("🎯 Optimality Gap", f"{m.get('Optimality_Gap (%)', 0)}%")
                c3.metric("📊 Utilization", f"{m.get('Global_Util (%)', 0)}%")
                c4.metric("⚡ Runtime", f"{m.get('Runtime_Sec', 0)} s")
                c5.metric("📉 Penalty Score", f"{m.get('Penalty_Score', 0)}")

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
        # st.subheader("3. Hybrid GA-Q")
        st.markdown(
            f"<h4>3. Hybrid GA-Q</h4>",
            unsafe_allow_html=True
        ) 
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
                # คอลัมน์ที่ 1 (ตัวอย่างการปรับ)
                c1.metric("🗓️ จำนวนวันที่ใช้", f"{m.get('Total_Days', 0)} วัน")
                # ใส่ข้อความไว้นอกกล่อง
                c1.markdown(f'<div class="lower-bound-text">Lower Bound: {m.get('Lower_Bound_Days', 0)} วัน</div>', unsafe_allow_html=True)

                # คอลัมน์อื่น ๆ ก็ทำเช่นเดียวกัน
                c2.metric("🎯 Optimality Gap", f"{m.get('Optimality_Gap (%)', 0)}%")
                c3.metric("📊 Utilization", f"{m.get('Global_Util (%)', 0)}%")
                c4.metric("⚡ Runtime", f"{m.get('Runtime_Sec', 0)} s")
                c5.metric("📉 Penalty Score", f"{m.get('Penalty_Score', 0)}")

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
        
        # st.subheader("สรุปผลการเปรียบเทียบรวมทั้ง 3 อัลกอริทึม")
        st.markdown(
            f"<h4>สรุปผลการเปรียบเทียบรวมทั้ง 3 อัลกอริทึม</h4>",
            unsafe_allow_html=True
        ) 
        
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
            df_compare.style.highlight_min(subset=["วันที่ใช้ (Days)", "Optimality Gap (%)", "Runtime (s)", "Penalty Score"], color='#DBEAFE')
                            .highlight_max(subset=["Utilization (%)"], color='#DBEAFE'),
            use_container_width=True, hide_index=True
        )

        # --- กราฟเปรียบเทียบรวม ---
        # st.markdown("<br><b>กราฟเปรียบเทียบการลู่เข้าหาคำตอบที่ดีที่สุด (Combined Convergence)</b>", unsafe_allow_html=True)
        # st.subheader("กราฟเปรียบเทียบการลู่เข้าหาคำตอบที่ดีที่สุด (Convergence Comparison of Algorithms)")
        st.markdown(
            f"<h4>กราฟเปรียบเทียบการลู่เข้าหาคำตอบที่ดีที่สุด (Convergence Comparison of Algorithms)</h4>",
            unsafe_allow_html=True
        ) 
        fig_combine, ax_combine = plt.subplots(figsize=(12, 5.5))
        ax_combine.axhline(y=res["ST Baseline"]['history'][0], color='r', linestyle='--', label=f'ST Baseline ({res["ST Baseline"]["history"][0]:.2f})')
        ax_combine.plot(res["Standard GA"]['history'], label='Standard GA', color='#1E40AF')
        ax_combine.plot(res["Hybrid GA-Q"]['history'], label='Hybrid GA-Q', color='#10B981')
        
        style_plot(ax_combine, "", "Generation", "Penalty Score")
        # ax_combine.set_xlabel("Generation"); ax_combine.set_ylabel("Penalty Score")
        ax_combine.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 1.15),
            ncol=3,
            frameon=False,
            fontsize=11
        )
        # ax_combine.grid(True, alpha=0.3)
        st.pyplot(fig_combine)

        st.markdown("---")

        # --- ตารางเวลา ---
        # st.subheader("การจัดตารางเวลาห้องผ่าตัด (Schedules)")

        st.markdown(
            f"<h4>การจัดตารางเวลาห้องผ่าตัด (Schedules)</h4>",
            unsafe_allow_html=True
        ) 
        
        tabs = st.tabs(["ST Baseline", "Standard GA", "Hybrid GA-Q", "ดาวน์โหลด CSV"])
        
        # ฟังก์ชันช่วยเตรียมข้อมูลสำหรับ Export เป็น DataFrame
        def prepare_export_data(algo_name, sched_data):
            export_list = []
            for day in sorted(sched_data.keys()):
                for or_id in sorted(sched_data[day].keys(), key=lambda x: str(x)):
                    for c in sched_data[day][or_id]:
                        row_data = {
                            "อัลกอริทึม (Algorithm)": algo_name,
                            "วันที่ (Day)": day + 1,
                            "ห้องผ่าตัด (OR)": or_id,
                            "รหัสเคส (ID)": str(c['Encounter ID']),
                            "แผนก (Service)": c.get('Actual_Dept', 'N/A'),
                            "เทคนิค (Technique)": c['Service'],
                            "เวลาเริ่ม (Start)": slot_to_time(c['start_slot'], operating_time, input_slot_duration),
                            "เวลาสิ้นสุด (End)": slot_to_time(c['end_slot'], operating_time, input_slot_duration),
                            "เวลาที่ใช้ (Mins)": c['booked_time'],
                            "น้ำหนัก (Weight)": round(c['Weight'], 2)
                        }
                        # ซ่อนเทคนิคดมยาสำหรับ Exp 1
                        if exp_mode == "Experiment 1 (Kaggle)":
                            del row_data["เทคนิค (Technique)"]
                        export_list.append(row_data)
            return pd.DataFrame(export_list)

        # -------------------------------------------------------------
        # แท็บที่ 1-3: แสดงผลตารางเวลาบน UI
        # -------------------------------------------------------------
        for idx, algo in enumerate(["ST Baseline", "Standard GA", "Hybrid GA-Q"]):
            with tabs[idx]:
                if algo in res:
                    sched_data = res[algo]['sched']
                    for day in sorted(sched_data.keys()):
                        
                        with st.expander(f"วันที่ {day + 1}", expanded=(day==0)):
                            for or_id in sorted(sched_data[day].keys(), key=lambda x: str(x)):
                                
                                # คำนวณ Utilization ย่อย
                                total_booked_today = sum(c['booked_time'] for c in sched_data[day][or_id])
                                util_percent = (total_booked_today / total_avail_minutes_per_day) * 100
                                
                                st.markdown(f"**ห้องผ่าตัด (OR): {or_id}** &nbsp;&nbsp;|&nbsp;&nbsp; ใช้งาน: **{total_booked_today}/{total_avail_minutes_per_day}** นาที (**{util_percent:.1f}%**)")
                                
                                df_display = prepare_export_data(algo, {day: {or_id: sched_data[day][or_id]}})
                                # ลบคอลัมน์ที่ไม่จำเป็นต้องแสดงใน UI ออก (เพราะมีหัวข้ออยู่แล้ว)
                                df_display = df_display.drop(columns=["อัลกอริทึม (Algorithm)", "วันที่ (Day)", "ห้องผ่าตัด (OR)"])

                                st.dataframe(
                                    df_display,
                                    hide_index=True,
                                    use_container_width=True,
                                    column_config={
                                        "รหัสเคส (ID)": st.column_config.TextColumn("รหัสเคส (ID)", width="small"),
                                        "เวลาที่ใช้ (Mins)": st.column_config.NumberColumn("เวลาที่ใช้ (Mins)", format="%d"),
                                        "น้ำหนัก (Weight)": st.column_config.NumberColumn("น้ำหนัก (Weight)", format="%.2f")
                                    }
                                )
                            # st.markdown("<br>", unsafe_allow_html=True)

        # -------------------------------------------------------------
        # แท็บที่ 4: ดาวน์โหลดข้อมูล (Download CSV)
        # -------------------------------------------------------------
        with tabs[3]:
            st.markdown(
                f"""
                <div class="download-center-container">
                    <h3 style='margin-top:0; color:#1E40AF; font-size:20px;'>
                        ดาวน์โหลดผลลัพธ์การจัดตารางเวลาห้องผ่าตัด: {exp_mode}
                    </h3>
                    <p style='color:#64748B; font-size:16px; margin-bottom:20px;'>
                        สามารถเลือกดาวน์โหลดข้อมูลตารางเวลาแยกตามอัลกอริทึม
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            all_dfs = []
            
            # ส่วนดาวน์โหลดแยกตาม Algorithm ในรูปแบบ Card Grid
            col1, col2, col3 = st.columns(3)
            algo_cols = [col1, col2, col3]
            algos = ["ST Baseline", "Standard GA", "Hybrid GA-Q"]
            
            for i, algo in enumerate(algos):
                if algo in res:
                    df_algo = prepare_export_data(algo, res[algo]['sched'])
                    all_dfs.append(df_algo)
                    csv = df_algo.to_csv(index=False).encode('utf-8-sig')
                    
                    with algo_cols[i]:

                      st.markdown(f"""
                          <div class="download-card">
                              <span class="download-label">{algo}</span>
                      """, unsafe_allow_html=True)

                      st.download_button(
                          label="ดาวน์โหลดไฟล์ CSV",
                          data=csv,
                          file_name=f"Schedule_{algo.replace(' ', '_')}_{exp_mode[:5]}.csv",
                          mime="text/csv",
                          key=f"dl_{algo}",
                          use_container_width=True
                      )

                      st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            # ส่วนดาวน์โหลดรวม (Combined)
            if all_dfs:
                st.markdown("""<div style='border-top: 1px solid #E2E8F0; margin: 10px 0 25px 0;'></div>""", unsafe_allow_html=True)
                
                # จัดกลุ่มปุ่มดาวน์โหลดรวมให้อยู่ในพื้นที่ที่เด่นขึ้นเล็กน้อย
                c_left, c_mid, c_right = st.columns([1, 2, 1])
                with c_mid:
                    st.markdown("<span style='display:block; text-align:center; font-size:16px; font-weight:600;'>สรุปการจัดตารางเวลาห้องผ่าตัดรวม</span>", unsafe_allow_html=True)
                    df_combined = pd.concat(all_dfs, ignore_index=True)
                    csv_combined = df_combined.to_csv(index=False).encode('utf-8-sig')
                    
                    st.download_button(
                        label="ดาวน์โหลดตารางเวลารวมทั้ง 3 อัลกอริทึม",
                        data=csv_combined,
                        file_name=f"Combined_Schedules_{exp_mode[:5]}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="dl_combined"
                    )
