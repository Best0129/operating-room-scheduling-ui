import numpy as np
import random
import math
from datetime import datetime, timedelta
from collections import defaultdict
# from config.ga_config import CLUSTER_TO_ORS

# --- TIME CONVERSION ---
def slot_to_time(slot_number, OPERATING_TIME, SLOT_DURATION_MIN):
    # คำนวณนาทีเริ่มต้นจากชั่วโมงเปิดห้องผ่าตัด
    start_hour = OPERATING_TIME[0]
    start_total_mins = int(start_hour * 60)
    
    # บวกนาทีที่ผ่านไปตามจำนวน Slot
    current_total_mins = start_total_mins + (slot_number * SLOT_DURATION_MIN)
    
    # แปลงกลับเป็นชั่วโมงและนาที
    hours = current_total_mins // 60
    mins = current_total_mins % 60
    
    # รูปแบบ String HH:MM (เติมเลข 0 ข้างหน้าถ้าหลักเดียว)
    return f"{hours:02d}:{mins:02d}"


# --- DECODE FUNCTION ---
def decode_individual(individual, surgeries, ALL_OR_IDS, TOTAL_SLOTS_PER_DAY, BUFFER_SLOTS):
    """
    ถอดรหัสโครโมโซมเป็นตารางเวลา: อิงตามตรรกะ Jupyter Notebook 100%
    ตรรกะ: หากเคสผ่าตัดจะทำให้เลิกงานเกินเวลาปกติ (Overtime) ให้ยกเคสนั้นไปเริ่มเช้าวันใหม่ทันที
    """
    # โครงสร้าง: { day_index: { or_id: [case1, case2, ...] } }
    OR_schedules = defaultdict(lambda: defaultdict(list))    
    room_status = {or_id: {'day': 0, 'clock': 0} for or_id in ALL_OR_IDS}
    
    for i, idx in enumerate(individual['order']):
        surgery = surgeries[idx]
        or_id = individual['assigned_or_list'][i]
        
        # 1. ดึงสถานะปัจจุบันของห้องที่ถูกเลือก
        curr_day = room_status[or_id]['day']
        curr_clock = room_status[or_id]['clock']
        
        # 2. คำนวณจุดเริ่มต้น (ต้องมี Buffer ถ้าไม่ใช่เคสแรกของวัน)
        start_slot = curr_clock
        if start_slot > 0: 
            start_slot += BUFFER_SLOTS
            
        # 3. คำนวณจุดสิ้นสุดที่เป็นไปได้
        potential_end = start_slot + surgery['slots_needed']
        
        # 🌟 4. ตรรกะการขึ้นวันใหม่ (อิงตาม Notebook)
        # ถ้าจัดลงไปแล้วเกินเวลาทำการปกติ (TOTAL_SLOTS_PER_DAY) ให้ไปเริ่มเช้าวันใหม่
        if potential_end > TOTAL_SLOTS_PER_DAY:
            curr_day += 1      # ขยับไปวันถัดไป
            start_slot = 0     # เริ่มที่ Slot 0 (เวลาเริ่มงานตอนเช้า)
            potential_end = surgery['slots_needed']
            
        # 5. บันทึกลงตาราง (เพิ่มคีย์ 'Service' จากฝั่ง UI เพื่อการแสดงผล)
        OR_schedules[curr_day][or_id].append({
            'Encounter ID': surgery['Encounter ID'],
            'Service': surgery.get('Service', 'N/A'), # ดึงจาก surgery เพื่อโชว์ใน UI
            'booked_time': surgery['booked_time'],
            'Weight': surgery['Weight'],
            'start_slot': start_slot,
            'end_slot': potential_end,
            'day': curr_day
        })
        
        # 6. อัปเดตสถานะห้องเพื่อใช้คำนวณเคสถัดไป
        room_status[or_id]['day'] = curr_day
        room_status[or_id]['clock'] = potential_end

    return OR_schedules, room_status


# --- FITNESS FUNCTION ---
def evaluate_fitness(OR_schedules, room_status, TOTAL_SLOTS, W_MAKESPAN, W_OVERTIME, W_IMBALANCE):
    """
    Fitness Function: อ้างอิงตามหลักการใน Jupyter Notebook
    เน้นการลด Makespan, Overtime และความไม่สมดุลของงาน (Imbalance)
    """
    weighted_overtime = 0
    total_finish_slots = [] 

    # 1. คำนวณเวลาจบงานของแต่ละห้อง (Global Finish Slot)
    for or_id, status in room_status.items():
        # Global Slot = (จำนวนวัน * total_slot) + slot ที่จบในวันนั้น
        # ช่วยให้เห็นภาพรวมเวลาจบงานตั้งแต่วันแรกจนถึงวันสุดท้าย
        finish_at = (status['day'] * TOTAL_SLOTS) + status['clock']
        total_finish_slots.append(finish_at)
    
    # Makespan: เวลาที่เคสสุดท้ายของทั้งชุดข้อมูลทำเสร็จ
    global_makespan = max(total_finish_slots) if total_finish_slots else 0
    
    # Imbalance: ความต่างของเวลาจบงานของแต่ละห้อง (ใช้ค่าเบี่ยงเบนมาตรฐาน SD)
    global_imbalance = np.std(total_finish_slots) if total_finish_slots else 0

    # 2. คำนวณ Overtime รวม (คิดน้ำหนักตามความสำคัญของเคส)
    for day in OR_schedules:
        for or_id in OR_schedules[day]:
            for case in OR_schedules[day][or_id]:
                if case['end_slot'] > TOTAL_SLOTS:
                    # คำนวณส่วนที่เกินจากเวลาปกติ
                    overlap = case['end_slot'] - max(TOTAL_SLOTS, case['start_slot'])
                    # ใช้ค่า Weight 0.5 เป็นค่าเริ่มต้นตาม Notebook
                    weighted_overtime += overlap * case.get('Weight', 0.5)

    # 3. Normalization (ปรับค่าให้อยู่ในสเกล 0-1 ตามหลักการใน Notebook)
    # ใช้ค่าคงที่ 60 (จำนวนวันโดยประมาณ) เป็นตัวหารสำหรับ Makespan
    norm_makespan = global_makespan / (60 * TOTAL_SLOTS) 
    # หารด้วยความจุรวมของทุกห้องผ่าตัดสำหรับ Overtime
    norm_overtime = weighted_overtime / (len(room_status) * TOTAL_SLOTS)
    # หารด้วยจำนวน Slot ต่อวันสำหรับ Imbalance
    norm_imbalance = global_imbalance / TOTAL_SLOTS

    # 4. คำนวณคะแนน Fitness (ยิ่งน้อยยิ่งดี)
    # ตัด W_PRIORITY ออกเพื่อให้เหมือนกับใน Notebook เป๊ะๆ
    fitness_score = (norm_makespan * W_MAKESPAN) + \
                    (norm_overtime * W_OVERTIME) + \
                    (norm_imbalance * W_IMBALANCE)
                    
    return fitness_score


def calculate_metrics(OR_schedules, room_status, TOTAL_SLOTS, SLOT_DURATION_MIN, ALL_OR_IDS, surgeries, mode):
    """
    คำนวณตัวชี้วัดประสิทธิภาพ (KPIs): รวมตรรกะจาก Notebook และระบบวัดผล Optimality
    [cite: 2026-03-12]
    """
    if not room_status:
        return {}

    total_booked_min = sum(s['booked_time'] for s in surgeries)
    num_rooms = len(ALL_OR_IDS)
    
    # 1. คำนวณ Makespan (จำนวนวันที่ใช้จริง) - อิงตาม Notebook
    # หาค่า day ที่สูงที่สุดจาก room_status และ +1 เพราะ index เริ่มที่ 0
    max_day_idx = max(status['day'] for status in room_status.values())
    total_days_used = max_day_idx + 1 
    
    # 2. คำนวณ Theoretical Lower Bound (LB)
    # สูตร: ผลรวมเวลาผ่าตัดทั้งหมด / ความจุรวมของห้องผ่าตัดทุกห้องใน 1 วัน
    daily_capacity_per_room = TOTAL_SLOTS * SLOT_DURATION_MIN
    total_daily_capacity = num_rooms * daily_capacity_per_room
    lb_days = math.ceil(total_booked_min / total_daily_capacity)
    
    # 3. คำนวณ Optimality Gap (%) 
    # วัดว่าจำนวนวันที่ใช้จริง ห่างจากค่าทางทฤษฎีที่น้อยที่สุดกี่เปอร์เซ็นต์
    gap_percent = ((total_days_used - lb_days) / lb_days) * 100 if lb_days > 0 else 0
    
    # 4. คำนวณ Total Overtime (สะสมจากทุกวันทุกห้อง) - อิงตาม Notebook
    total_ot_min = 0
    for day in OR_schedules:
        for or_id in OR_schedules[day]:
            for case in OR_schedules[day][or_id]:
                # ตรวจสอบว่าจบเกินเวลาทำการปกติหรือไม่
                if case['end_slot'] > TOTAL_SLOTS:
                    # คำนวณเฉพาะส่วนที่ล้นออกมาจากขอบเขต TOTAL_SLOTS
                    ot_slots = case['end_slot'] - max(TOTAL_SLOTS, case['start_slot'])
                    total_ot_min += ot_slots * SLOT_DURATION_MIN
                    
    # 5. คำนวณ Utilization (%) - อิงตาม Notebook
    # สูตร: (เวลาผ่าตัดรวม) / (จำนวนห้อง * จำนวนวันที่เปิดใช้จริง * เวลาทำการต่อวัน)
    total_capacity_available = num_rooms * total_days_used * daily_capacity_per_room
    global_util = (total_booked_min / total_capacity_available) * 100 if total_capacity_available > 0 else 0
    
    # 6. ดึง Penalty Score จาก Fitness Function ที่เรา Sync กับ Notebook แล้ว
    from config.ga_config import W_MAKESPAN, W_OVERTIME, W_IMBALANCE
    penalty = evaluate_fitness(OR_schedules, room_status, TOTAL_SLOTS, W_MAKESPAN, W_OVERTIME, W_IMBALANCE)

    return {
        'Total_Days': int(total_days_used),
        'Lower_Bound_Days': int(lb_days),
        'Optimality_Gap (%)': round(gap_percent, 2),
        'Global_Util (%)': round(global_util, 2),
        'Total_OT_Min': int(total_ot_min),
        'Penalty_Score': round(penalty, 4),
        'Total_Booked_Min': total_booked_min
    }
