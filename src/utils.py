import numpy as np
import random
from datetime import datetime, timedelta
from collections import defaultdict
from config.ga_config import CLUSTER_TO_ORS

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
    # โครงสร้าง: { day_index: { or_id: [case1, case2, ...] } }
    OR_schedules = defaultdict(lambda: defaultdict(list))    
    room_status = {or_id: {'day': 0, 'clock': 0} for or_id in ALL_OR_IDS}
    
    for i, idx in enumerate(individual['order']):
        surgery = surgeries[idx]
        or_id = individual['assigned_or_list'][i]
        
        # ดึงสถานะปัจจุบัน
        curr_day = room_status[or_id]['day']
        curr_clock = room_status[or_id]['clock']
        
        # คำนวณจุดเริ่มต้น (บวก Buffer ถ้าไม่ใช่เคสแรกของวัน)
        start_slot = curr_clock
        if start_slot > 0: 
            start_slot += BUFFER_SLOTS
            
        # ตรรกะการขยับวันใหม่
        # ถ้าเริ่มงานหลังจากหมดเวลาทำการปกติแล้ว ให้ไปเริ่มเช้าวันถัดไป
        if start_slot >= TOTAL_SLOTS_PER_DAY:
            curr_day += 1
            start_slot = 0
            
        potential_end = start_slot + surgery['slots_needed']
        
        # 4. บันทึกลงตาราง
        OR_schedules[curr_day][or_id].append({
            'Encounter ID': surgery['Encounter ID'],
            'Service': surgery['Service'],
            'booked_time': surgery['booked_time'],
            'Weight': surgery['Weight'],
            'start_slot': start_slot,
            'end_slot': potential_end,
            'day': curr_day
        })
        
        # อัปเดตสถานะห้อง
        room_status[or_id]['day'] = curr_day
        room_status[or_id]['clock'] = potential_end

    return OR_schedules, room_status


# --- FITNESS FUNCTION ---
def evaluate_fitness(OR_schedules, room_status, TOTAL_SLOTS, W_MAKESPAN, W_OVERTIME, W_IMBALANCE, W_PRIORITY=0.3):
    weighted_overtime = 0
    priority_delay_penalty = 0
    total_finish_slots = [] 

    # คำนวณเวลาจบงานของแต่ละห้อง
    for or_id, status in room_status.items():
        finish_at = (status['day'] * TOTAL_SLOTS) + status['clock']
        total_finish_slots.append(finish_at)
    
    global_makespan = max(total_finish_slots) 
    global_imbalance = np.std(total_finish_slots)

    # คำนวณ Overtime และ Priority Delay (ความล่าช้าของเคสสำคัญ)
    total_cases = 0
    for day in OR_schedules:
        for or_id, cases in OR_schedules[day].items():
            for case in cases:
                total_cases += 1
                # คิด Overtime Weight (เน้นไม่ให้เคสเสี่ยงสูงต้องรอผ่าตัดดึกๆ) 
                if case['end_slot'] > TOTAL_SLOTS:
                    overlap = case['end_slot'] - max(TOTAL_SLOTS, case['start_slot'])
                    weighted_overtime += overlap * case.get('Weight', 0.4)
                
                # เพิ่ม Priority Delay: เคส Weight สูง ยิ่งจัดวันช้ายิ่งโทษหนัก
                # ช่วยให้เคส Emergency ถูกดึงมาผ่าตัดวันแรกๆ เสมอ 
                priority_delay_penalty += (case.get('Weight', 0.4) * case['day'])

    # Normalization (ปรับเป็นสเกล 0-1)
    # ใช้จำนวนเคสหารจำนวนห้องเป็นตัวกะระยะเวลาคร่าวๆ
    estimated_days = max(1, total_cases / len(room_status))
    norm_makespan = global_makespan / (estimated_days * 3 * TOTAL_SLOTS) 
    
    norm_overtime = weighted_overtime / (len(room_status) * TOTAL_SLOTS)
    norm_imbalance = global_imbalance / (estimated_days * TOTAL_SLOTS)
    
    # Normalization สำหรับ Priority (หารด้วยจำนวนเคส)
    norm_priority = priority_delay_penalty / max(1, total_cases)

    # คำนวณ Fitness Score (ยิ่งน้อยยิ่งดี)
    fitness_score = (norm_makespan * W_MAKESPAN) + \
                    (norm_overtime * W_OVERTIME) + \
                    (norm_imbalance * W_IMBALANCE) + \
                    (norm_priority * W_PRIORITY)
                    
    return fitness_score
