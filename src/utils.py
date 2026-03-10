import numpy as np
import random
from datetime import datetime, timedelta
from collections import defaultdict
from config.ga_config import CLUSTER_TO_ORS

# --- TIME CONVERSION ---
def slot_to_time(slot_number, OPERATING_TIME, SLOT_DURATION_MIN):
    start_hour = OPERATING_TIME[0]
    minutes_passed = slot_number * SLOT_DURATION_MIN
    start_dt = datetime(2025, 1, 1, int(start_hour), int((start_hour % 1) * 60))
    final_dt = start_dt + timedelta(minutes=minutes_passed)
    return final_dt.strftime("%H:%M")

# --- DECODE FUNCTION ---
def decode_individual(individual, surgeries, ALL_OR_IDS, TOTAL_SLOTS_PER_DAY, BUFFER_SLOTS):
    # OR_schedules จะเก็บข้อมูลแยกเป็นรายวัน เพื่อให้คำนวณ Penalty รายวันได้
    # โครงสร้าง: { day_index: { or_id: [case1, case2, ...] } }
    OR_schedules = defaultdict(lambda: defaultdict(list))    
    room_status = {or_id: {'day': 0, 'clock': 0} for or_id in ALL_OR_IDS}
    
    for i, idx in enumerate(individual['order']):
        surgery = surgeries[idx]
        or_id = individual['assigned_or_list'][i]
        
        # ดึงสถานะปัจจุบันของห้องที่ถูกเลือก
        curr_day = room_status[or_id]['day']
        curr_clock = room_status[or_id]['clock']
        
        # คำนวณจุดเริ่มต้น (ต้องมี Buffer ถ้าไม่ใช่เคสแรกของวัน)
        start_slot = curr_clock
        if start_slot > 0: 
            start_slot += BUFFER_SLOTS
            
        # คำนวณจุดสิ้นสุดที่เป็นไปได้ ถ้าจัดลงใน วันนี้
        # ถ้าจัดลงไปแล้วเกินเวลาทำการปกติ (TOTAL_SLOTS_PER_DAY) ให้ไปเริ่มเช้าวันใหม่
        potential_end = start_slot + surgery['slots_needed']
        if potential_end > TOTAL_SLOTS_PER_DAY:
            curr_day += 1      # ขยับไปวันถัดไป
            start_slot = 0     # เริ่มที่ Slot 0 (7:00 น.)
            potential_end = surgery['slots_needed']
            
        OR_schedules[curr_day][or_id].append({
            'Encounter ID': surgery['Encounter ID'],
            'booked_time': surgery['booked_time'],
            'Weight': surgery['Weight'],
            'start_slot': start_slot,
            'end_slot': potential_end,
            'day': curr_day
        })
        
        room_status[or_id]['day'] = curr_day
        room_status[or_id]['clock'] = potential_end
    return OR_schedules, room_status

# --- FITNESS FUNCTION ---
def evaluate_fitness(OR_schedules, room_status, TOTAL_SLOTS, W_MAKESPAN, W_OVERTIME, W_IMBALANCE):
    weighted_overtime = 0
    total_finish_slots = [] 

    # คำนวณเวลาจบงานของแต่ละห้อง
    for or_id, status in room_status.items():
        # Global Slot = (day * total_slot) + slot ที่จบในวันนั้น
        finish_at = (status['day'] * TOTAL_SLOTS) + status['clock']
        total_finish_slots.append(finish_at)
    # Makespan เวลาที่เคสสุดท้ายของทั้งชุดข้อมูลทั้งหมดทำเสร็จ
    global_makespan = max(total_finish_slots) 
    
    # Imbalance ความต่างของเวลาจบงานของแต่ละห้อง (SD)
    global_imbalance = np.std(total_finish_slots)

    # คำนวณ Overtime จากทุกวัน
    for day in OR_schedules:
        for or_id, cases in OR_schedules[day].items():
            for case in cases:
                if case['end_slot'] > TOTAL_SLOTS:
                    overlap = case['end_slot'] - max(TOTAL_SLOTS, case['start_slot'])
                    weighted_overtime += overlap * case.get('Weight', 0.5)

    # Normalization (ปรับค่าให้อยู่ในสเกล 0-1 เพื่อคูณกับ Weight)
    norm_makespan = global_makespan / (60 * TOTAL_SLOTS) 
    norm_overtime = weighted_overtime / (len(room_status) * TOTAL_SLOTS)
    norm_imbalance = global_imbalance / TOTAL_SLOTS

    fitness_score = (norm_makespan * W_MAKESPAN) + (norm_overtime * W_OVERTIME) + (norm_imbalance * W_IMBALANCE)
    # คะแนนรวม (ยิ่งน้อยยิ่งดี)
    return fitness_score
