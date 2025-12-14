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
def decode_individual(individual, surgeries):
    OR_schedules = defaultdict(list)
    # current_or_slots = defaultdict(lambda: total_slots) 

    order = individual.get("order", [])
    assigned_or_list = individual.get("assigned_or_list", [])

    # วนลูปตามลำดับของเคสใน 'order' Array
    for i, idx in enumerate(order):
        # i คือตำแหน่งใน Array ซึ่งใช้ในการดึง OR Suite ID จาก assigned_or_list
        # idx คือ Case Index (0 ถึง N-1)
        
        surgery = surgeries[idx]
        cluster = surgery.get('cluster', None)

        # ใช้ OR ที่ assigned ไว้แล้ว (ดึงจาก assigned_or_list โดยใช้ตำแหน่ง i)
        or_id = assigned_or_list[i] if i < len(assigned_or_list) else None

        if or_id is None:
            # Fallback (ควรเกิดขึ้นน้อยมากถ้า generate_initial_population ทำงานถูก)
            if cluster in CLUSTER_TO_ORS:
                or_id = random.choice(CLUSTER_TO_ORS[cluster])
            else:
                # ใช้ OR Suite ที่กำหนดไว้ในข้อมูลตั้งต้น
                or_id = surgery.get('or_suite') 
        
        timeline = OR_schedules[or_id]
        if not timeline:
            start_slot = 0
        else:
            # บวก buffer slot จากเคสก่อนหน้า
            start_slot = timeline[-1]['end_slot'] + surgery['buffer_slots'] 

        end_slot = start_slot + surgery['slots_needed']

        # if end_slot > current_or_slots[or_id]: # ถ้าเกิน slot ที่กำหนดไว้
        #     current_or_slots[or_id] = end_slot # ขยาย slot ที่ใช้ได้ใน OR นั้น

        OR_schedules[or_id].append({
            'Encounter ID': surgery['Encounter ID'],
            'Service': surgery['service'],
            'Cluster': cluster,
            'OR Suite': surgery['or_suite'],
            'Assigned OR': or_id, # เพิ่ม Assigned OR
            'start_slot': start_slot,
            'end_slot': end_slot,
            'slots_used': surgery['slots_needed'],
            'booked_time': surgery['booked_time'],
            'buffer_slot': surgery['buffer_slots']
        })

    # คำนวณ slot สุดท้ายที่ใช้ในแต่ละ OR
    total_used_slots = {
        or_id: (schedule[-1]['end_slot'] if schedule else 0)
        for or_id, schedule in OR_schedules.items()
    }

    return OR_schedules, total_used_slots

# --- FITNESS FUNCTION ---
def evaluate_fitness(OR_schedules, total_used_slots, total_slots, W_OVERTIME, W_IMBALANCE):
    
    total_overtime = 0
    total_imbalance = 0
    
    total_overtime = 0  
    total_imbalance = 0
    
    # 1. คำนวณ Overtime
    for or_id, final_slot_used in total_used_slots.items():
        overtime = max(0, final_slot_used - total_slots)
        total_overtime += overtime

    # 2. คำนวณ Imbalance
    for cluster_id, ors in CLUSTER_TO_ORS.items():
        if len(ors) > 1: # มีมากกว่า 1 OR ใน cluster นี้
            makespan_in_cluster = [total_used_slots.get(or_id, 0) for or_id in ors]
            
            if len(makespan_in_cluster) > 1 and np.sum(makespan_in_cluster) > 0:
                std_dev = np.std(makespan_in_cluster)
                total_imbalance += std_dev
    
    # 3. คำนวณ Fitness Score
    overtime_penalty = total_overtime * W_OVERTIME
    imbalance_penalty = total_imbalance * W_IMBALANCE
    
    fitness_score = overtime_penalty + imbalance_penalty
    
    return fitness_score
