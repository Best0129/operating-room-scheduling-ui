import pandas as pd
from collections import defaultdict
from config.ga_config import CONFIGS

def run_ST(surgeries, TOTAL_SLOTS, BUFFER_SLOTS, mode):
    """
    ST Baseline: อิงตามหลักการใน Jupyter Notebook 100%
    ใช้ตรรกะการขึ้นวันใหม่เมื่อพยากรณ์ว่างานจะจบเกินเวลาทำการ (Hard Constraint)
    """
    # 1. การดึง Config ตามโหมดการทดลอง (ส่วนเสริมสำหรับ UI)
    current_cfg = CONFIGS[mode]
    cluster_to_ors = current_cfg["CLUSTER_TO_ORS"]
    all_or_ids = [or_id for ors in cluster_to_ors.values() for or_id in ors]

    # 2. เรียงลำดับเคสตาม Weight จากมากไปน้อย (เหมือน Notebook)
    sorted_indices = sorted(range(len(surgeries)), key=lambda i: surgeries[i]['Weight'], reverse=True)
    
    # 3. เริ่มทุกห้องที่ วันที่ 0, เวลา 0 (เหมือน Notebook)
    room_status = {or_id: {'day': 0, 'clock': 0} for or_id in all_or_ids}
    OR_schedules = defaultdict(lambda: defaultdict(list))

    # 4. วนลูปตามลำดับเคสที่จัดเรียงแล้ว
    for idx in sorted_indices:
        surgery = surgeries[idx]
        allowed_ors = cluster_to_ors.get(surgery['cluster'], [])
        
        if not allowed_ors:
            continue 

        # 5. เลือกห้องที่ Global Time ต่ำที่สุด (เหมือน Notebook)
        # Global Time = (day * TOTAL_SLOTS) + clock
        assigned_or = min(allowed_ors, key=lambda r: (room_status[r]['day'] * TOTAL_SLOTS) + room_status[r]['clock'])
        
        curr_day = room_status[assigned_or]['day']
        curr_clock = room_status[assigned_or]['clock']
        
        # 6. คำนวณจุดเริ่ม (บวก Buffer ถ้าไม่ใช่เคสแรกของวัน)
        start_slot = curr_clock
        if start_slot > 0:
            start_slot += BUFFER_SLOTS
            
        # 7. คำนวณจุดสิ้นสุดที่เป็นไปได้ (potential_end)
        potential_end = start_slot + surgery['slots_needed']
        
        # 🌟 8. การขึ้นวันใหม่ (Overflow) - ตรรกะสำคัญจาก Notebook
        # ถ้าจัดลงไปแล้วเกินเวลาทำการปกติ ให้ไปเริ่มเช้าวันใหม่
        if potential_end > TOTAL_SLOTS:
            curr_day += 1
            start_slot = 0
            potential_end = surgery['slots_needed']
            
        # 9. บันทึกลงตาราง (เพิ่ม Service สำหรับ UI)
        OR_schedules[curr_day][assigned_or].append({
            'Encounter ID': surgery['Encounter ID'],
            'Service': surgery.get('Service', 'N/A'), # เพิ่มเพื่อให้ UI แสดงผลได้
            'Actual_Dept': surgery.get('Actual_Dept', 'N/A'), # เพิ่มเพื่อให้ UI แสดงผลได้
            'booked_time': surgery['booked_time'],
            'Weight': surgery['Weight'],
            'start_slot': start_slot,
            'end_slot': potential_end,
            'day': curr_day
        })
        
        room_status[assigned_or]['day'] = curr_day
        room_status[assigned_or]['clock'] = potential_end

    return OR_schedules, room_status
