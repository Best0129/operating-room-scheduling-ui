import random
import copy
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from src.utils import decode_individual, evaluate_fitness
from config.ga_config import * 

# ----------------------------------------------------
def initialize_q_table():
    # Q-table: 2 States (High/Low) x 4 Actions
    return np.zeros((2, 4))


def get_state(population, fitness_var_threshold):
    # ดึงค่า Fitness ทั้งหมดออกมา
    fitness_values = [ind['fitness'] for ind in population if ind['fitness'] is not None]
    if not fitness_values: return 1 # ถ้ายังไม่มีค่า ให้มองว่า Diversity ต่ำไว้ก่อน
    
    # คำนวณ CV = SD / Mean
    mean_f = np.mean(fitness_values)
    std_f = np.std(fitness_values)
    cv = std_f / mean_f if mean_f != 0 else 0
    
    # State 0: High Diversity (ค้นหาต่อไป)
    # State 1: Low Diversity (ติดหล่ม - ต้องทำ Mutation หนักๆ)
    return 0 if cv > fitness_var_threshold else 1


# Q-Action Mapping
OPERATOR_MAP = {
    0: {'crossover': 'single', 'mutation_rate': 0.01}, # Single-point + low mutation
    1: {'crossover': 'two',    'mutation_rate': 0.01}, # Two-point + low mutation
    2: {'crossover': 'single', 'mutation_rate': 0.10}, # Single-point + high mutation
    3: {'crossover': 'two',    'mutation_rate': 0.10}, # Two-point + high mutation
}


# ----------------------------------------------------
# CORE GA (ใช้ร่วมกัน)
# ----------------------------------------------------
def generate_initial_population(surgeries, POP_SIZE, CLUSTER_TO_ORS):
    population = []
    
    # ดึงคีย์ทั้งหมดที่มีใน CLUSTER_TO_ORS มาเตรียมไว้เป็นค่าเริ่มต้นกรณีหา Cluster ไม่เจอ
    available_clusters = list(CLUSTER_TO_ORS.keys())
    
    for _ in range(POP_SIZE):
        # 1. สร้างลำดับเคสผ่าตัด (Permutation)
        order = list(range(len(surgeries)))
        random.shuffle(order)
        
        # 2. การสุ่มเลือกห้องผ่าตัด (OR Assignment)
        assigned_or_list = []
        for i in order:
            target_cluster = surgeries[i].get('cluster')
            
            # ตรวจสอบความถูกต้องของ Cluster ตามที่คุณเขียนมา [cite: 2026-03-01, 2026-03-10]
            if target_cluster in CLUSTER_TO_ORS and CLUSTER_TO_ORS[target_cluster]:
                assigned_or_list.append(random.choice(CLUSTER_TO_ORS[target_cluster]))
            else:
                # 🌟 Fallback Strategy: หากข้อมูลผิดพลาด ให้สุ่มจาก Cluster แรกที่มี
                # เพื่อให้ Algorithm ยังรันต่อไปได้โดยไม่พังกลางคัน
                fallback_cluster = available_clusters[0]
                assigned_or_list.append(random.choice(CLUSTER_TO_ORS[fallback_cluster]))
        
        # 3. บันทึกโครโมโซมเข้าสู่ประชากร
        population.append({
            'order': order, 
            'assigned_or_list': assigned_or_list, 
            'fitness': None
        })
        
    return population


def tournament_selection(population, tournament_size, num_parents):
    selected = []
    
    # จุดที่เพิ่ม: ป้องกันกรณี Tournament Size ใหญ่กว่าจำนวนประชากร
    # หากตั้งค่ามาเกิน ให้ใช้ขนาดประชากรทั้งหมดแทน
    actual_size = min(tournament_size, len(population))
    
    for _ in range(num_parents):
        # สุ่มตัวอย่างผู้ท้าชิง
        candidates = random.sample(population, actual_size)
        
        # เลือกตัวที่เก่งที่สุด (Fitness น้อยที่สุด = Penalty ต่ำที่สุด)
        # เนื่องจากงานของเราเป็นการลด Penalty (Minimization Problem) [cite: 2026-03-01]
        best_candidate = min(candidates, key=lambda ind: ind['fitness'])
        selected.append(best_candidate)
        
    return selected


# ----------------------------------------------------
# Crossover/Mutation OPERATORS (ตาม Action ที่จะให้ Q-Agent เลือก)
# ----------------------------------------------------
def crossover_single_point(parent1, parent2):
    """
    Order Crossover (OX): รักษาลำดับของงานผ่าตัดและห้องผ่าตัดที่ได้รับมอบหมาย
    เหมาะสำหรับทั้ง Experiment 1 และ 2 [cite: 2026-03-01]
    """
    p1_order, p2_order = parent1['order'], parent2['order']
    p1_or, p2_or = parent1['assigned_or_list'], parent2['assigned_or_list']
    size = len(p1_order)
    
    # 1. Safety check: หากมีข้อมูลน้อยเกินไป ไม่ต้อง Crossover
    if size < 2:
        return copy.deepcopy(parent1)
    
    # 2. เลือกช่วง (Segment) สำหรับการสืบทอด
    start, end = sorted(random.sample(range(size), 2))
    
    # 3. จัดการส่วน 'order' (OX Logic)
    offspring_order = [None] * size
    segment = p1_order[start:end+1]
    offspring_order[start:end+1] = segment
    
    segment_set = set(segment)
    # กรองเอาเฉพาะตัวที่ไม่มีใน segment เพื่อนำมาเติมส่วนที่เหลือ
    fill_elements = [item for item in p2_order if item not in segment_set]
    
    ptr = (end + 1) % size
    for item in fill_elements:
        offspring_order[ptr] = item
        ptr = (ptr + 1) % size
        
    # 4. จัดการส่วน 'assigned_or_list' (Mapping Logic)
    # สร้าง Map เพื่อจำว่า ID ผ่าตัดนี้ เดิมที Parent แต่ละคนให้ไปอยู่ห้องไหน [cite: 2026-03-10]
    p1_map = {case_idx: room_id for case_idx, room_id in zip(p1_order, p1_or)}
    p2_map = {case_idx: room_id for case_idx, room_id in zip(p2_order, p2_or)}

    # ลูกจะรับห้องผ่าตัดตาม ID งานผ่าตัด:
    # - ถ้างานนั้นอยู่ใน segment จะเอาห้องจาก Parent 1
    # - ถ้างานนั้นอยู่นอก segment จะเอาห้องจาก Parent 2
    offspring_assigned_or = [
        p1_map[offspring_order[i]] if start <= i <= end else p2_map[offspring_order[i]]
        for i in range(size)
    ]
    
    return {
        'order': offspring_order, 
        'assigned_or_list': offspring_assigned_or, 
        'fitness': None
    }


def crossover_two_point(parent1, parent2):
    """
    Two-point Crossover (OX variant): รับส่วนหัวและท้ายจาก P1 
    และเติมส่วนกลางด้วยลำดับที่เหลือจาก P2
    """
    p1_order, p2_order = parent1['order'], parent2['order']
    p1_or, p2_or = parent1['assigned_or_list'], parent2['assigned_or_list']
    size = len(p1_order)
    
    # 1. Safety Check: ป้องกัน Error สำหรับข้อมูลขนาดเล็ก
    if size < 4:
        return copy.deepcopy(parent1)
    
    # 2. เลือกจุดตัด (Crossover Points)
    cp1 = random.randint(1, max(2, size // 3))
    cp2 = random.randint(min(size - 2, (2 * size) // 3), size - 1)
    
    offspring_order = [None] * size
    # คัดลอกส่วนหัวและท้ายจาก P1
    offspring_order[:cp1] = p1_order[:cp1]
    offspring_order[cp2:] = p1_order[cp2:]
    
    # 3. กรองตัวที่เหลือจาก P2 (ไม่เอาตัวที่ซ้ำกับที่คัดลอกมาแล้ว)
    # กรองเฉพาะค่าที่ไม่ใช่ None เพื่อทำความสะอาด Set
    p1_segment_set = {item for item in offspring_order if item is not None}
    fill_elements = [item for item in p2_order if item not in p1_segment_set]
    
    # เติมส่วนที่ว่าง (None) ด้วยสมาชิกจาก P2 ตามลำดับ
    ptr = 0
    for item in fill_elements:
        while ptr < size and offspring_order[ptr] is not None:
            ptr += 1
        if ptr < size:
            offspring_order[ptr] = item

    # 4. Mapping ห้องผ่าตัด (รักษา Cluster Constraints) [cite: 2026-03-10]
    p1_map = {idx: r for idx, r in zip(p1_order, p1_or)}
    p2_map = {idx: r for idx, r in zip(p2_order, p2_or)}
    
    offspring_assigned_or = [
        p1_map[offspring_order[i]] if (i < cp1 or i >= cp2) else p2_map[offspring_order[i]]
        for i in range(size)
    ]
    
    return {
        'order': offspring_order, 
        'assigned_or_list': offspring_assigned_or, 
        'fitness': None
    }


def mutate_with_rate(individual, surgeries, rate, CLUSTER_TO_ORS):
    """
    Mutation: เพิ่มความหลากหลายให้กับประชากร
    รองรับทั้งการสลับลำดับและการเปลี่ยนห้องผ่าตัดตามเงื่อนไขของแต่ละ Dataset [cite: 2026-03-01]
    """
    new_order = individual['order'][:]
    new_or_list = individual['assigned_or_list'][:]
    size = len(new_order)
    
    if size < 2:
        return individual

    # 1. Swap Mutation: สลับลำดับคิวงาน
    if random.random() < rate:
        i, j = random.sample(range(size), 2)
        new_order[i], new_order[j] = new_order[j], new_order[i]
        new_or_list[i], new_or_list[j] = new_or_list[j], new_or_list[i] 

    # 2. Assignment Mutation: สุ่มเปลี่ยนห้องผ่าตัด (ยังคงรักษาเงื่อนไข Cluster) [cite: 2026-03-10]
    if random.random() < rate:
        # ปรับจำนวนจุดที่แก้ตามขนาดข้อมูล (ประมาณ 5% ของจำนวนเคส หรืออย่างน้อย 1 จุด)
        num_mutations = max(1, int(size * 0.05))
        
        for _ in range(num_mutations):
            mut_idx = random.randrange(size)
            case_idx = new_order[mut_idx]
            case_cluster = surgeries[case_idx].get('cluster')
            
            # ตรวจสอบว่า Cluster นี้มีห้องผ่าตัดรองรับในการทดลองปัจจุบันหรือไม่
            if case_cluster in CLUSTER_TO_ORS and CLUSTER_TO_ORS[case_cluster]:
                new_or_list[mut_idx] = random.choice(CLUSTER_TO_ORS[case_cluster])
    
    return {
        'order': new_order, 
        'assigned_or_list': new_or_list, 
        'fitness': None
    }


# ----------------------------------------------------
# RUN (Standard GA และ Hybrid GA-Q)
# ----------------------------------------------------

def run_ga_standard(surgeries, num_gen, pop_size, total_slots, mode, st_progress=None):
    """
    Standard GA: รันการจัดตารางแบบมาตรฐาน รองรับ 2 ชุดข้อมูลและ Streamlit UI [cite: 2026-03-07]
    """
    # 1. ดึงค่า Config เฉพาะของโหมดนั้นๆ มาเตรียมไว้ [cite: 2026-03-01]
    current_cfg = CONFIGS[mode]
    cluster_mapping = current_cfg["CLUSTER_TO_ORS"]
    all_or_ids = [or_id for ors in cluster_mapping.values() for or_id in ors]

    # 2. สร้างประชากรเริ่มต้น (ส่ง cluster_mapping เข้าไปแก้ KeyError)
    population = generate_initial_population(surgeries, pop_size, cluster_mapping)
    
    # 3. Initial Evaluation
    for individual in population:
        # แก้ไขการรับค่าจาก decode_individual (คืนค่า sched และ status) [cite: 2026-03-07]
        OR_schedules, room_status = decode_individual(
            individual, surgeries, all_or_ids, total_slots, BUFFER_SLOTS
        )
        # คำนวณ Fitness (ส่ง room_status และ Weights ให้ครบถ้วน) [cite: 2026-03-01]
        individual['fitness'] = evaluate_fitness(
            OR_schedules, room_status, total_slots, W_MAKESPAN, W_OVERTIME, W_IMBALANCE
        ) 

    best_fitness_history = []
    
    # 4. EVOLUTIONARY CYCLE
    # เปลี่ยนจาก tqdm เป็น range ปกติเพื่อใช้ st_progress บน UI [cite: 2026-03-07]
    for gen in range(num_gen):
        population.sort(key=lambda ind: ind['fitness'])
        
        # Elitism: เก็บตัวที่ดีที่สุดไว้
        elites = population[:NUM_ELITES] 
        next_population = copy.deepcopy(elites)
        
        # Selection: คัดเลือกพ่อแม่
        num_parents = int(pop_size * 0.5) 
        parents = tournament_selection(population, TOURNAMENT_SIZE, num_parents)
        
        while len(next_population) < pop_size:
            parent1, parent2 = random.choice(parents), random.choice(parents) 

            # --- Crossover (Fixed Rate) ---
            if random.random() < CROSSOVER_RATE:
                # ใช้ Two-point Crossover ตามที่คุณต้องการ
                offspring = crossover_two_point(parent1, parent2) 
            else:
                offspring = copy.deepcopy(parent1)
            
            # --- Mutation (Fixed Rate) ---
            # แก้ไข: ส่ง cluster_mapping เข้าไปด้วยเพื่อให้สุ่มห้องได้ถูกต้องตาม Dataset [cite: 2026-03-10]
            offspring = mutate_with_rate(offspring, surgeries, MUTATION_RATE, cluster_mapping) 
            
            # --- Evaluation ของลูกที่เกิดใหม่ ---
            child_sched, child_status = decode_individual(
                offspring, surgeries, all_or_ids, total_slots, BUFFER_SLOTS
            )
            offspring['fitness'] = evaluate_fitness(
                child_sched, child_status, total_slots, W_MAKESPAN, W_OVERTIME, W_IMBALANCE
            )
            
            next_population.append(offspring)

        # 5. สลับประชากรและเก็บประวัติ
        population = next_population
        current_best = min(population, key=lambda ind: ind['fitness'])
        best_fitness_history.append(current_best['fitness'])

        # ส่วนอัปเดต UI Progress Bar [cite: 2026-03-07]
        if st_progress:
            st_progress.progress((gen + 1) / num_gen)

    # 6. คืนค่าผลลัพธ์ที่ดีที่สุด
    final_best_individual = min(population, key=lambda ind: ind['fitness'])
    final_sched, final_status = decode_individual(
        final_best_individual, surgeries, all_or_ids, total_slots, BUFFER_SLOTS
    )
    
    return final_best_individual, best_fitness_history, final_sched, final_status


def run_ga_hybrid_q(surgeries, num_gen, pop_size, total_slots, mode, st_progress=None):
    """
    Hybrid GA-Q: ใช้ Q-Learning ในการเลือก Operator ให้เหมาะสมกับสถานะประชากร
    รองรับ Experiment 1 & 2 และการแสดงผลบน Streamlit
    """
    # 1. เตรียมค่า Config ตามโหมดการทดลอง
    current_cfg = CONFIGS[mode]
    cluster_mapping = current_cfg["CLUSTER_TO_ORS"]
    all_or_ids = [or_id for ors in cluster_mapping.values() for or_id in ors]
    
    # 2. เริ่มต้นระบบ GA และ Q-Learning
    population = generate_initial_population(surgeries, pop_size, cluster_mapping)
    q_table = initialize_q_table()
    epsilon = EPSILON_START
    
    # 3. Initial Evaluation
    for individual in population:
        # ใช้ decode และ evaluate เวอร์ชั่นใหม่ [cite: 2026-03-07]
        OR_schedules, room_status = decode_individual(
            individual, surgeries, all_or_ids, total_slots, BUFFER_SLOTS
        )
        individual['fitness'] = evaluate_fitness(
            OR_schedules, room_status, total_slots, W_MAKESPAN, W_OVERTIME, W_IMBALANCE
        )

    # คำนวณ Threshold สำหรับการแบ่ง State ของ Q-Learning (High/Low Diversity)
    initial_var = np.var([ind['fitness'] for ind in population]) 
    fitness_var_threshold = initial_var * FITNESS_VAR_THRESHOLD_FACTOR
    
    best_fitness_history = []
    old_best_fitness = min(population, key=lambda ind: ind['fitness'])['fitness'] 
    best_fitness_history.append(old_best_fitness) 
    
    # 4. EVOLUTIONARY CYCLE WITH Q-LEARNING
    for gen in range(num_gen):
        
        # --- Q-LEARNING: สังเกตสถานะและเลือก Action ---
        current_state = get_state(population, fitness_var_threshold)
        
        # ε-greedy Strategy: เลือก Action เพื่อ Explore หรือ Exploit [cite: 2026-03-01]
        if random.random() < epsilon:
            action = random.randint(0, 3) 
        else:
            action = np.argmax(q_table[current_state]) 
            
        # ดึง Operators (Crossover/Mutation Rate) ตาม Action ที่เลือก
        operators = OPERATOR_MAP[action]
        crossover_type = operators['crossover']
        mutation_rate = operators['mutation_rate'] 
        
        # --- GA: การวิวัฒนาการ ---
        population.sort(key=lambda ind: ind['fitness'])
        elites = population[:NUM_ELITES] 
        next_population = copy.deepcopy(elites)
        
        num_parents = int(pop_size * 0.5) 
        parents = tournament_selection(population, TOURNAMENT_SIZE, num_parents)
        
        while len(next_population) < pop_size:
            parent1, parent2 = random.choice(parents), random.choice(parents) 

            # 🌟 Crossover: เลือกใช้ตามที่ Q-Agent แนะนำ
            if random.random() < CROSSOVER_RATE:
                if crossover_type == 'single':
                    # เราปรับเป็น crossover_order (OX) เพื่อประสิทธิภาพสูงสุด
                    offspring = crossover_single_point(parent1, parent2)
                else:
                    offspring = crossover_two_point(parent1, parent2)
            else:
                offspring = copy.deepcopy(parent1)
            
            # 🌟 Mutation: ใช้ Rate ที่ Q-Agent กำหนด และส่ง cluster_mapping ให้ถูกต้อง
            offspring = mutate_with_rate(offspring, surgeries, mutation_rate, cluster_mapping) 
            
            # Evaluation ของลูก
            child_sched, child_status = decode_individual(
                offspring, surgeries, all_or_ids, total_slots, BUFFER_SLOTS
            )
            offspring['fitness'] = evaluate_fitness(
                child_sched, child_status, total_slots, W_MAKESPAN, W_OVERTIME, W_IMBALANCE
            )
            
            next_population.append(offspring)

        population = next_population
        
        # --- Q-LEARNING: อัปเดตความรู้ (Update Q-Table) ---
        new_best_fitness = min(population, key=lambda ind: ind['fitness'])['fitness']
        
        # Reward: ยิ่งค่า Fitness ลดลงมาก (ดีขึ้น) ยิ่งได้รางวัลมาก [cite: 2026-03-01]
        reward = old_best_fitness - new_best_fitness 
        
        next_state = get_state(population, fitness_var_threshold)
        
        # Q-learning Update Equation (Bellman Equation)
        old_q_value = q_table[current_state, action]
        max_future_q = np.max(q_table[next_state])
        new_q_value = (1 - ALPHA) * old_q_value + ALPHA * (reward + GAMMA * max_future_q)
        q_table[current_state, action] = new_q_value
        
        # อัปเดตพารามิเตอร์สำหรับรอบถัดไป
        old_best_fitness = new_best_fitness
        epsilon = max(0.01, epsilon * EPSILON_DECAY) 

        # 🌟 อัปเดต UI Progress Bar [cite: 2026-03-07]
        if st_progress:
            st_progress.progress((gen + 1) / num_gen)

        best_fitness_history.append(new_best_fitness)

    # 5. คืนค่าคำตอบที่ดีที่สุด
    final_best_individual = min(population, key=lambda ind: ind['fitness'])
    final_sched, final_status = decode_individual(
        final_best_individual, surgeries, all_or_ids, total_slots, BUFFER_SLOTS
    )
    
    return final_best_individual, best_fitness_history, final_sched, final_status
