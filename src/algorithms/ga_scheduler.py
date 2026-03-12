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
    
    for _ in range(POP_SIZE):
        # 1. สร้างลำดับเคสผ่าตัด (Permutation) 
        order = list(range(len(surgeries)))
        random.shuffle(order)
        
        # 2. การสุ่มเลือกห้องผ่าตัด (OR Assignment) - ใช้ตรรกะ List Comprehension เหมือน Jupyter
        # แต่ปรับให้เขียนอ่านง่ายขึ้นในโปรเจกต์ขนาดใหญ่
        try:
            assigned_or_list = [
                random.choice(CLUSTER_TO_ORS[surgeries[i]['cluster']]) 
                for i in order
            ]
        except KeyError as e:
            # กรณีเจอ Cluster ในข้อมูลจริงที่ไม่ตรงกับใน Config (ga_config.py)
            # ให้แสดง Error แจ้งผู้ใช้ และหยุดการทำงานเพื่อความถูกต้องของงานวิจัย
            raise KeyError(f"❌ ไม่พบชื่อ Cluster '{e.args[0]}' ใน CONFIGS ของโหมดปัจจุบัน "
                          f"กรุณาตรวจสอบการตั้งค่าในไฟล์ ga_config.py")

        # 3. บันทึกโครโมโซมเข้าสู่ประชากร 
        population.append({
            'order': order, 
            'assigned_or_list': assigned_or_list, 
            'fitness': None
        })
        
    return population


def tournament_selection(population, tournament_size, num_parents):
    selected = []
    
    # ป้องกัน ValueError จาก random.sample หาก tournament_size > len(population)
    actual_size = min(tournament_size, len(population))
    
    for _ in range(num_parents):
        # 1. สุ่มผู้ท้าชิงจากประชากร (เหมือน Jupyter)
        candidates = random.sample(population, actual_size)
        
        # 2. เลือกตัวที่มีค่า Fitness ต่ำที่สุด (Minimization Problem)
        winner = min(candidates, key=lambda ind: ind['fitness'])
        selected.append(winner)
        
    return selected


# ----------------------------------------------------
# Crossover/Mutation OPERATORS (ตาม Action ที่จะให้ Q-Agent เลือก)
# ----------------------------------------------------
def crossover_single_point(parent1, parent2):
    """
    Crossover: ใช้เทคนิค Order Crossover (OX) ตามหลักการใน Jupyter Notebook
    รักษาความต่อเนื่องของลำดับเคส (Order) และการจับคู่ห้องผ่าตัด (Assignment)
    """
    p1_order, p2_order = parent1['order'], parent2['order']
    p1_or, p2_or = parent1['assigned_or_list'], parent2['assigned_or_list']
    size = len(p1_order)
    
    # 1. Safety Check (จาก UI): ป้องกัน Error กรณีข้อมูลมีน้อย
    if size < 2:
        return copy.deepcopy(parent1)
    
    # 2. เลือกช่วง Segment (จาก Notebook): ใช้ random.sample เพื่อหาจุดเริ่มและจบ
    start, end = sorted(random.sample(range(size), 2))
    
    # 3. สร้าง Order ของลูก (OX Logic - เหมือน Notebook เป๊ะๆ)
    offspring_order = [None] * size
    segment = p1_order[start:end+1]
    offspring_order[start:end+1] = segment # คัดลอกส่วนจากพ่อมาวาง
    
    segment_set = set(segment)
    # กรองเอาเฉพาะเคสที่ยังไม่มีในตัวลูก โดยรักษาลำดับตามแม่ (Parent 2)
    fill_elements = [item for item in p2_order if item not in segment_set]
    
    ptr = (end + 1) % size
    for item in fill_elements:
        offspring_order[ptr] = item
        ptr = (ptr + 1) % size
        
    # 4. สร้าง Mapping การจับคู่ห้อง (OR Mapping Logic)
    # เพื่อให้ลูกจำได้ว่า เคส ID ไหน พ่อแม่เคยจัดให้ลงห้องไหน (ป้องกันการสลับห้องมั่ว)
    p1_map = {idx: r for idx, r in zip(p1_order, p1_or)}
    p2_map = {idx: r for idx, r in zip(p2_order, p2_or)}

    # ลูกจะรับห้องผ่าตัดตามเงื่อนไข: 
    # - ถ้าอยู่ในช่วงที่คัดลอกจากพ่อมา ให้ใช้ห้องของพ่อ
    # - ถ้านอกเหนือจากนั้น ให้ใช้ห้องของแม่
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
    Two-point Crossover: เก็บส่วนหัวและส่วนท้ายจาก P1 และเติมส่วนกลางจาก P2
    อิงตามหลักการ Jupyter Notebook โดยเพิ่มการจัดการหน่วยความจำ (Deepcopy) 
    และการตรวจสอบขนาดข้อมูลเพื่อป้องกัน Error
    """
    p1_order, p2_order = parent1['order'], parent2['order']
    p1_or, p2_or = parent1['assigned_or_list'], parent2['assigned_or_list']
    size = len(p1_order)
    
    # 1. Safety Check (จาก UI): หากข้อมูลมีน้อยเกินไป (เช่น < 4 เคส) ไม่ควรทำ 2-point
    if size < 4:
        return copy.deepcopy(parent1)
    
    # 2. เลือกจุดตัด (ตามหลักการ Jupyter: แบ่งครึ่งเพื่อหาจุดหัวและท้าย)
    # cp1 คือจุดตัดช่วงแรก, cp2 คือจุดตัดช่วงหลัง
    cp1 = random.randint(1, size // 2)
    cp2 = random.randint(size // 2, size - 1)
    
    # 3. สร้างลำดับ (Order) สำหรับลูก
    offspring_order = [None] * size
    # คัดลอกส่วนหัวและส่วนท้ายจาก Parent 1 (หัวใจหลักของงานวิจัย)
    offspring_order[:cp1] = p1_order[:cp1]
    offspring_order[cp2:] = p1_order[cp2:]
    
    # กรองเอาเฉพาะเคสที่ยังไม่ได้ถูกเลือกใส่ในตัวลูก (ตัด None ออกก่อนทำ Set)
    p1_segment_set = {item for item in offspring_order if item is not None}
    
    # ดึงเคสที่เหลือจาก Parent 2 โดยรักษาลำดับเดิมไว้
    fill_elements = [item for item in p2_order if item not in p1_segment_set]
    
    # เติมส่วนที่ว่าง (None) ด้วยสมาชิกจาก Parent 2 ตามลำดับ (เหมือน Jupyter)
    ptr = 0
    for item in fill_elements:
        while ptr < size and offspring_order[ptr] is not None:
            ptr += 1
        if ptr < size:
            offspring_order[ptr] = item

    # 4. Mapping ห้องผ่าตัด (OR Assignment Logic)
    # เพื่อให้ลูกจำได้ว่า Case ID นี้ พ่อหรือแม่เคยจัดลงห้องไหน
    p1_map = {idx: r for idx, r in zip(p1_order, p1_or)}
    p2_map = {idx: r for idx, r in zip(p2_order, p2_or)}
    
    # ตรรกะการสืบทอดห้อง:
    # - ส่วนหัวและท้าย (ที่มาจาก P1) -> ใช้ห้องของ P1
    # - ส่วนกลาง (ที่มาจาก P2) -> ใช้ห้องของ P2
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
    Mutation: การกลายพันธุ์เพื่อรักษาความหลากหลายทางพันธุกรรม (Diversity)
    ประกอบด้วย 2 ส่วน: สลับลำดับคิว และ สุ่มเปลี่ยนห้องผ่าตัด
    """
    new_order = individual['order'][:]
    new_or_list = individual['assigned_or_list'][:]
    size = len(new_order)
    
    if size < 2:
        return individual

    # 1. Swap Mutation: สุ่มสลับคิวงาน 2 งาน (เหมือนกันทั้ง UI และ Notebook)
    if random.random() < rate:
        i, j = random.sample(range(size), 2)
        new_order[i], new_order[j] = new_order[j], new_order[i]
        new_or_list[i], new_or_list[j] = new_or_list[j], new_or_list[i] 

    # 2. Assignment Mutation: สุ่มเปลี่ยนห้องผ่าตัด (ยังคงรักษาเงื่อนไข Cluster)
    if random.random() < rate:
        # 🌟 จุดตัดสินใจ: ใน Notebook คุณใช้ค่าคงที่ 5 
        # แต่แนะนำให้ใช้ % ของจำนวนเคส (เช่น 5% ของงานทั้งหมด) 
        # เพื่อให้การกลายพันธุ์ไม่ "ทำลาย" โครงสร้างที่ดีเดิมมากเกินไปในชุดข้อมูลเล็ก
        num_mutations = max(1, int(size * 0.05)) 
        
        for _ in range(num_mutations):
            mut_idx = random.randrange(size)
            case_idx = new_order[mut_idx]
            case_cluster = surgeries[case_idx]['cluster'] # เข้าถึงแบบ Notebook
            
            # ตรวจสอบความถูกต้องของ Cluster ก่อนสุ่มห้องใหม่
            if case_cluster in CLUSTER_TO_ORS and CLUSTER_TO_ORS[case_cluster]:
                new_or_list[mut_idx] = random.choice(CLUSTER_TO_ORS[case_cluster])
    
    return {
        'order': new_order, 
        'assigned_or_list': new_or_list, 
        'fitness': None
    }


def standard_ga_crossover(parent1, parent2):
    """
    Standard GA Crossover: อิงตาม Notebook 100%
    ใช้ OX สำหรับลำดับ และ Mapping สำหรับห้องผ่าตัด
    """
    size = len(parent1['order'])
    if size < 2:
        return copy.deepcopy(parent1)

    # 1. เลือกช่วง Segment จาก Parent 1
    start, end = sorted(random.sample(range(size), 2))
    
    # 2. Crossover ส่วนลำดับ (Order) - OX Technique
    child_order = [None] * size
    segment = parent1['order'][start:end+1]
    child_order[start:end+1] = segment
    
    set_segment = set(segment)
    # กรองเอาเฉพาะตัวที่ Parent 2 มีแต่ใน segment ไม่มี
    remaining = [item for item in parent2['order'] if item not in set_segment]
    
    ptr = (end + 1) % size
    for item in remaining:
        child_order[ptr] = item
        ptr = (ptr + 1) % size
        
    # 3. Crossover ส่วนห้อง (Room Assignment) - Map Inheritance
    child_or_list = [None] * size
    p1_or_map = {case_idx: r for case_idx, r in zip(parent1['order'], parent1['assigned_or_list'])}
    p2_or_map = {case_idx: r for case_idx, r in zip(parent2['order'], parent2['assigned_or_list'])}
    
    for i in range(size):
        case_idx = child_order[i]
        # logic: ถ้าตำแหน่งอยู่ในช่วง segment ให้ใช้ห้องจาก P1, นอกนั้นใช้ห้องจาก P2 (ตาม Case ID)
        if start <= i <= end:
            child_or_list[i] = p1_or_map[case_idx]
        else:
            child_or_list[i] = p2_or_map[case_idx]
            
    return {'order': child_order, 'assigned_or_list': child_or_list, 'fitness': None}


def standard_ga_mutation(individual, mutation_rate, surgeries, CLUSTER_TO_ORS):
    """
    Standard GA Mutation: อิงตาม Notebook 100%
    ใช้ Inversion Mutation (กลับด้าน) และ Random Assignment Mutation
    """
    mutated_order = individual['order'][:]
    mutated_or_list = individual['assigned_or_list'][:]
    size = len(mutated_order)

    # 1. Mutation ลำดับ (Order) - Inversion Technique
    if random.random() < mutation_rate:
        start, end = sorted(random.sample(range(size), 2))
        
        # จำกัดระยะไม่ให้พังคำตอบเดิมเกินไป (Capped at 50 จาก Notebook)
        if end - start > 50: 
            end = start + 50   
            
        # กลับด้านส่วนที่เลือก (ทั้งลำดับและห้องเพื่อให้สอดคล้องกัน)
        mutated_order[start:end] = mutated_order[start:end][::-1]
        mutated_or_list[start:end] = mutated_or_list[start:end][::-1]

    # 2. Mutation การเลือกห้อง (OR Assignment) - Random Points
    if random.random() < mutation_rate:
        # สุ่มแก้ 1-10 จุด (จาก Notebook)
        num_mutations = random.randint(1, 10) 
        for _ in range(num_mutations):
            idx = random.randrange(size)
            case_idx = mutated_order[idx]
            case_cluster = surgeries[case_idx]['cluster']
            
            if case_cluster in CLUSTER_TO_ORS:
                possible_ors = CLUSTER_TO_ORS[case_cluster]
                mutated_or_list[idx] = random.choice(possible_ors)
    
    return {
        'order': mutated_order, 
        'assigned_or_list': mutated_or_list, 
        'fitness': None
    }


# ----------------------------------------------------
# RUN (Standard GA และ Hybrid GA-Q)
# ----------------------------------------------------

def run_ga_standard(surgeries, num_gen, pop_size, total_slots, mode, patience=50, st_progress=None):
    """
    Standard GA: อิงตามหลักการ Jupyter Notebook 100% 
    เพิ่มระบบ Early Stopping และใช้ตรรกะ Inversion Mutation
    """
    # 1. เตรียมข้อมูลพื้นฐานตาม Mode
    current_cfg = CONFIGS[mode]
    cluster_mapping = current_cfg["CLUSTER_TO_ORS"]
    all_or_ids = [or_id for ors in cluster_mapping.values() for or_id in ors]
    buffer_slots = 1 # หรือดึงจาก CONFIGS

    # 2. ตั้งค่าการหยุดก่อนกำหนด (Early Stopping)
    no_improvement_count = 0
    best_so_far = float('inf')
    stop_gen = num_gen - 1
    history = []

    # 3. สร้างประชากรเริ่มต้นและคำนวณ Fitness
    population = generate_initial_population(surgeries, pop_size, cluster_mapping)
    for ind in population:
        sched, status = decode_individual(ind, surgeries, all_or_ids, total_slots, buffer_slots)
        ind['fitness'] = evaluate_fitness(sched, status, total_slots, W_MAKESPAN, W_OVERTIME, W_IMBALANCE)

    # 4. EVOLUTIONARY CYCLE
    for gen in range(num_gen):
        # เรียงลำดับ (Fitness น้อย = เก่ง)
        population.sort(key=lambda x: x['fitness'])
        current_best_fitness = population[0]['fitness']
        history.append(current_best_fitness)

        # ตรรกะ Early Stopping (อิงตาม Notebook)
        if current_best_fitness < best_so_far:
            best_so_far = current_best_fitness
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            stop_gen = gen
            remaining_gens = num_gen - (gen + 1)
            history.extend([current_best_fitness] * remaining_gens)
            break

        # ELITISM: เก็บตัวที่ดีที่สุดไว้
        next_gen = [copy.deepcopy(ind) for ind in population[:NUM_ELITES]]
        
        # SELECTION: คัดเลือกพ่อแม่
        num_parents = int(pop_size * 0.5)
        parents = tournament_selection(population, TOURNAMENT_SIZE, num_parents)
        
        # REPRODUCTION: สร้างลูกหลาน
        while len(next_gen) < pop_size:
            # ใช้ random.sample เพื่อไม่ให้ได้พ่อแม่คนเดียวกัน (อิงตาม Notebook)
            p1, p2 = random.sample(parents, 2)
            
            # --- Crossover ---
            if random.random() < CROSSOVER_RATE:
                child = standard_ga_crossover(p1, p2) 
            else:
                child = {'order': p1['order'][:], 'assigned_or_list': p1['assigned_or_list'][:], 'fitness': None}
            
            # --- Mutation ---
            # ใช้ตรรกะ Inversion Mutation จาก Notebook
            child = standard_ga_mutation(child, MUTATION_RATE, surgeries, cluster_mapping)
            
            # --- Evaluation ---
            sched, status = decode_individual(child, surgeries, all_or_ids, total_slots, buffer_slots)
            child['fitness'] = evaluate_fitness(sched, status, total_slots, W_MAKESPAN, W_OVERTIME, W_IMBALANCE)
            next_gen.append(child)
        
        population = next_gen

        # อัปเดต Progress Bar บน UI
        if st_progress:
            st_progress.progress((gen + 1) / num_gen)

    # 5. คืนค่าผลลัพธ์ที่ดีที่สุด
    final_best = min(population, key=lambda x: x['fitness'])
    final_sched, final_status = decode_individual(final_best, surgeries, all_or_ids, total_slots, buffer_slots)
    
    return final_best, history, final_sched, final_status

def run_ga_hybrid_q(surgeries, num_gen, pop_size, total_slots, mode, patience=50, st_progress=None):
    """
    Hybrid GA-Q-learning: อิงตามหลักการ Jupyter Notebook 100%
    ประสานการทำงานระหว่าง Genetic Algorithm และ Q-Learning เพื่อปรับตัวแปรอัตโนมัติ
    """
    # 1. เตรียมค่า Config ตามโหมดการทดลอง
    current_cfg = CONFIGS[mode]
    cluster_mapping = current_cfg["CLUSTER_TO_ORS"]
    all_or_ids = [or_id for ors in cluster_mapping.values() for or_id in ors]
    
    # 2. เริ่มต้นระบบ (เหมือนใน Notebook)
    population = generate_initial_population(surgeries, pop_size, cluster_mapping)
    q_table = initialize_q_table()
    epsilon = EPSILON_START
    stop_gen = num_gen - 1
    
    # 3. Initial Evaluation
    for ind in population:
        sched, status = decode_individual(ind, surgeries, all_or_ids, total_slots, BUFFER_SLOTS)
        ind['fitness'] = evaluate_fitness(sched, status, total_slots, 
                                        W_MAKESPAN, 
                                        W_OVERTIME, 
                                        W_IMBALANCE)

    # คำนวณขีดจำกัดความหลากหลายเริ่มต้น (Diversity Threshold)
    fitness_var_threshold = np.var([ind['fitness'] for ind in population]) * FITNESS_VAR_THRESHOLD_FACTOR
    
    no_improvement_count = 0
    best_fitness_history = []
    old_best_fitness = min(population, key=lambda ind: ind['fitness'])['fitness']
    best_fitness_history.append(old_best_fitness)

    # 4. EVOLUTIONARY CYCLE WITH Q-LEARNING
    for gen in range(num_gen):
        # --- Q-LEARNING: Observation & Decision ---
        current_state = get_state(population, fitness_var_threshold)
        
        # select_action logic (ε-greedy)
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_table[current_state])
            
        ops = OPERATOR_MAP[action]

        # --- GA: Evolution (Reproduction) ---
        population.sort(key=lambda ind: ind['fitness'])
        next_population = [copy.deepcopy(ind) for ind in population[:NUM_ELITES]] # Elitism
        
        parents = tournament_selection(population, TOURNAMENT_SIZE, int(pop_size * 0.5))
        
        while len(next_population) < pop_size:
            # ใช้ random.sample เพื่อความหลากหลาย (เหมือน Notebook)
            p1, p2 = random.sample(parents, 2)
            
            # Crossover based on action
            if random.random() < CROSSOVER_RATE:
                if ops['crossover'] == 'single':
                    offspring = crossover_single_point(p1, p2)
                else:
                    offspring = crossover_two_point(p1, p2)
            else:
                offspring = {'order': p1['order'][:], 'assigned_or_list': p1['assigned_or_list'][:], 'fitness': None}
            
            # Mutation based on action rate
            offspring = mutate_with_rate(offspring, surgeries, ops['mutation_rate'], cluster_mapping)
            
            # Evaluation ของลูก
            child_sched, child_status = decode_individual(offspring, surgeries, all_or_ids, total_slots, BUFFER_SLOTS)
            offspring['fitness'] = evaluate_fitness(child_sched, child_status, total_slots, 
                                                 W_MAKESPAN, 
                                                 W_OVERTIME, 
                                                 W_IMBALANCE)
            next_population.append(offspring)

        # --- Q-LEARNING: Update Knowledge ---
        new_best_fitness = min(next_population, key=lambda ind: ind['fitness'])['fitness']
        reward = old_best_fitness - new_best_fitness # Reward คือค่าความต่างของ Fitness ที่ดีขึ้น
        
        # ตรรกะ Early Stopping (อิงตาม Notebook)
        if new_best_fitness < old_best_fitness:
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        next_state = get_state(next_population, fitness_var_threshold)
        
        # Bellman Equation Update (update_q_values logic)
        old_q = q_table[current_state, action]
        max_future_q = np.max(q_table[next_state])
        q_table[current_state, action] = (1 - ALPHA) * old_q + ALPHA * (reward + GAMMA * max_future_q)

        # อัปเดตพารามิเตอร์สำหรับรอบถัดไป
        population = next_population
        old_best_fitness = new_best_fitness
        epsilon = max(0.01, epsilon * EPSILON_DECAY)
        best_fitness_history.append(new_best_fitness)

        # ตรวจสอบ Early Stopping
        if no_improvement_count >= patience:
            stop_gen = gen
            remaining_gens = num_gen - (gen + 1)
            best_fitness_history.extend([new_best_fitness] * remaining_gens)
            break

        # อัปเดต UI Progress Bar
        if st_progress:
            st_progress.progress((gen + 1) / num_gen)

    # 5. คืนค่าคำตอบที่ดีที่สุด
    final_best = min(population, key=lambda ind: ind['fitness'])
    final_sched, final_status = decode_individual(final_best, surgeries, all_or_ids, total_slots, BUFFER_SLOTS)
    
    return final_best, best_fitness_history, final_sched, final_status, stop_gen
