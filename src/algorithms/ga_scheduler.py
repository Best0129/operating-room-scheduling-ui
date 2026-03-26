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
        # สร้างลำดับเคสผ่าตัด (Permutation) 
        order = list(range(len(surgeries)))
        random.shuffle(order)

        try:
            assigned_or_list = [
                random.choice(CLUSTER_TO_ORS[surgeries[i]['cluster']]) 
                for i in order
            ]
        except KeyError as e:
            raise KeyError(f"ไม่พบชื่อ Cluster '{e.args[0]}'"
                          f"กรุณาตรวจสอบการตั้งค่าในไฟล์ ga_config.py")
        population.append({
            'order': order, 
            'assigned_or_list': assigned_or_list, 
            'fitness': None
        })
    return population


def tournament_selection(population, tournament_size, num_parents):
    selected = []
    actual_size = min(tournament_size, len(population))
    for _ in range(num_parents):
        candidates = random.sample(population, actual_size)
        winner = min(candidates, key=lambda ind: ind['fitness'])
        selected.append(winner)
    return selected


# ----------------------------------------------------
# Crossover/Mutation OPERATORS (ตาม Action ที่จะให้ Q-Agent เลือก)
# ----------------------------------------------------
def crossover_single_point(parent1, parent2):
    p1_order, p2_order = parent1['order'], parent2['order']
    p1_or, p2_or = parent1['assigned_or_list'], parent2['assigned_or_list']
    size = len(p1_order)

    if size < 2:
        return copy.deepcopy(parent1)

    start, end = sorted(random.sample(range(size), 2))

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
    p1_order, p2_order = parent1['order'], parent2['order']
    p1_or, p2_or = parent1['assigned_or_list'], parent2['assigned_or_list']
    size = len(p1_order)
    
    if size < 4:
        return copy.deepcopy(parent1)
    
    # cp1 คือจุดตัดช่วงแรก, cp2 คือจุดตัดช่วงหลัง
    cp1 = random.randint(1, size // 2)
    cp2 = random.randint(size // 2, size - 1)
    
    # สร้างลำดับ (Order) สำหรับลูก
    offspring_order = [None] * size
    offspring_order[:cp1] = p1_order[:cp1]
    offspring_order[cp2:] = p1_order[cp2:]
    p1_segment_set = {item for item in offspring_order if item is not None}
    fill_elements = [item for item in p2_order if item not in p1_segment_set]
    
    ptr = 0
    for item in fill_elements:
        while ptr < size and offspring_order[ptr] is not None:
            ptr += 1
        if ptr < size:
            offspring_order[ptr] = item

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
    new_order = individual['order'][:]
    new_or_list = individual['assigned_or_list'][:]
    size = len(new_order)
    
    if size < 2:
        return individual

    if random.random() < rate:
        i, j = random.sample(range(size), 2)
        new_order[i], new_order[j] = new_order[j], new_order[i]
        new_or_list[i], new_or_list[j] = new_or_list[j], new_or_list[i] 

    if random.random() < rate:
        num_mutations = max(1, int(size * 0.05)) 
        
        for _ in range(num_mutations):
            mut_idx = random.randrange(size)
            case_idx = new_order[mut_idx]
            case_cluster = surgeries[case_idx]['cluster'] 
            if case_cluster in CLUSTER_TO_ORS and CLUSTER_TO_ORS[case_cluster]:
                new_or_list[mut_idx] = random.choice(CLUSTER_TO_ORS[case_cluster])
    
    return {
        'order': new_order, 
        'assigned_or_list': new_or_list, 
        'fitness': None
    }


def standard_ga_crossover(parent1, parent2):
    size = len(parent1['order'])
    if size < 2:
        return copy.deepcopy(parent1)

    start, end = sorted(random.sample(range(size), 2))

    child_order = [None] * size
    segment = parent1['order'][start:end+1]
    child_order[start:end+1] = segment
    
    set_segment = set(segment)
    remaining = [item for item in parent2['order'] if item not in set_segment]
    
    ptr = (end + 1) % size
    for item in remaining:
        child_order[ptr] = item
        ptr = (ptr + 1) % size

    child_or_list = [None] * size
    p1_or_map = {case_idx: r for case_idx, r in zip(parent1['order'], parent1['assigned_or_list'])}
    p2_or_map = {case_idx: r for case_idx, r in zip(parent2['order'], parent2['assigned_or_list'])}
    
    for i in range(size):
        case_idx = child_order[i]
        if start <= i <= end:
            child_or_list[i] = p1_or_map[case_idx]
        else:
            child_or_list[i] = p2_or_map[case_idx]
    return {'order': child_order, 'assigned_or_list': child_or_list, 'fitness': None}


def standard_ga_mutation(individual, mutation_rate, surgeries, CLUSTER_TO_ORS):
    mutated_order = individual['order'][:]
    mutated_or_list = individual['assigned_or_list'][:]
    size = len(mutated_order)

    if random.random() < mutation_rate:
        start, end = sorted(random.sample(range(size), 2))

        if end - start > 50: 
            end = start + 50   

        mutated_order[start:end] = mutated_order[start:end][::-1]
        mutated_or_list[start:end] = mutated_or_list[start:end][::-1]

    if random.random() < mutation_rate:
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
def run_ga_standard(surgeries, num_gen, pop_size, total_slots, mode, patience=50, st_progress=None, chart_placeholder=None):
    current_cfg = CONFIGS[mode]
    cluster_mapping = current_cfg["CLUSTER_TO_ORS"]
    
    all_or_ids = list(set(str(or_id).strip() for ors in cluster_mapping.values() for or_id in ors))
    buffer_slots = BUFFER_SLOTS 

    no_improvement_count = 0
    best_so_far = float('inf')
    stop_gen = num_gen - 1
    history = []

    # สร้างประชากรเริ่มต้น
    population = generate_initial_population(surgeries, pop_size, cluster_mapping)
    for ind in population:
        sched, status = decode_individual(ind, surgeries, all_or_ids, total_slots, buffer_slots)
        ind['fitness'] = evaluate_fitness(sched, status, total_slots, W_MAKESPAN, W_OVERTIME, W_IMBALANCE)

    # EVOLUTIONARY CYCLE
    for gen in range(num_gen):
        # เรียงลำดับ (Fitness น้อย = เก่ง)
        population.sort(key=lambda x: x['fitness'])
        current_best_fitness = population[0]['fitness']
        history.append(current_best_fitness)

        # ตรรกะ Early Stopping
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
            p1, p2 = random.sample(parents, 2)
            
            if random.random() < CROSSOVER_RATE:
                child = standard_ga_crossover(p1, p2) 
            else:
                child = {'order': p1['order'][:], 'assigned_or_list': p1['assigned_or_list'][:], 'fitness': None}
            
            child = standard_ga_mutation(child, MUTATION_RATE, surgeries, cluster_mapping)
            
            sched, status = decode_individual(child, surgeries, all_or_ids, total_slots, buffer_slots)
            child['fitness'] = evaluate_fitness(sched, status, total_slots, W_MAKESPAN, W_OVERTIME, W_IMBALANCE)
            next_gen.append(child)
        
        population = next_gen

        if st_progress:
            st_progress.progress((gen + 1) / num_gen)
            
        # อัปเดตกราฟทุกๆ 10 Generation
        if chart_placeholder and gen % 10 == 0:
            chart_placeholder.line_chart(history)

    if chart_placeholder:
        chart_placeholder.line_chart(history)

    final_best = min(population, key=lambda x: x['fitness'])
    final_sched, final_status = decode_individual(final_best, surgeries, all_or_ids, total_slots, buffer_slots)
    return final_best, history, final_sched, final_status


def run_ga_hybrid_q(surgeries, num_gen, pop_size, total_slots, mode, patience=50, st_progress=None, chart_placeholder=None):
    current_cfg = CONFIGS[mode]
    cluster_mapping = current_cfg["CLUSTER_TO_ORS"]
    
    all_or_ids = list(set(str(or_id).strip() for ors in cluster_mapping.values() for or_id in ors))
    
    population = generate_initial_population(surgeries, pop_size, cluster_mapping)
    q_table = initialize_q_table()
    epsilon = EPSILON_START
    stop_gen = num_gen - 1
    
    for ind in population:
        sched, status = decode_individual(ind, surgeries, all_or_ids, total_slots, BUFFER_SLOTS)
        ind['fitness'] = evaluate_fitness(sched, status, total_slots, W_MAKESPAN, W_OVERTIME, W_IMBALANCE)

    fitness_var_threshold = np.var([ind['fitness'] for ind in population]) * FITNESS_VAR_THRESHOLD_FACTOR
    
    no_improvement_count = 0
    best_fitness_history = []
    old_best_fitness = min(population, key=lambda ind: ind['fitness'])['fitness']
    best_fitness_history.append(old_best_fitness)

    for gen in range(num_gen):
        current_state = get_state(population, fitness_var_threshold)
        
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(q_table[current_state])
            
        ops = OPERATOR_MAP[action]

        population.sort(key=lambda ind: ind['fitness'])
        next_population = [copy.deepcopy(ind) for ind in population[:NUM_ELITES]]
        
        parents = tournament_selection(population, TOURNAMENT_SIZE, int(pop_size * 0.5))
        
        while len(next_population) < pop_size:
            p1, p2 = random.sample(parents, 2)
            
            if random.random() < CROSSOVER_RATE:
                if ops['crossover'] == 'single':
                    offspring = crossover_single_point(p1, p2)
                else:
                    offspring = crossover_two_point(p1, p2)
            else:
                offspring = {'order': p1['order'][:], 'assigned_or_list': p1['assigned_or_list'][:], 'fitness': None}
            
            offspring = mutate_with_rate(offspring, surgeries, ops['mutation_rate'], cluster_mapping)
            
            child_sched, child_status = decode_individual(offspring, surgeries, all_or_ids, total_slots, BUFFER_SLOTS)
            offspring['fitness'] = evaluate_fitness(child_sched, child_status, total_slots, W_MAKESPAN, W_OVERTIME, W_IMBALANCE)
            next_population.append(offspring)

        new_best_fitness = min(next_population, key=lambda ind: ind['fitness'])['fitness']
        reward = old_best_fitness - new_best_fitness 
        
        if new_best_fitness < old_best_fitness:
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        next_state = get_state(next_population, fitness_var_threshold)
        old_q = q_table[current_state, action]
        max_future_q = np.max(q_table[next_state])
        q_table[current_state, action] = (1 - ALPHA) * old_q + ALPHA * (reward + GAMMA * max_future_q)

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

        if st_progress:
            st_progress.progress((gen + 1) / num_gen)
            
        if chart_placeholder and gen % 10 == 0:
            chart_placeholder.line_chart(best_fitness_history)

    if chart_placeholder:
        chart_placeholder.line_chart(best_fitness_history)

    final_best = min(population, key=lambda ind: ind['fitness'])
    final_sched, final_status = decode_individual(final_best, surgeries, all_or_ids, total_slots, BUFFER_SLOTS)
    
    return final_best, best_fitness_history, final_sched, final_status, stop_gen
