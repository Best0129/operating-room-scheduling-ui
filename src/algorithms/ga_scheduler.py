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
        order = list(range(len(surgeries)))
        random.shuffle(order)
        assigned_or_list = [random.choice(CLUSTER_TO_ORS[surgeries[i]['cluster']]) for i in order]
        population.append({'order': order, 'assigned_or_list': assigned_or_list, 'fitness': None})
    return population


def tournament_selection(population, tournament_size, num_parents):
    selected = []
    for _ in range(num_parents):
        candidates = random.sample(population, tournament_size)
        selected.append(min(candidates, key=lambda ind: ind['fitness']))
    return selected


# ----------------------------------------------------
# Crossover/Mutation OPERATORS (ตาม Action ที่จะให้ Q-Agent เลือก)
# ----------------------------------------------------
def crossover_single_point(parent1, parent2):
    # จุดตัดเดียว
    p1_order, p2_order = parent1['order'], parent2['order']
    p1_or, p2_or = parent1['assigned_or_list'], parent2['assigned_or_list']
    size = len(p1_order)
    
    start, end = sorted(random.sample(range(size), 2))
    
    # Crossover order
    offspring_order = [None] * size
    segment = p1_order[start:end+1]
    offspring_order[start:end+1] = segment
    
    segment_set = set(segment)
    fill_elements = [item for item in p2_order if item not in segment_set]
    
    ptr = (end + 1) % size
    for item in fill_elements:
        offspring_order[ptr] = item
        ptr = (ptr + 1) % size
        
    # Crossover assigned_or_list
    # สร้าง Map จาก ID เคสไปที่ห้องผ่าตัดที่ถูกเลือกไว้
    p1_map = {idx: r for idx, r in zip(p1_order, p1_or)}
    p2_map = {idx: r for idx, r in zip(p2_order, p2_or)}

    offspring_assigned_or = [
        p1_map[offspring_order[i]] if start <= i <= end else p2_map[offspring_order[i]]
        for i in range(size)
    ]
    return {'order': offspring_order, 'assigned_or_list': offspring_assigned_or, 'fitness': None}


def crossover_two_point(parent1, parent2):
    # จุดตัดสองจุด (เลือกช่วงหัวกับช่วงท้าย)
    p1_order, p2_order = parent1['order'], parent2['order']
    size = len(p1_order)
    
    # เลือก 2 ช่วงจาก P1 (หัวกับท้าย)
    cp1 = random.randint(1, size // 2)
    cp2 = random.randint(size // 2, size - 1)
    
    offspring_order = [None] * size
    # เก็บส่วนหัวและท้ายจาก P1
    offspring_order[:cp1] = p1_order[:cp1]
    offspring_order[cp2:] = p1_order[cp2:]
    
    # กรองตัวที่เหลือจาก P2
    p1_segment_set = set(offspring_order) 
    fill_elements = [item for item in p2_order if item not in p1_segment_set]
    
    ptr = 0
    for item in fill_elements:
        while offspring_order[ptr] is not None:
            ptr += 1
        offspring_order[ptr] = item

    # Mapping ห้องผ่าตัด
    p1_map = {idx: r for idx, r in zip(p1_order, parent1['assigned_or_list'])}
    p2_map = {idx: r for idx, r in zip(p2_order, parent2['assigned_or_list'])}
    
    offspring_assigned_or = [
        p1_map[offspring_order[i]] if (i < cp1 or i >= cp2) else p2_map[offspring_order[i]]
        for i in range(size)
    ]
    return {'order': offspring_order, 'assigned_or_list': offspring_assigned_or, 'fitness': None}


def mutate_with_rate(individual, surgeries, rate):
    new_order = individual['order'][:]
    new_or_list = individual['assigned_or_list'][:]
    size = len(new_order)
    
    # Swap Mutation
    if random.random() < rate:
        i, j = random.sample(range(size), 2)
        new_order[i], new_order[j] = new_order[j], new_order[i]
        new_or_list[i], new_or_list[j] = new_or_list[j], new_or_list[i] 

    # Assignment Mutation (ลองเปลี่ยนห้อง)
    if random.random() < rate:
        # สุ่มแก้ 5 จุดพร้อมกัน
        for _ in range(5):
            mut_idx = random.randrange(size)
            case_idx = new_order[mut_idx]
            case_cluster = surgeries[case_idx]['cluster']
            if case_cluster in CLUSTER_TO_ORS:
                new_or_list[mut_idx] = random.choice(CLUSTER_TO_ORS[case_cluster])
    
    return {'order': new_order, 'assigned_or_list': new_or_list, 'fitness': None}


# ----------------------------------------------------
# RUN (Standard GA และ Hybrid GA-Q)
# ----------------------------------------------------

def run_ga_standard(surgeries, num_gen, pop_size, total_slots, operating_time, slot_duration):
    """
    Standard GA: ใช้อัตรา Crossover/Mutation แบบ FIXED
    """
    population = generate_initial_population(surgeries, pop_size)
    
    # Initial Evaluation
    for individual in population:
        OR_schedules, total_used_slots = decode_individual(individual, surgeries)
        individual['fitness'] = evaluate_fitness(OR_schedules, total_used_slots, total_slots, W_OVERTIME, W_IMBALANCE) 

    best_fitness_history = []
    best_fitness_history.append(min(population, key=lambda ind: ind['fitness'])['fitness']) 
    
    # EVOLUTIONARY CYCLE
    for generation in tqdm(range(num_gen), desc="Standard GA"):
        population.sort(key=lambda ind: ind['fitness'])
        elites = population[:NUM_ELITES] 
        next_population = copy.deepcopy(elites)
        
        num_parents = int(pop_size * 0.5) 
        parents = tournament_selection(population, TOURNAMENT_SIZE, num_parents)
        
        while len(next_population) < pop_size:
            parent1, parent2 = random.choice(parents), random.choice(parents) 

            # Crossover (Fixed Rate)
            if random.random() < CROSSOVER_RATE:
                # ใช้ Two-point Crossover เป็น Default ใน Standard GA
                offspring = crossover_two_point(parent1, parent2) 
            else:
                offspring = copy.deepcopy(parent1)
            
            # Mutation (Fixed Rate)
            offspring = mutate_with_rate(offspring, surgeries, MUTATION_RATE) # ใช้ MUTATION_RATE จาก config
            
            # Evaluation
            OR_schedules, total_used_slots = decode_individual(offspring, surgeries)
            offspring['fitness'] = evaluate_fitness(OR_schedules, total_used_slots, total_slots, W_OVERTIME, W_IMBALANCE)
            
            next_population.append(offspring)

        # Replacement & Logging
        population = next_population
        best_fitness_history.append(min(population, key=lambda ind: ind['fitness'])['fitness'])

    final_best_individual = min(population, key=lambda ind: ind['fitness'])
    OR_schedules, total_used_slots = decode_individual(final_best_individual, surgeries)
    
    return final_best_individual, best_fitness_history, OR_schedules, total_used_slots


def run_ga_hybrid_q(surgeries, num_gen, pop_size, total_slots, operating_time, slot_duration):
    """
    Hybrid GA-Q: ใช้ Q-Learning ในการเลือก Operator ในแต่ละ Generation
    """
    
    population = generate_initial_population(surgeries, pop_size)
    q_table = initialize_q_table()
    epsilon = EPSILON_START
    
    # Initial Evaluation
    for individual in population:
        OR_schedules, total_used_slots = decode_individual(individual, surgeries)
        individual['fitness'] = evaluate_fitness(OR_schedules, total_used_slots, total_slots, W_OVERTIME, W_IMBALANCE)

    # คำนวณ Threshold สำหรับ State
    initial_var = np.var([ind['fitness'] for ind in population]) # ค่า Variance เริ่มต้น ความหลากหลาย 
    fitness_var_threshold = initial_var * FITNESS_VAR_THRESHOLD_FACTOR
    
    best_fitness_history = []
    old_best_fitness = min(population, key=lambda ind: ind['fitness'])['fitness'] 
    best_fitness_history.append(old_best_fitness) 
    
    for generation in tqdm(range(num_gen), desc="Hybrid GA-Q"):
        
        # Q-LEARNING OBSERVATION AND DECISION
        current_state = get_state(population, fitness_var_threshold)
        
        # ε-greedy Strategy
        if random.random() < epsilon:
            action = random.randint(0, 3) # Exploration
        else:
            action = np.argmax(q_table[current_state]) # Exploitation
            
        # SET OPERATORS
        operators = OPERATOR_MAP[action]
        crossover_type = operators['crossover']
        mutation_rate = operators['mutation_rate'] 
        
        # ELITISM
        population.sort(key=lambda ind: ind['fitness'])
        elites = population[:NUM_ELITES] 
        next_population = copy.deepcopy(elites)
        
        # REPRODUCTION
        num_parents = int(pop_size * 0.5) 
        parents = tournament_selection(population, TOURNAMENT_SIZE, num_parents)
        
        while len(next_population) < pop_size:
            parent1, parent2 = random.choice(parents), random.choice(parents) 

            # Crossover เลือกโดย Q-Agent
            if random.random() < CROSSOVER_RATE:
                if crossover_type == 'single':
                    offspring = crossover_single_point(parent1, parent2)
                else:
                    offspring = crossover_two_point(parent1, parent2)
            else:
                offspring = copy.deepcopy(parent1)
            
            # Mutation โดยใช้ Rate ที่เลือกโดย Q-Agent
            offspring = mutate_with_rate(offspring, surgeries, mutation_rate) 
            
            # Evaluation
            OR_schedules, total_used_slots = decode_individual(offspring, surgeries)
            offspring['fitness'] = evaluate_fitness(OR_schedules, total_used_slots, total_slots, W_OVERTIME, W_IMBALANCE)
            
            next_population.append(offspring)

        population = next_population
        
        # คำนวณ Fitness ใหม่ และ Reward สำหรับ Q-Learning
        new_best_fitness = min(population, key=lambda ind: ind['fitness'])['fitness']
        
        # คำนวณ Reward: (ค่าเก่า - ค่าใหม่)
        reward = old_best_fitness - new_best_fitness # ถ้าค่าใหม่ดีกว่า (น้อยกว่า) จะได้รางวัลเป็นบวก
        
        next_state = get_state(population, fitness_var_threshold)
        
        # Q-learning Update Equation
        old_q_value = q_table[current_state, action]
        max_future_q = np.max(q_table[next_state])
        
        new_q_value = (1 - ALPHA) * old_q_value + ALPHA * (reward + GAMMA * max_future_q)
        q_table[current_state, action] = new_q_value
        
        # อัปเดตค่าสำหรับรอบถัดไป
        old_best_fitness = new_best_fitness
        epsilon = max(0.01, epsilon * EPSILON_DECAY) # ลดค่า Exploration

        # Logging
        best_fitness_history.append(new_best_fitness)

    final_best_individual = min(population, key=lambda ind: ind['fitness'])
    OR_schedules, total_used_slots = decode_individual(final_best_individual, surgeries)
    
    return final_best_individual, best_fitness_history, OR_schedules, total_used_slots
