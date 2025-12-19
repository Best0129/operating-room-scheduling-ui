import random
import copy
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from src.utils import decode_individual, evaluate_fitness
from config.ga_config import * 

# ----------------------------------------------------
def initialize_q_table():
    """Q-table: 2 States (High/Low Diversity) x 4 Actions"""
    return np.zeros((2, 4))

def get_state(population, fitness_var_threshold):
    """
    กำหนด State จากความหลากหลาย (Diversity) ของ Population
    State 0: High Diversity / State 1: Low Diversity
    """
    fitness_var = np.var([ind['fitness'] for ind in population])
    return 0 if fitness_var > fitness_var_threshold else 1
    # ถ้าค่าที่คำนวณได้มากกว่า เกณฑ์ = High Diversity (State 0)
    # ถ้าน้อยกว่าหรือเท่ากับ = Low Diversity (State 1)

# Q-Action Mapping
OPERATOR_MAP = {
    0: {'crossover': 'single', 'mutation_rate': 0.01}, # Single-point + low mutation
    1: {'crossover': 'two', 'mutation_rate': 0.01},    # Two-point + low mutation
    2: {'crossover': 'single', 'mutation_rate': 0.10}, # Single-point + high mutation
    3: {'crossover': 'two', 'mutation_rate': 0.10},    # Two-point + high mutation
}


# ----------------------------------------------------
# CORE GA (ใช้ร่วมกัน)
# ----------------------------------------------------

def generate_initial_population(surgeries, pop_size):
    population = []
    cluster_groups = defaultdict(list)
    for i, s in enumerate(surgeries):
        cluster_groups[s['cluster']].append(i) 

    for _ in range(pop_size):
        individual_order = [] 
        individual_assigned_or = [] 
        temp_assigned_or = {}

        cluster_order = list(cluster_groups.keys())
        random.shuffle(cluster_order)
        
        for cluster in cluster_order:
            idxs = cluster_groups[cluster][:]
            random.shuffle(idxs) 
            possible_ors = CLUSTER_TO_ORS.get(cluster, None)
            
            for idx in idxs: 
                if possible_ors:
                    chosen_or = random.choice(possible_ors)
                    temp_assigned_or[idx] = chosen_or 
                individual_order.append(idx) 

        for idx in individual_order:
            individual_assigned_or.append(temp_assigned_or.get(idx, None))

        population.append({
            'order': individual_order, 
            'assigned_or_list': individual_assigned_or, 
            'fitness': None
        })
    return population

def tournament_selection(population, tournament_size, num_parents):
    selected_parents = []
    for _ in range(num_parents):
        tournament = random.sample(population, tournament_size)
        winner = min(tournament, key=lambda ind: ind['fitness'])
        selected_parents.append(winner)
    return selected_parents


# ----------------------------------------------------
# Crossover/Mutation OPERATORS (ตาม Action ที่จะให้ Q-Agent เลือก)
# ----------------------------------------------------

def crossover_single_point(parent1, parent2):
    """จุดตัดเดียว (OX1)"""
    p1_order = parent1['order']
    p2_order = parent2['order']
    size = len(p1_order)
    
    start, end = sorted(random.sample(range(size), 2)) # สุ่ม 2 จุด เพื่อเลือกช่วงกลาง
    
    # 1. Crossover 'order' (OX1)
    offspring_order = [None] * size
    offspring_order[start:end+1] = p1_order[start:end+1]
    
    fill_elements = [item for item in p2_order if item not in offspring_order]
    
    fill_idx = end + 1
    for item in fill_elements:
        if fill_idx >= size:
            fill_idx = 0 
        offspring_order[fill_idx] = item
        fill_idx += 1
        
    # 2. Crossover 'assigned_or_list' (ใช้ Map-based approach)
    p1_or = parent1['assigned_or_list']
    p2_or = parent2['assigned_or_list']
    offspring_or_map = {}
    
    for i in range(start, end + 1):
        offspring_or_map[offspring_order[i]] = p1_or[p1_order.index(offspring_order[i])]

    for item in fill_elements:
        if item not in offspring_or_map:
            offspring_or_map[item] = p2_or[p2_order.index(item)]
    
    offspring_assigned_or = [offspring_or_map[idx] for idx in offspring_order]

    return {'order': offspring_order, 'assigned_or_list': offspring_assigned_or, 'fitness': None}


def crossover_two_point(parent1, parent2):
    """จุดตัดสองจุด (เลือก 2 ช่วง)"""
    p1_order = parent1['order']
    p2_order = parent2['order']
    size = len(p1_order)
    
    # สุ่ม 2 จุดตัด (p1, p2) และเลือกช่วง [0:p1] และ [p2:] จาก P1
    p1 = random.randint(1, size // 2)
    p2 = random.randint(size // 2, size - 1)
    
    offspring_order = [None] * size
    
    # เลือกช่วง [0:p1] และ [p2:] จาก P1
    offspring_order[:p1] = p1_order[:p1]
    offspring_order[p2:] = p1_order[p2:]
    
    # เติมส่วนที่เหลือจาก P2
    fill_elements = [item for item in p2_order if item not in offspring_order]
    
    fill_idx = p1 
    for item in fill_elements:
        offspring_order[fill_idx] = item
        fill_idx += 1

    # 2. Crossover 'assigned_or_list' (ใช้ Map-based approach)
    offspring_or_map = {}
    
    # ORs จาก P1
    for i in list(range(p1)) + list(range(p2, size)):
        offspring_or_map[offspring_order[i]] = parent1['assigned_or_list'][p1_order.index(offspring_order[i])]

    # ORs จาก P2 (เติมให้เต็ม)
    for item in fill_elements:
        if item not in offspring_or_map:
            offspring_or_map[item] = parent2['assigned_or_list'][p2_order.index(item)]

    offspring_assigned_or = [offspring_or_map[idx] for idx in offspring_order]

    return {'order': offspring_order, 'assigned_or_list': offspring_assigned_or, 'fitness': None}


def mutate_with_rate(individual, surgeries, rate):
    """Mutation Logic ที่รับ Rate เข้ามาใช้ในการควบคุม"""
    mutated_individual = copy.deepcopy(individual) 
    order = mutated_individual['order']
    or_list = mutated_individual['assigned_or_list']
    size = len(order)
    
    # 1. Mutation: Order (Swap Mutation)
    if random.random() < rate:
        idx1, idx2 = random.sample(range(size), 2)
        order[idx1], order[idx2] = order[idx2], order[idx1]
        or_list[idx1], or_list[idx2] = or_list[idx2], or_list[idx1] 


    # 2. Mutation: Assignment (Random OR Reassignment)
    if random.random() < rate:
        mut_idx = random.randrange(size)
        case_index = order[mut_idx]
        case_cluster = surgeries[case_index]['cluster']

        if case_cluster and case_cluster in CLUSTER_TO_ORS:
            possible_ors = CLUSTER_TO_ORS[case_cluster]

            if len(possible_ors) > 1:
                current_or = or_list[mut_idx]
                available_ors = [or_id for or_id in possible_ors if or_id != current_or]
                
                if available_ors:
                    or_list[mut_idx] = random.choice(available_ors)
    
    return mutated_individual


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
