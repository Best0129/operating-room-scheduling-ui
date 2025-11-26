# src/algorithms/ga_scheduler.py
import random
import copy
import numpy as np
from collections import defaultdict

# Import Shared Logic and Constants
from src.utils import decode_individual, evaluate_fitness
from config.ga_config import *

# --- 1. INITIALIZATION ---
def generate_initial_population(surgeries, pop_size):
    population = []

    # 1. จัดกลุ่ม Case Index ตาม Cluster
    cluster_groups = defaultdict(list)
    for i, s in enumerate(surgeries):
        # i คือ Case Index (0 ถึง N-1)
        cluster_groups[s['cluster']].append(i) 

    for _ in range(pop_size):
        # Array 1: สำหรับลำดับการผ่าตัด (ลำดับ Case Index)
        individual_order = [] 
        # Array 2: สำหรับการจัดสรรห้องผ่าตัด (ห้องที่จัดให้)
        individual_assigned_or = [] 
        
        # Dictionary ชั่วคราว: เก็บ OR ที่สุ่มได้สำหรับแต่ละ Case Index (0-N-1) เพื่อการเข้าถึงง่าย
        temp_assigned_or = {}

        cluster_order = list(cluster_groups.keys())
        random.shuffle(cluster_order) # สุ่มลำดับ cluster 

        for cluster in cluster_order:
            idxs = cluster_groups[cluster][:] # Copy index ของเคสใน cluster
            random.shuffle(idxs) # สุ่มลำดับเคสใน cluster
            possible_ors = CLUSTER_TO_ORS.get(cluster, None) # OR ที่ cluster นี้ใช้ได้
            
            for idx in idxs: 
                if possible_ors:
                    # สุ่ม OR ที่จะใช้ และบันทึกใน Dictionary ชั่วคราว
                    chosen_or = random.choice(possible_ors)
                    temp_assigned_or[idx] = chosen_or 
                
                # บันทึกลำดับเคสใน Array 1
                individual_order.append(idx) 

        for idx in individual_order:
            # ใช้ Case Index (idx) จาก Array 1 เพื่อดึงค่า OR ที่สุ่มไว้จาก Dictionary ชั่วคราว
            individual_assigned_or.append(temp_assigned_or.get(idx, None))

        # บันทึก Individual ในรูปแบบ Numerical Array 2 ชุด
        population.append({
            'order': individual_order,        # Array 1: ลำดับ Case Index (Permutation)
            'assigned_or_list': individual_assigned_or # Array 2: OR Suite ID ตามลำดับเคสใน Array 1
        })
    
    return population

# --- 2. SELECTION ---
def tournament_selection(population, tournament_size, num_parents):
    selected_parents = []
    for _ in range(num_parents):
        tournament = random.sample(population, tournament_size)
        winner = min(tournament, key=lambda ind: ind['fitness'])
        selected_parents.append(winner)
    return selected_parents

# --- 3. CROSSOVER ---
def crossover(parent1, parent2):
    p1_order = parent1['order']
    p2_order = parent2['order']
    size = len(p1_order)
    
    # จุดที่จะเริ่มทำ crossover ไปถึงจุดสิ้นสุด
    start, end = sorted(random.sample(range(size), 2))
    
    # 1. Crossover 'order' (OX1)
    offspring_order = [None] * size
    offspring_order[start:end+1] = p1_order[start:end+1]
    
    # หา ลำดับที่ยังอยู่ใน p2 แต่ยังไม่มีใน p1
    fill_elements = [item for item in p2_order if item not in offspring_order]
    
    fill_idx = end + 1 # เริ่มเติมจากข้างหลังมาก่อน
    for item in fill_elements:
        if fill_idx >= size:
            fill_idx = 0 # ถ้ามันถึงสุดท้ายของ array แล้วให้กลับไปใส่ ที่ index 0
        offspring_order[fill_idx] = item
        fill_idx += 1
        
    # 2. Crossover 'assigned_or_list' (Two-Point Crossover)
    p1_or = parent1['assigned_or_list']
    p2_or = parent2['assigned_or_list']
    
    offspring_or_map = {}
    
    # หาห้อง จากตำแหนงของเตสที่อยู่ใน p1 ที่ crossover (order) ไว้ก่อนหน้าแล้ว
    for i in range(start, end + 1):
        offspring_or_map[offspring_order[i]] = p1_or[p1_order.index(offspring_order[i])]

    # หาห้อง จากตำแหนงของเตสที่อยู่ใน p2 ที่ crossover (order) ไว้ก่อนหน้าแล้ว
    for item in fill_elements:
        if item not in offspring_or_map:
            offspring_or_map[item] = p2_or[p2_order.index(item)]
    
    # รวมเป็น ห้อง ตามลำดับตาม offspring_order จาก p1, p2
    offspring_assigned_or = [offspring_or_map[idx] for idx in offspring_order]

    # สร้าง Individual ใหม่
    offspring = {'order': offspring_order, 'assigned_or_list': offspring_assigned_or, 'fitness': None}
    
    return offspring

# --- 4. MUTATION ---
def mutation(individual, mutation_rate, surgeries):
    # ต้องสร้าง copy ใหม่เพื่อป้องกันการแก้ไข parent โดยตรง
    mutated_individual = copy.deepcopy(individual) 
    order = mutated_individual['order']
    or_list = mutated_individual['assigned_or_list']
    size = len(order)
    
    # 1. Mutation: Order (Swap Mutation)
    if random.random() < mutation_rate:
        # สลับลำดับเคส (Sequencing)
        idx1, idx2 = random.sample(range(size), 2)
        order[idx1], order[idx2] = order[idx2], order[idx1]
        
        # ต้องสลับ OR list ตามลำดับใหม่ด้วย
        or_list[idx1], or_list[idx2] = or_list[idx2], or_list[idx1] 


    # 2. Mutation: Assignment (Random OR Reassignment)
    if random.random() < mutation_rate:
        # สุ่มเลือกตำแหน่งที่จะกลายพันธุ์ OR Assignment
        mut_idx = random.randrange(size)
        case_index = order[mut_idx]
        
        # ค้นหา Cluster ของเคสนี้
        case_cluster = surgeries[case_index]['cluster']

        if case_cluster and case_cluster in CLUSTER_TO_ORS:
            possible_ors = CLUSTER_TO_ORS[case_cluster]

            if len(possible_ors) > 1:
                current_or = or_list[mut_idx]
                # ต้องไม่ใช่ห้องเดิม
                available_ors = [or_id for or_id in possible_ors if or_id != current_or]
                
                if available_ors:
                    or_list[mut_idx] = random.choice(available_ors)
    
    return mutated_individual

# --- 5. MAIN RUN LOOP ---
def run_ga(surgeries, num_gen, pop_size, total_slots, slot_duration):
    
    population = generate_initial_population(surgeries, pop_size)
    
    # 1.1 Calculate initial fitness
    for individual in population:
        OR_schedules, total_used_slots = decode_individual(individual, surgeries)
        individual['fitness'] = evaluate_fitness(total_used_slots, total_slots, W_OVERTIME, W_IMBALANCE)
    
    best_fitness_history = []
    best_in_gen_0 = min(population, key=lambda ind: ind['fitness'])
    best_fitness_history.append(best_in_gen_0['fitness']) 
    
    # 2. EVOLUTIONARY CYCLE
    for generation in range(num_gen):
        population.sort(key=lambda ind: ind['fitness'])
        elites = population[:NUM_ELITES]
        next_population = copy.deepcopy(elites)
        
        num_parents = int(pop_size * 0.5) 
        parents = tournament_selection(population, TOURNAMENT_SIZE, num_parents)
        
        while len(next_population) < pop_size:
            parent1, parent2 = random.choice(parents), random.choice(parents) 
            
            if random.random() < CROSSOVER_RATE:
                offspring = crossover(parent1, parent2) 
            else:
                offspring = copy.deepcopy(parent1)
            
            offspring = mutation(offspring, MUTATION_RATE, surgeries)
            
            OR_schedules, total_used_slots = decode_individual(offspring, surgeries)
            offspring['fitness'] = evaluate_fitness(total_used_slots, total_slots, W_OVERTIME, W_IMBALANCE)
            
            next_population.append(offspring)

        population = next_population
        best_in_gen = min(population, key=lambda ind: ind['fitness'])
        best_fitness_history.append(best_in_gen['fitness'])

    final_best_individual = min(population, key=lambda ind: ind['fitness'])
    OR_schedules, total_used_slots = decode_individual(final_best_individual, surgeries)
    
    return final_best_individual, best_fitness_history, OR_schedules, total_used_slots
