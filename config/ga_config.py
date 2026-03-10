# =====================================================
# 1. TIME and SLOT PARAMETERS
# =====================================================
OPERATING_TIME = (7.0, 15.0)  # เริ่ม 07:00 ถึง 15:00 น.
SLOT_DURATION_MIN = 15        # 1 slot = 15 นาที
BUFFER_SLOTS = 1              # Buffer time = 1 slot

# Total Slots (8 ชม. * 60 นาที / 15 นาที/slot = 32 slots)
# TOTAL_SLOTS = int((OPERATING_TIME[1] - OPERATING_TIME[0]) * 60 / SLOT_DURATION_MIN)
TOTAL_SLOTS_PER_DAY = int((OPERATING_TIME[1] - OPERATING_TIME[0]) * 60 / SLOT_DURATION_MIN)
MAX_SIMULATION_DAYS = 365


# =====================================================
# 2. GA HYPERPARAMETERS (Default Settings)
# =====================================================
POP_SIZE = 100          
NUM_GENERATIONS = 1000
CROSSOVER_RATE = 0.9   
MUTATION_RATE = 0.4 
TOURNAMENT_SIZE = 4    
ELITISM_RATE = 0.05    
NUM_ELITES = max(1, int(POP_SIZE * ELITISM_RATE))

# ค่าน้ำหนักความสำคัญ (Fitness Weights)
W_MAKESPAN = 10.0      # เพื่อจัดให้ใช้เวลาน้อยที่สุด
W_TOTAL_DAYS = 5.0     # เพื่อจัดให้ใช้จำนวนวันน้อยที่สุด
W_OVERTIME = 2.0       # ทำงานเกินเวลาน้อยที่สุด
W_IMBALANCE = 1.0      # กระจายงานอย่างสมดุลมากที่สุด


# =====================================================
# 3. Q-LEARNING HYPERPARAMETERS (Meta-Controller)
# =====================================================
ALPHA = 0.1             # Learning Rate
GAMMA = 0.9             # Discount Factor
EPSILON_START = 0.9     # Initial Exploration Rate
EPSILON_DECAY = 0.995   # Rate to decay epsilon each generation

# Threshold สำหรับวัด Diversity (เทียบกับค่า Variance เริ่มต้น)
# ค่านี้จะถูกคำนวณแบบ Dynamic ใน GA Loop
FITNESS_VAR_THRESHOLD_FACTOR = 0.05


# =====================================================
# 4. MAPPING AND CONSTRAINTS
# =====================================================
# ชุดข้อมูลที่ 1
SERVICE_TO_CLUSTER = {
    'ENT': 'B', 'General': 'A', 'OBGYN': 'B', 'Ophthalmology': 'C',
    'Orthopedics': 'A', 'Pediatrics': 'C', 'Plastic': 'E',
    'Podiatry': 'D', 'Urology': 'B', 'Vascular': 'C'
}

CLUSTER_TO_ORS = {
    'A': [2, 8],  # Orthopedics, General
    'B': [4, 5],  # OBGYN, Urology, ENT
    'C': [3, 7],  # Ophthalmology, Pediatrics, Vascular
    'D': [1],     # Podiatry
    'E': [6]      # Plastic
}

ALL_OR_IDS = [or_id for ors in CLUSTER_TO_ORS.values() for or_id in ors]


# ===================================================== 
# ชุดข้อมูลที่ 2 Experiment 2: Thai Anesthesia Dataset
# SERVICE_TO_CLUSTER = {
#     # Cluster A
#     'GA c ETT/TT': 'A', 
#     'Combined [ETT+NB]': 'A', 
#     'Combined [ETT+SB]': 'A',
    
#     # Cluster B
#     'Spinal Block': 'B', 
#     'SB': 'B', 
#     'Nerve Block': 'B', 
#     'Epidural Block': 'B', 
#     'RA [Spinal block+Nerve block]': 'B',
    
#     # Cluster C
#     'GA c LMA': 'C', 
#     'GA c mask': 'C', 
#     'Combined [LMA+NB]': 'C', 
#     'SB, LMA': 'C',
    
#     # Cluster D
#     'IV/TIVA': 'D', 
#     'MAC': 'D', 
#     'Fail block': 'D', 
#     '-': 'D'
# }

# CLUSTER_TO_ORS = {
#     'A': [601, 602, 603, 604],
    
#     'B': [605, 606, 607, 608],
    
#     'C': [801, 803, 804, 805],
    
#     'D': [806, 807, 808, 701, ' จิตเวช']
# }

# ALL_OR_IDS = [or_id for ors in CLUSTER_TO_ORS.values() for or_id in ors]