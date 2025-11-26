# =====================================================
# 1. TIME and SLOT PARAMETERS
# =====================================================
OPERATING_TIME = (7.0, 15.0)  # เริ่ม 07:00 ถึง 15:00 น.
SLOT_DURATION_MIN = 15        # 1 slot = 15 นาที
BUFFER_SLOTS = 1              # Buffer time (Turnover time) = 1 slot

# Calculated Total Slots (8 ชม. * 60 นาที / 15 นาที/slot = 32 slots)
TOTAL_SLOTS = int((OPERATING_TIME[1] - OPERATING_TIME[0]) * 60 / SLOT_DURATION_MIN)


# =====================================================
# 2. GA HYPERPARAMETERS (Default Settings)
# =====================================================
POP_SIZE = 50           
NUM_GENERATIONS = 200   
CROSSOVER_RATE = 0.9    
MUTATION_RATE = 0.2     

TOURNAMENT_SIZE = 2     
ELITISM_RATE = 0.02     # 4% Elitism (2 out of 50)
NUM_ELITES = max(1, int(POP_SIZE * ELITISM_RATE))

# Fitness Weights
W_OVERTIME = 5.0        # Weight สำหรับ Overtime Penalty
W_IMBALANCE = 0.5       # Weight สำหรับ Imbalance Penalty


# =====================================================
# 3. MAPPING AND CONSTRAINTS
# =====================================================
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
