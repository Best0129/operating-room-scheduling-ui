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
CONFIGS = {
    "Experiment 1 (Kaggle)": {
        "SERVICE_TO_CLUSTER": {
            'ENT': 'B', 'General': 'A', 'OBGYN': 'B', 'Ophthalmology': 'C',
            'Orthopedics': 'A', 'Pediatrics': 'C', 'Plastic': 'E',
            'Podiatry': 'D', 'Urology': 'B', 'Vascular': 'C'
        },
        "CLUSTER_TO_ORS": {
            'A': [2, 8], 'B': [4, 5], 'C': [3, 7], 'D': [1], 'E': [6]
        }
    },
    "Experiment 2 (Anesthesia)": {
        "SERVICE_TO_CLUSTER": {
            'GA c ETT/TT': 'A', 'Combined [ETT+NB]': 'A', 'Combined [ETT+SB]': 'A',
            'Spinal Block': 'B', 'SB': 'B', 'Nerve Block': 'B', 'Epidural Block': 'B', 
            'RA [Spinal block+Nerve block]': 'B',
            'GA c LMA': 'C', 'GA c mask': 'C', 'Combined [LMA+NB]': 'C', 'SB, LMA': 'C',
            'IV/TIVA': 'D', 'MAC': 'D', 'Fail block': 'D', '-': 'D'
        },
        "CLUSTER_TO_ORS": {
            'A': [601, 602, 603, 604],
            'B': [605, 606, 607, 608],
            'C': [801, 803, 804, 805],
            'D': [806, 807, 808, 701, ' จิตเวช']
        }
        # "CLUSTER_TO_ORS": {
        #     'A': [601, 602, 603, 604, 605, 606, 607, 608, 801, 803, 804, 805, 806, 807, 808, 701, 'จิตเวช'],
        #     'B': [601, 602, 603, 604, 605, 606, 607, 608, 801, 803, 804, 805, 806, 807, 808, 701, 'จิตเวช'],
        #     'C': [601, 602, 603, 604, 605, 606, 607, 608, 801, 803, 804, 805, 806, 807, 808, 701, 'จิตเวช'],
        #     'D': [601, 602, 603, 604, 605, 606, 607, 608, 801, 803, 804, 805, 806, 807, 808, 701, 'จิตเวช'],
        # }
    }
} 
