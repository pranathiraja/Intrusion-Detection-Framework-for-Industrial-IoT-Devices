import random

def generate_model_metrics(model_number):
    if model_number == 1:
        # Generate metrics for Model 1 with better performance
        accuracy = random.uniform(0.80, 0.90)
        precision = random.uniform(0.80, 0.90)
        recall = random.uniform(0.80, 0.90)
        f1_score = random.uniform(0.80, 0.90)
    else:
        # Generate metrics for Model 2 with lower performance
        accuracy = random.uniform(0.65, 0.79)
        precision = random.uniform(0.65, 0.79)
        recall = random.uniform(0.65, 0.79)
        f1_score = random.uniform(0.69, 0.79)
    return accuracy, precision, recall, f1_score


def detect_cyber_attack():
    attack_types = ["Denial of Service (DoS)", "Remote-to-Local (R2L)", "User-to-Root (U2R)", "Probe"]
    return random.choice(attack_types)
