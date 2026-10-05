"""Rule-based, non-medical 'Personalized wellbeing suggestions' (no LLM involved)."""


def suggestions(inputs: dict, category: str, early_warning: bool = False):
    s = []
    if category == "High" or early_warning:
        s.append("Consider speaking with a qualified mental-health professional or a trusted support service.")
    if inputs["sleep_hours"] < 6:
        s.append("Try maintaining a consistent sleep schedule and a calming wind-down routine before bed.")
    if inputs["nightmares"] >= 6:
        s.append("Frequent sleep disturbance may be worth discussing with a professional; keep a regular bedtime and a quiet sleep space.")
    if inputs["stress"] >= 7:
        s.append("Consider relaxation, slow-breathing exercises, or structured daily activities.")
    if inputs["anxiety"] >= 7:
        s.append("Grounding exercises (e.g. naming things you can see/hear) may help in anxious moments.")
    if inputs["social_interaction"] <= 3 or inputs["social_isolation"] >= 7:
        s.append("Consider connecting with a trusted person, community group or support network.")
    if inputs["mood"] <= 3.5:
        s.append("Small, achievable daily goals and time outdoors can support mood; share how you feel with someone you trust.")
    if inputs["activity_level"] <= 3:
        s.append("Gentle physical activity such as a short daily walk may help.")
    if inputs["trauma_symptoms"] >= 7 or inputs["intrusive_thoughts"] >= 7 or inputs["hypervigilance"] >= 7:
        s.append("Trauma-informed counselling or survivor support organisations can offer specialised help.")
    if not s:
        s.append("Your indicators look balanced. Keep up healthy routines and continue periodic check-ins.")
    s.append("Further professional assessment may be appropriate if you are worried about how you feel. "
             "In an emergency, contact local emergency services or a crisis helpline.")
    return s
