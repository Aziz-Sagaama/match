import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_budget_score(budget_a, budget_b):
    a_min, a_max = budget_a
    b_min, b_max = budget_b
    if a_max < b_min or b_max < a_min:
        return 0.0
    overlap_min = max(a_min, b_min)
    overlap_max = min(a_max, b_max)
    overlap = max(0, overlap_max - overlap_min)
    range_total = max(a_max, b_max) - min(a_min, b_min)
    return overlap / range_total if range_total > 0 else 1.0

def compute_location_score(loc_a, loc_b):
    return 1.0 if any(loc.strip() in [x.strip() for x in loc_b] for loc in loc_a) else 0.0

def compute_habits_score(habits_a, habits_b):
    keys = set(habits_a.keys()) & set(habits_b.keys())
    if not keys:
        return 0.0
    matches = sum(habits_a[k] == habits_b[k] for k in keys)
    return matches / len(keys)

def compute_psychological_score(vec_a, vec_b):
    vec_a = np.array(vec_a).reshape(1, -1)
    vec_b = np.array(vec_b).reshape(1, -1)
    return cosine_similarity(vec_a, vec_b)[0][0]

def compute_match(student_a, student_b):
    habit_score = compute_habits_score(student_a["habits"], student_b["habits"])
    budget_score = compute_budget_score(student_a["budget"], student_b["budget"])
    location_score = compute_location_score(student_a["preferred_location"], student_b["preferred_location"])
    housing_score = 1.0 if student_a["housing_type"] == student_b["housing_type"] else 0.0
    psycho_score = compute_psychological_score(student_a["psych_vector"], student_b["psych_vector"])

    final_score = (
        0.2 * habit_score +
        0.2 * budget_score +
        0.2 * location_score +
        0.2 * housing_score +
        0.2 * psycho_score
    )
    return round(final_score * 100, 2)

def get_roommate_profiles():
    conn = psycopg2.connect(
        host='ep-calm-rain-abdfhn9a-pooler.eu-west-2.aws.neon.tech',
        port=5432,
        database='neondb',
        user='neondb_owner',
        password='npg_ibfFy50EmXsl',
        sslmode='require'
    )
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT
            user_id,
            budget_min,
            budget_max,
            preferred_location,
            housing_type,
            sleep_time,
            social_behavior,
            clean_frequency,
            noise_level,
            smoke_status,
            importance_respect,
            importance_communication,
            importance_privacy,
            importance_shared_activities,
            importance_quiet
        FROM roommate_profiles
    """)
    profiles = cur.fetchall()
    cur.close()
    conn.close()

    processed = []
    for p in profiles:
        habits = {
            "sleep_time": p["sleep_time"],
            "social_behavior": p["social_behavior"],
            "clean_frequency": p["clean_frequency"],
            "noise_level": p["noise_level"],
            "smoking": p["smoke_status"] == "yes"
        }
        psych_vector = [
            (p["importance_respect"] or 0) / 5,
            (p["importance_communication"] or 0) / 5,
            (p["importance_privacy"] or 0) / 5,
            (p["importance_shared_activities"] or 0) / 5,
            (p["importance_quiet"] or 0) / 5
        ]
        profile = {
            "user_id": p["user_id"],
            "budget": (p["budget_min"], p["budget_max"]),
            "preferred_location": p["preferred_location"].split(",") if p["preferred_location"] else [],
            "housing_type": p["housing_type"],
            "habits": habits,
            "psych_vector": psych_vector
        }
        processed.append(profile)
    return processed

def compute_all_matches(profiles):
    matches = []
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            score = compute_match(profiles[i], profiles[j])
            matches.append({
                "user1": profiles[i]["user_id"],
                "user2": profiles[j]["user_id"],
                "score": score
            })
    return matches

if __name__ == "__main__":
    profiles = get_roommate_profiles()
    matches = compute_all_matches(profiles)
    for match in matches:
        print(f"Matching {match['user1']} & {match['user2']}: {match['score']}% compatible")
