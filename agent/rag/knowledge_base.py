"""
agent/rag/knowledge_base.py
────────────────────────────
Curated fitness and nutrition knowledge for the RAG pipeline.

Each entry is a "document" with:
  - id:       unique identifier
  - topic:    broad category (workout | diet | general)
  - title:    short title
  - content:  factual information (evidence-based)
  - source:   citation or reference

These documents are embedded and stored in FAISS for retrieval.
"""

from __future__ import annotations

KNOWLEDGE_DOCS: list[dict] = [
    # ══════════════════════════════════════════════════════════════
    #  WORKOUT KNOWLEDGE
    # ══════════════════════════════════════════════════════════════
    {
        "id": "w001",
        "topic": "workout",
        "title": "Hypertrophy Rep Ranges",
        "content": (
            "For muscle hypertrophy, research indicates that training with "
            "6-12 reps per set at 65-85% of 1RM is most effective. However, "
            "recent meta-analyses (Schoenfeld et al., 2017) suggest that a "
            "wide range of rep ranges (6-30 reps) can produce similar "
            "hypertrophy when sets are taken close to failure. Volume "
            "(total sets per muscle group per week) is the primary driver "
            "of hypertrophy, with 10-20 sets per muscle group per week "
            "recommended for intermediate to advanced lifters."
        ),
        "source": "Schoenfeld BJ, et al. (2017). Strength and Hypertrophy Adaptations. J Strength Cond Res.",
    },
    {
        "id": "w002",
        "topic": "workout",
        "title": "Progressive Overload Principle",
        "content": (
            "Progressive overload is the gradual increase of stress placed "
            "on the body during training. This can be achieved by increasing "
            "weight, reps, sets, training frequency, or reducing rest periods. "
            "A systematic review by Plotkin et al. (2022) confirmed that "
            "progressive overload is essential for continued strength and "
            "muscle gains. Recommended progression: increase load by 2.5-5% "
            "when target reps are achieved for all prescribed sets."
        ),
        "source": "Plotkin DL, et al. (2022). Progressive Overload. Sports Med.",
    },
    {
        "id": "w003",
        "topic": "workout",
        "title": "Rest Periods Between Sets",
        "content": (
            "Rest periods significantly affect training outcomes. For "
            "strength: 3-5 minutes between sets of compound lifts. For "
            "hypertrophy: 1-3 minutes is sufficient (Grgic et al., 2017). "
            "Shorter rest (<60s) can increase metabolic stress but may "
            "reduce total volume. For endurance: 30-60 seconds. Research "
            "shows longer rest periods (>2 min) allow for greater total "
            "volume, which may be more important for hypertrophy."
        ),
        "source": "Grgic J, et al. (2017). Effects of Rest Interval Duration. Sports Med.",
    },
    {
        "id": "w004",
        "topic": "workout",
        "title": "Compound vs Isolation Exercises",
        "content": (
            "Compound exercises (squat, deadlift, bench press, rows, "
            "overhead press) recruit multiple muscle groups and allow heavier "
            "loads, making them more time-efficient and better for overall "
            "strength development. Isolation exercises (curls, lateral raises, "
            "leg extensions) target specific muscles and are useful for "
            "addressing weaknesses or adding volume. A well-rounded program "
            "should prioritize compound movements and supplement with "
            "isolation work."
        ),
        "source": "NSCA Essentials of Strength Training and Conditioning, 4th Ed.",
    },
    {
        "id": "w005",
        "topic": "workout",
        "title": "Training Frequency",
        "content": (
            "Training each muscle group 2x per week produces superior "
            "hypertrophy compared to 1x per week at matched volume "
            "(Schoenfeld et al., 2016). Common splits: Full Body 3x/week, "
            "Upper/Lower 4x/week, Push/Pull/Legs 6x/week. For beginners, "
            "3 full-body sessions per week are recommended. For intermediates, "
            "4-5 sessions using upper/lower or PPL splits are effective."
        ),
        "source": "Schoenfeld BJ, et al. (2016). Training Frequency. Sports Med.",
    },
    {
        "id": "w006",
        "topic": "workout",
        "title": "Warm-Up Protocol",
        "content": (
            "An effective warm-up includes: (1) 5-10 minutes light cardio "
            "to raise core temperature, (2) dynamic stretching targeting "
            "working muscles, (3) activation drills for stabilizers, "
            "(4) progressive warm-up sets starting at 50% working weight. "
            "Static stretching before strength training can reduce force "
            "production and should be avoided pre-workout (Behm & Chaouachi, "
            "2011). Save static stretching for post-workout."
        ),
        "source": "Behm DG, Chaouachi A. (2011). Effects of Stretching. Eur J Appl Physiol.",
    },
    {
        "id": "w007",
        "topic": "workout",
        "title": "Deload Weeks",
        "content": (
            "A deload is a planned reduction in training volume or intensity "
            "(typically 40-60% reduction) for 1 week every 4-8 weeks. "
            "Deloads allow for recovery, reduce injury risk, and can enhance "
            "subsequent performance through supercompensation. Signs you need "
            "a deload: persistent fatigue, stalled progress, joint pain, "
            "decreased motivation. During a deload, maintain frequency but "
            "reduce volume by 50% and intensity by 10-20%."
        ),
        "source": "Haff GG, Triplett NT. Essentials of Strength Training and Conditioning. NSCA.",
    },
    {
        "id": "w008",
        "topic": "workout",
        "title": "HIIT for Fat Loss",
        "content": (
            "High-Intensity Interval Training (HIIT) involves alternating "
            "between high-intensity bursts (85-95% max HR) and recovery "
            "periods. HIIT produces similar fat loss to moderate-intensity "
            "continuous training (MICT) in ~40% less time (Wewege et al., "
            "2017). Recommended protocols: 20-30 seconds work / 60-90 "
            "seconds rest, 8-12 rounds, 2-3x per week. HIIT should not "
            "replace all steady-state cardio — a mix is optimal."
        ),
        "source": "Wewege M, et al. (2017). HIIT vs MICT for Body Composition. Obes Rev.",
    },
    {
        "id": "w009",
        "topic": "workout",
        "title": "Mind-Muscle Connection",
        "content": (
            "Focusing attention on the target muscle during exercise "
            "(internal focus) can increase muscle activation by up to 20% "
            "for isolation exercises at moderate loads (Calatayud et al., "
            "2016). This is most effective at loads below 60% 1RM. At higher "
            "loads (>80% 1RM), an external focus (moving the weight) is "
            "more effective for performance. For hypertrophy-focused training, "
            "use controlled tempos and concentrate on the working muscle."
        ),
        "source": "Calatayud J, et al. (2016). Mind-Muscle Connection. Eur J Sport Sci.",
    },
    {
        "id": "w010",
        "topic": "workout",
        "title": "Training to Failure",
        "content": (
            "Training to muscular failure means performing reps until you "
            "cannot complete another with proper form. Research suggests "
            "that training 1-3 RIR (reps in reserve) is equally effective "
            "for hypertrophy as training to failure, with less fatigue "
            "accumulation (Vieira et al., 2021). Reserve failure training "
            "for the last set of an exercise or for isolation movements. "
            "Avoid failure on heavy compound lifts for safety."
        ),
        "source": "Vieira AF, et al. (2021). Training to Failure. Scand J Med Sci Sports.",
    },

    # ══════════════════════════════════════════════════════════════
    #  DIET / NUTRITION KNOWLEDGE
    # ══════════════════════════════════════════════════════════════
    {
        "id": "d001",
        "topic": "diet",
        "title": "Protein Requirements for Muscle Growth",
        "content": (
            "The International Society of Sports Nutrition (ISSN) recommends "
            "1.6-2.2 g protein per kg body weight per day for individuals "
            "engaged in resistance training (Jäger et al., 2017). Higher "
            "intakes (up to 3.1 g/kg) may help preserve lean mass during "
            "caloric deficits. Protein should be distributed across 3-5 "
            "meals with 20-40g per meal to maximize muscle protein synthesis. "
            "Complete protein sources include: meat, fish, eggs, dairy, soy."
        ),
        "source": "Jäger R, et al. (2017). ISSN Position Stand: Protein. J Int Soc Sports Nutr.",
    },
    {
        "id": "d002",
        "topic": "diet",
        "title": "Caloric Deficit for Fat Loss",
        "content": (
            "A caloric deficit of 500 kcal/day below TDEE produces ~0.5 kg "
            "(1 lb) fat loss per week. Aggressive deficits (>750 kcal) risk "
            "muscle loss, metabolic adaptation, and hormonal disruption. "
            "The Mifflin-St Jeor equation estimates BMR: "
            "Men: 10×weight(kg) + 6.25×height(cm) - 5×age - 5. "
            "Women: 10×weight(kg) + 6.25×height(cm) - 5×age - 161. "
            "Multiply BMR by activity factor (1.2-1.9) for TDEE."
        ),
        "source": "Mifflin MD, et al. (1990). Am J Clin Nutr. ISSN Position Stand on Diets.",
    },
    {
        "id": "d003",
        "topic": "diet",
        "title": "Macronutrient Distribution",
        "content": (
            "Evidence-based macronutrient ranges for active individuals: "
            "Protein: 25-35% of calories (1.6-2.2 g/kg). "
            "Fat: 20-35% of calories (minimum 0.5 g/kg for hormonal health). "
            "Carbohydrates: remainder (3-7 g/kg depending on activity level). "
            "For strength athletes: prioritize protein and carbs. "
            "For endurance athletes: higher carb intake (5-10 g/kg). "
            "Flexible dieting / IIFYM (If It Fits Your Macros) is effective "
            "when overall calories and protein are controlled."
        ),
        "source": "Helms ER, et al. (2014). Evidence-Based Recommendations for Contest Prep. J Int Soc Sports Nutr.",
    },
    {
        "id": "d004",
        "topic": "diet",
        "title": "Meal Timing and Frequency",
        "content": (
            "Total daily protein and calorie intake matters more than meal "
            "timing for most goals. However, distributing protein across "
            "3-5 meals (every 3-5 hours) may optimize muscle protein "
            "synthesis (Areta et al., 2013). Pre-workout: consume a meal "
            "with protein + carbs 1-3 hours before training. Post-workout: "
            "consume 20-40g protein within 2 hours. The 'anabolic window' "
            "is less critical than once believed — total daily intake is key."
        ),
        "source": "Areta JL, et al. (2013). Protein Distribution. J Physiol.",
    },
    {
        "id": "d005",
        "topic": "diet",
        "title": "Creatine Supplementation",
        "content": (
            "Creatine monohydrate is the most researched and effective "
            "ergogenic supplement. Benefits: increased strength, power, and "
            "lean mass. Dosage: 3-5 g/day (loading phase optional: 20 g/day "
            "for 5-7 days). No clinically significant side effects in "
            "healthy individuals. ISSN position: creatine is safe and "
            "effective for enhancing high-intensity exercise performance "
            "and lean body mass gains (Kreider et al., 2017)."
        ),
        "source": "Kreider RB, et al. (2017). ISSN Exercise & Sports Nutrition Review. J Int Soc Sports Nutr.",
    },
    {
        "id": "d006",
        "topic": "diet",
        "title": "Hydration Guidelines",
        "content": (
            "General recommendation: 30-40 ml water per kg body weight daily. "
            "During exercise: 400-800 ml per hour depending on sweat rate. "
            "Signs of dehydration: dark urine, thirst, decreased performance. "
            "Electrolyte replacement needed for exercise >60 minutes or in "
            "hot conditions. A 2% body weight loss from dehydration can "
            "reduce performance by 10-20%. Weigh before and after exercise — "
            "replace 150% of weight lost within 4-6 hours."
        ),
        "source": "ACSM Position Stand on Fluid Replacement. Med Sci Sports Exerc.",
    },
    {
        "id": "d007",
        "topic": "diet",
        "title": "Intermittent Fasting",
        "content": (
            "Common IF protocols: 16:8 (16h fast, 8h eating), 5:2 (5 normal "
            "days, 2 restricted days), OMAD (one meal a day). Research shows "
            "IF produces similar weight loss to continuous caloric restriction "
            "when calories are matched (Varady et al., 2022). IF may improve "
            "insulin sensitivity and autophagy. Not recommended for: "
            "pregnant women, those with eating disorder history, or athletes "
            "requiring multiple daily training sessions."
        ),
        "source": "Varady KA, et al. (2022). Intermittent Fasting Outcomes. Annu Rev Nutr.",
    },
    {
        "id": "d008",
        "topic": "diet",
        "title": "Fiber Intake",
        "content": (
            "Recommended fiber intake: 25-38 g/day (14g per 1000 kcal). "
            "Benefits: improved digestion, satiety, blood sugar regulation, "
            "and reduced cardiovascular disease risk. Good sources: "
            "vegetables (broccoli, spinach), fruits (berries, apples), "
            "legumes (lentils, beans), whole grains (oats, quinoa), "
            "nuts and seeds. Increase fiber gradually to avoid GI discomfort "
            "and ensure adequate water intake."
        ),
        "source": "Dietary Guidelines for Americans, 2020-2025. USDA.",
    },
    {
        "id": "d009",
        "topic": "diet",
        "title": "Vegetarian/Vegan Protein Sources",
        "content": (
            "Complete plant proteins: soy, quinoa, hemp seed, buckwheat. "
            "High-protein plant foods: lentils (18g/cup), chickpeas (15g/cup), "
            "tofu (20g/cup), tempeh (31g/cup), seitan (25g/100g), edamame "
            "(17g/cup). Combine grains + legumes for complete amino acid "
            "profiles. Vegan athletes may need 10-20% higher total protein "
            "intake due to lower digestibility of plant proteins (van Vliet "
            "et al., 2015). Consider B12, iron, zinc, and omega-3 supplementation."
        ),
        "source": "van Vliet S, et al. (2015). Plant-Based Protein Quality. J Nutr.",
    },
    {
        "id": "d010",
        "topic": "diet",
        "title": "Reverse Dieting",
        "content": (
            "Reverse dieting involves gradually increasing calories (50-150 "
            "kcal per week) after a prolonged caloric deficit to minimize "
            "fat regain and restore metabolic rate. This approach helps "
            "rebuild metabolic capacity, restore hormonal balance (leptin, "
            "thyroid, testosterone), and reduce the 'rebound' effect. "
            "Monitor weight and waist measurements weekly. Continue until "
            "maintenance calories are reached or slight surplus for muscle "
            "building."
        ),
        "source": "Trexler ET, et al. (2014). Metabolic Adaptation. J Int Soc Sports Nutr.",
    },

    # ══════════════════════════════════════════════════════════════
    #  GENERAL / SAFETY KNOWLEDGE
    # ══════════════════════════════════════════════════════════════
    {
        "id": "g001",
        "topic": "general",
        "title": "Sleep and Recovery",
        "content": (
            "Sleep is critical for recovery and performance. Recommended: "
            "7-9 hours per night for adults. Sleep deprivation (<6 hours) "
            "reduces strength by 5-10%, impairs glucose metabolism, "
            "increases cortisol, decreases testosterone, and increases "
            "injury risk. Strategies: consistent sleep schedule, cool/dark "
            "room, limit caffeine after 2pm, avoid screens 1 hour before bed. "
            "Naps of 20-30 minutes can partially offset sleep debt."
        ),
        "source": "Watson NF, et al. (2015). Sleep Duration Recommendations. Sleep.",
    },
    {
        "id": "g002",
        "topic": "general",
        "title": "Anabolic Steroids — Safety Warning",
        "content": (
            "Anabolic-androgenic steroids (AAS) are controlled substances "
            "that carry significant health risks including: liver damage, "
            "cardiovascular disease, hormonal disruption, infertility, "
            "psychiatric effects (aggression, depression), and dependence. "
            "FITGEN.AI does NOT provide guidance on illegal PED use. "
            "Consult a medical professional for any hormonal concerns. "
            "Natural alternatives: optimize training, nutrition, sleep, "
            "and stress management."
        ),
        "source": "NIDA. Anabolic Steroids DrugFacts. National Institute on Drug Abuse.",
    },
    {
        "id": "g003",
        "topic": "general",
        "title": "Minimum Calorie Safety Thresholds",
        "content": (
            "Very low calorie diets (<1200 kcal for women, <1500 kcal for "
            "men) should only be undertaken under medical supervision. "
            "Risks: nutrient deficiencies, muscle loss, metabolic slowdown, "
            "gallstones, electrolyte imbalances. A safe rate of weight loss "
            "is 0.5-1 kg per week (0.5-1% of body weight). Crash diets "
            "are counterproductive for long-term health and body composition."
        ),
        "source": "AHA/ACC/TOS Guidelines on Weight Management. Circulation.",
    },
    {
        "id": "g004",
        "topic": "general",
        "title": "Exercise During Pregnancy",
        "content": (
            "ACOG recommends 150 minutes of moderate-intensity exercise per "
            "week during uncomplicated pregnancies. Benefits: reduced "
            "gestational diabetes risk, improved mood, easier labor. "
            "Avoid: contact sports, exercises with fall risk, supine "
            "exercises after first trimester, hot yoga, scuba diving. "
            "Stop exercise if: vaginal bleeding, dizziness, chest pain, "
            "fluid leaking. Always consult OB-GYN before starting or "
            "continuing exercise during pregnancy."
        ),
        "source": "ACOG Committee Opinion 804. Physical Activity in Pregnancy. 2020.",
    },
    {
        "id": "g005",
        "topic": "general",
        "title": "Body Recomposition",
        "content": (
            "Body recomposition (gaining muscle while losing fat) is "
            "possible for: beginners, detrained individuals, overweight "
            "individuals, and those on PEDs. For natural intermediates, it "
            "is slower and requires: maintenance calories or slight deficit "
            "(~200 kcal), high protein (2-2.4 g/kg), progressive resistance "
            "training, adequate sleep. Track progress via body measurements "
            "and progress photos rather than scale weight alone."
        ),
        "source": "Barakat C, et al. (2020). Body Recomposition. Strength Cond J.",
    },
]


def get_all_documents() -> list[dict]:
    """Return all knowledge documents."""
    return KNOWLEDGE_DOCS


def get_documents_by_topic(topic: str) -> list[dict]:
    """Return documents filtered by topic."""
    return [d for d in KNOWLEDGE_DOCS if d["topic"] == topic]
