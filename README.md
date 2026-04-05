# FITGEN.AI — Your AI-Powered Personal Fitness Coach

> **FITGEN.AI** is a conversational AI fitness assistant built with **LangGraph** and **OpenAI**. It designs personalised workout routines, nutrition plans, and recovery strategies by combining expert fitness knowledge with empathetic coaching.

---

## Table of Contents

1. [App Concept](#app-concept)
2. [Project Structure](#project-structure)
3. [Setup & Run](#setup--run)
4. [System Prompt Engineering](#system-prompt-engineering)
   - [Zero-Shot Prompting](#1-zero-shot-prompting)
   - [Few-Shot Prompting](#2-few-shot-prompting)
   - [Chain-of-Thought Prompting](#3-chain-of-thought-prompting)
   - [Analogical Prompting](#4-analogical-prompting)
   - [Generate-Knowledge Prompting](#5-generate-knowledge-prompting)
5. [Advanced Techniques Applied](#advanced-techniques-applied)
6. [Reflection on Prompt Effectiveness](#reflection-on-prompt-effectiveness)

---

## App Concept

**FITGEN.AI** addresses a common problem: most people want to get fit but lack access to personalised, expert coaching. Generic fitness apps offer cookie-cutter plans, and personal trainers are expensive.

FITGEN.AI bridges this gap by acting as an always-available AI fitness coach that:

- **Assesses** the user's fitness level, goals, equipment, and constraints through natural conversation.
- **Designs** customised workout plans, nutrition guidance, and recovery protocols.
- **Educates** users on the *why* behind each recommendation using exercise science.
- **Adapts** in real-time based on user feedback, injuries, or schedule changes.

The system is powered by a LangGraph-based agent with a carefully engineered system prompt that was developed by applying and evaluating multiple prompting techniques.

---

## Project Structure

```
FITGEN.AI/
├── .env.example            # Template for required env vars (API keys)
├── .gitignore
├── requirements.txt        # Python dependencies
├── README.md               # This document
│
├── app.py                  # Entry-point — runs the conversational loop
│
└── agent/
    ├── __init__.py
    ├── graph.py            # LangGraph graph definition & system prompt
    └── state.py            # Shared state schema (TypedDict)
```

---

## Setup & Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/FITGEN.AI.git
cd FITGEN.AI

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key

# 5. Run the app
python app.py
```

---

## System Prompt Engineering

The FITGEN.AI system prompt was developed iteratively by applying five foundational prompting techniques to the same task: *"Design a system prompt for an AI fitness coach."* Each technique shaped a different aspect of the final prompt.

### 1. Zero-Shot Prompting

**Technique:** Provide the model with a task description and no examples. Rely on the model's pre-trained knowledge to generate the response.

**Prompt Used:**
```
You are FITGEN.AI, an expert AI fitness coach and wellness advisor.
Design personalised workout routines, nutrition plans, and recovery
strategies. Prioritise safety and evidence-based recommendations.
```

**Sample Response (model output):**
```
Sure! I'd be happy to help you with your fitness goals. Could you
tell me:
1. What is your current fitness level?
2. What are your primary goals?
3. Do you have any injuries or limitations?
4. What equipment do you have access to?
```

**Observations:**
- The model understood the role and asked reasonable intake questions.
- Responses were *generic* — they lacked structure, depth, and personality.
- No consistent format for delivering plans (sometimes bullets, sometimes paragraphs).
- **Verdict:** Good starting point, but too vague for production.

---

### 2. Few-Shot Prompting

**Technique:** Provide 2–3 example exchanges that demonstrate the ideal interaction style, structure, and depth.

**Prompt Used (appended to Zero-Shot base):**
```
Below are examples of ideal exchanges. Mirror this style:

--- Example 1 ---
User: "I'm a complete beginner. I want to lose weight but I only
have dumbbells at home and 30 minutes a day."

FITGEN.AI: "Great starting point! Here's a 4-week beginner plan:

Weeks 1-2 — Foundation Phase
| Day | Focus        | Key Exercises                        | Sets × Reps |
|-----|-------------|--------------------------------------|-------------|
| Mon | Full Body A | Goblet Squats, DB Rows, Floor Press  | 3 × 12      |
| Wed | Full Body B | DB Deadlifts, Shoulder Press, Lunges | 3 × 12      |
| Fri | Full Body C | DB Swings, Push-ups, Plank (30 s)    | 3 × 10-12   |

Pair this with a modest calorie deficit (~300-500 kcal)
and aim for 0.8 g protein per lb of body weight."

--- Example 2 ---
User: "How much protein do I need to build muscle?"

FITGEN.AI: "Current evidence suggests 0.7-1.0 g per pound of body
weight. If you weigh 170 lbs, aim for 120-170 g/day spread across
3-5 meals."
```

**Sample Response (model output for "I want to gain muscle but I'm vegetarian"):**
```
Absolutely! Here's a muscle-building approach for vegetarians:

**Protein Sources (aim for 0.8-1.0 g/lb):**
| Food           | Protein per serving |
|---------------|-------------------|
| Lentils (1 cup) | 18 g             |
| Greek Yogurt    | 15-20 g          |
| Tofu (½ block)  | 20 g             |
| Quinoa (1 cup)  | 8 g              |

**Sample Day:**
- Breakfast: Greek yogurt parfait (20 g protein)
- Lunch: Lentil & quinoa bowl (26 g protein)
- Dinner: Tofu stir-fry with edamame (30 g protein)
- Snacks: Protein shake + almonds (25 g protein)

Total: ~101 g — supplement with a plant-based protein powder to hit
your target.
```

**Observations:**
- **Dramatic improvement** in structure and consistency — the model mirrored the table format and concise style.
- Responses became more *actionable* with specific numbers and meal plans.
- Few-shot examples act as a powerful "style guide" for the model.
- **Verdict:** Essential technique; the quality jump from Zero-Shot to Few-Shot was the most impactful.

---

### 3. Chain-of-Thought Prompting

**Technique:** Instruct the model to reason through the problem step-by-step before presenting the answer.

**Prompt Used (appended to Few-Shot base):**
```
When generating any plan, follow these reasoning steps:
1. Understand — Restate the user's goal, constraints, and context.
2. Recall Knowledge — Surface relevant exercise science principles.
3. Decompose — Break into sub-problems (muscle groups, macros, etc.).
4. Plan — Draft structured answer (sets, reps, rest, alternatives).
5. Verify — Self-check for safety, balance, and goal alignment.
6. Present — Deliver the final answer clearly.
```

**Sample Response (model output for "Design a 4-day upper/lower split for intermediate lifters"):**
```
Let me work through this step by step:

**Understanding your request:** You're an intermediate lifter looking
for a 4-day upper/lower split — this suggests you have a solid
training base and can handle higher volume.

**Key principles at play:**
- Intermediate lifters benefit from 10-20 sets per muscle group/week
- Upper/Lower splits allow each muscle group to be trained 2×/week
- Progressive overload should target 2.5-5 lb increases bi-weekly

**Here's your programme:**

**Day 1 — Upper Strength**
| Exercise          | Sets × Reps | Rest  |
|------------------|-------------|-------|
| Barbell Bench    | 4 × 5       | 3 min |
| Barbell Row      | 4 × 5       | 3 min |
| OHP              | 3 × 8       | 2 min |
| Weighted Pull-up | 3 × 6-8     | 2 min |
| Face Pulls       | 3 × 15      | 1 min |

[... Day 2-4 follow similar structure ...]

**Safety check:** All movements are bilateral with no excessive spinal
loading. I've included face pulls for shoulder health and balanced
push/pull ratios.
```

**Observations:**
- CoT produced **more thorough and well-reasoned** plans.
- The model naturally surfaced relevant principles (*why* this split works, not just *what* to do).
- Self-verification step caught potential issues (e.g., imbalanced push/pull).
- Slight downside: responses became longer. We addressed this by instructing the model to keep reasoning *brief* unless asked.
- **Verdict:** Critical for complex queries; significantly improved answer quality and trustworthiness.

---

### 4. Analogical Prompting

**Technique:** Instruct the model to use real-world analogies when explaining complex concepts to improve user comprehension.

**Prompt Used (appended to CoT base):**
```
When explaining complex fitness or nutrition concepts, use relatable
real-world analogies. For example:
- Compare progressive overload to gradually adding pages to a book
  you carry every day.
- Compare rest and recovery to charging a phone — you can't run at
  100% if you never plug in.
```

**Sample Response (model output for "Why do I need rest days?"):**
```
Great question — and one that many eager gym-goers struggle with!

**The Science:** When you train, you create microscopic tears in
muscle fibres. Growth happens during *recovery*, not during the
workout itself. This process (muscle protein synthesis) peaks
24-48 hours post-training.

🔋 **Think of it like charging your phone.** Your workout drains the
battery. Rest days are when you plug in and recharge. If you keep
using your phone at 5% battery, it eventually shuts down. Your body
works the same way — without rest, you hit overtraining: fatigue,
injury, and stalled progress.

**Practical guidelines:**
- Beginners: 2-3 rest days per week
- Intermediate: 1-2 rest days (active recovery counts)
- Advanced: At least 1 full rest day + deload weeks every 4-6 weeks

🏗️ *Another way to think about it: you wouldn't expect a construction
crew to build 24/7 without breaks. Quality work requires
downtime.*
```

**Observations:**
- Analogies made technical concepts **immediately relatable**, especially for beginners.
- The phone battery and construction crew analogies were the most effective — they were simple and universally understood.
- Risk: overuse of analogies can feel patronising for advanced users. We added the guideline "use analogies selectively."
- **Verdict:** Excellent for educational content and beginner audiences; adds a warm, approachable tone.

---

### 5. Generate-Knowledge Prompting

**Technique:** Instruct the model to internally generate 2–3 relevant knowledge statements (grounded in science) before answering, and weave them into the response.

**Prompt Used (appended to Analogical base):**
```
Before answering any technical fitness or nutrition question,
internally generate 2-3 relevant knowledge statements grounded in
exercise science. Use these as the factual basis for your response.

Example knowledge statements:
- "Progressive overload is the gradual increase of stress placed
  upon the body during training."
- "A caloric surplus of ~250-500 kcal above maintenance is
  recommended for lean muscle gain."
```

**Sample Response (model output for "How should I train to run a faster 5K?"):**
```
Let me pull together the key principles first:

📚 **Relevant Exercise Science:**
1. *The SAID principle* — your body adapts specifically to the
   demands placed on it. To run faster, you must train at or near
   race pace.
2. *VO₂ max improvement* — interval training at 90-95% max heart
   rate is the most effective way to increase aerobic capacity.
3. *Running economy* — strength training (especially plyometrics)
   improves neuromuscular efficiency and stride power.

**Your 5K Speed Programme (8 Weeks):**

| Day | Session          | Details                            |
|-----|------------------|------------------------------------|
| Mon | Easy Run         | 3-4 mi at conversational pace      |
| Tue | Interval Training| 6 × 800 m at goal pace, 2 min rest |
| Wed | Strength + Core  | Squats, lunges, box jumps, planks  |
| Thu | Tempo Run        | 2 mi at "comfortably hard" pace    |
| Fri | Rest             |                                    |
| Sat | Long Run         | 5-6 mi at easy pace                |
| Sun | Active Recovery  | Light yoga or walk                 |

**Why this works:** The intervals target VO₂ max (principle #2),
the tempo run builds lactate threshold, and the strength day
improves running economy (principle #3). Easy runs build your
aerobic base without excess fatigue.
```

**Observations:**
- GK prompting produced the most **scientifically grounded** responses.
- Responses explained the *why* behind every recommendation, building user trust.
- The generated knowledge acted as an internal fact-check, reducing the chance of hallucination.
- **Verdict:** Essential for any fitness AI; the combination of GK + CoT yielded the most reliable and educational answers.

---

## Advanced Techniques Applied

After developing the initial system prompt using the five foundational techniques above, we applied advanced techniques to improve robustness and production-readiness:

### Decomposition
Complex user requests (e.g., "Design me a 12-week transformation plan") are automatically broken into sub-tasks: assessment → goal-setting → programme design → nutrition → recovery → progress tracking. This prevents the model from generating an overwhelming monolithic response and ensures nothing is missed.

### Self-Consistency
When multiple valid approaches exist (e.g., PPL vs. Upper/Lower vs. Full Body splits), the prompt instructs the model to consider 2–3 options, evaluate trade-offs for the user's specific context, and recommend the best fit. This reduces the variance of responses across sessions.

### Self-Criticism
Before finalising any plan, the model internally asks: *"Is this safe? Is this realistic for this user? Did I miss anything?"* and revises if needed. This catches issues like prescribing exercises that conflict with reported injuries or suggesting unrealistic calorie targets.

### Ensembling & Universal Self-Consistency
During prompt development, we generated responses from multiple prompt variants and selected the best structural elements from each. This "ensembling" of prompt designs resulted in a final prompt that combines the strengths of each technique.

---

## Reflection on Prompt Effectiveness

### Technique Comparison

| Technique | Quality Impact | Ease of Use | Best For |
|-----------|---------------|-------------|----------|
| **Zero-Shot** | ⭐⭐ Low | ⭐⭐⭐⭐⭐ Very Easy | Initial prototyping, simple queries |
| **Few-Shot** | ⭐⭐⭐⭐⭐ Very High | ⭐⭐⭐⭐ Easy | Controlling output format and tone |
| **Chain-of-Thought** | ⭐⭐⭐⭐⭐ Very High | ⭐⭐⭐ Moderate | Complex reasoning, multi-step plans |
| **Analogical** | ⭐⭐⭐⭐ High | ⭐⭐⭐⭐ Easy | Education, beginner-friendly content |
| **Generate-Knowledge** | ⭐⭐⭐⭐⭐ Very High | ⭐⭐⭐ Moderate | Scientific accuracy, trust-building |

### Key Insights

1. **Few-Shot had the single biggest impact.** The jump from Zero-Shot to Few-Shot was dramatic — it transformed vague, inconsistent responses into structured, actionable plans. Providing examples is the most reliable way to control model output.

2. **Chain-of-Thought + Generate-Knowledge is the power combo.** When combined, these two techniques produced responses that were both well-reasoned *and* scientifically grounded. CoT ensured logical structure; GK ensured factual accuracy.

3. **Analogical prompting adds warmth without sacrificing precision.** It made FITGEN.AI feel more like a real coach and less like a textbook. However, it needed to be constrained — too many analogies made responses feel condescending.

4. **Self-Criticism was the most effective advanced technique.** It served as an internal safety net, catching issues that other techniques missed (e.g., exercise contraindications for injuries).

5. **Layering techniques is more effective than using any single technique.** The final system prompt is not simply the sum of individual techniques — the interplay between them creates emergent qualities. CoT gives structure to GK-generated knowledge; Few-Shot examples demonstrate how analogies should be used; Self-Criticism validates the output of all other techniques.

### Challenges

- **Prompt length:** Layering multiple techniques creates long system prompts. We mitigated this by being concise in instructions and relying on examples to carry most of the weight.
- **Balancing verbosity vs. completeness:** CoT and GK tend to produce longer responses. We added explicit instructions to keep reasoning brief unless the user requests details.
- **Example selection for Few-Shot:** The examples chosen heavily influenced output style. We iterated through several examples before finding the right balance of depth, structure, and personality.

### Conclusion

Prompt engineering is not a one-shot activity — it is an iterative design process. Each technique revealed different strengths and weaknesses, and the final FITGEN.AI system prompt is the result of systematically layering, testing, and refining these techniques. The most important lesson: **start simple (Zero-Shot), add structure (Few-Shot), add reasoning (CoT), add grounding (GK), and then validate everything (Self-Criticism).**

---


---

## License

This project is developed as part of an academic assignment. All rights reserved.
