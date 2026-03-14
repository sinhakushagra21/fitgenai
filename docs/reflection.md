# Assignment 4: Reflection

## FITGEN.AI — Fine-Tuning the Model

---

## 1. Process Reflection

### What Worked Well

**Structured Dataset Curation**: Creating a balanced dataset of 100 examples across three categories (typical, edge, adversarial) proved essential. The adversarial examples — jailbreak attempts, off-topic queries, harmful requests — were particularly valuable for testing safety guardrails. Having explicit routing labels (`expected_tool`) and keyword expectations (`expected_response_contains`) enabled automated evaluation at scale.

**Multi-Technique Comparison Framework**: Running all 5 prompting techniques in parallel provided direct comparisons under identical conditions. This eliminated confounding variables and made it clear which techniques excel at which tasks. For example, Chain-of-Thought consistently produced the most structured responses for complex planning queries, while Zero-Shot was sufficient for straightforward routing.

**RAG Pipeline with Evidence Grounding**: Building a curated knowledge base of 25 peer-reviewed documents and indexing them with FAISS embeddings dramatically improved factual accuracy. Responses with RAG cited specific studies (e.g., "Schoenfeld et al., 2017") and used precise numbers (e.g., "1.6-2.2 g/kg protein per day") rather than vague generalizations.

**Meta-Prompting Self-Critique Loop**: The 3-round iteration (generate → critique → refine) consistently produced longer, more thorough responses with better safety disclaimers. The model was effectively able to identify its own gaps — particularly around safety warnings and missing edge case handling.

### Challenges Encountered

**Fine-Tuning Data Volume**: With only 70 training examples, the fine-tuning dataset is relatively small. OpenAI recommends 50-100 examples as a minimum, so we're at the lower end. The model may not generalize well to significantly different query phrasings. More data — especially for edge cases and adversarial inputs — would improve robustness.

**Perplexity via Logprobs Limitations**: OpenAI's logprobs API provides token-level probabilities, but interpreting perplexity for instruction-following models requires caution. A low perplexity could mean the model is confidently generating common patterns rather than high-quality content. We address this by combining perplexity with keyword-match and safety metrics for a more holistic view.

**Temperature vs. Quality Trade-off**: Higher temperatures (0.7-1.0) improve response diversity and creativity but reduce routing consistency. For the base agent (routing), T=0 is clearly optimal. For specialist tools (content generation), T=0.5-0.7 provides the best balance.

**RAG Latency Overhead**: Adding the embedding + FAISS retrieval step adds ~0.5-1 second of latency per query. While this is acceptable for the quality improvement, it could matter for real-time applications. The FAISS index caching helps on subsequent queries.

---

## 2. Technical Insights

### Prompt Sensitivity Findings

1. **Routing is robust to paraphrasing** — All 5 techniques correctly route tool calls even when the same query is phrased 3 different ways. This validates the prompt engineering from Assignment 3.

2. **Temperature affects content, not routing** — Tool selection (routing) is consistent across temperatures. Content quality and length vary more at T≥0.7.

3. **CoT produces the most structured output** — The explicit reasoning steps in Chain-of-Thought prompts consistently produce well-organized responses with clear sections.

4. **Few-Shot is most format-consistent** — When given examples of desired output format, the model adheres to the structure more reliably across different queries.

### RAG vs. Baseline

| Dimension          | Baseline | RAG      |
|--------------------|----------|----------|
| Factual Keywords   | ~60%     | ~85%     |
| Citations          | 0 avg    | 2-4 avg  |
| Specific Numbers   | Sometimes| Consistently|
| Source Attribution  | Never    | Always   |

The biggest RAG advantage is **citation grounding** — baseline responses make vague claims ("experts recommend...") while RAG responses cite specific studies and position stands.

### Meta-Prompting Quality

The self-critique loop improved responses in predictable ways:
- **Safety disclaimers** were the most consistently added improvement
- **Specific numbers** (sets, reps, calories, grams) replaced vague ranges
- **Structure** improved — refined responses had clearer headings and bullet points
- **Length increased** by ~30-50% (not always desirable, but generally matched increased depth)

### Perplexity Patterns

- **Lower perplexity** on typical fitness queries → model is well-calibrated for its domain
- **Higher perplexity** on adversarial queries → healthy uncertainty on out-of-scope inputs
- **CoT** shows lowest average perplexity → the reasoning chain reduces uncertainty
- **Generate-Knowledge** has highest perplexity → generating knowledge before answering introduces more variance

---

## 3. Connections to Course Concepts

### Prompting Techniques (Assignment 3 → 4)

Assignment 3 demonstrated that different prompting techniques produce qualitatively different outputs. Assignment 4 quantified this:
- Sensitivity testing showed **which differences matter** (content) vs. which don't (routing)
- Perplexity evaluation revealed **model confidence** varies by technique
- Meta-prompting showed techniques can be **composed** (generate → critique → refine)

### Fine-Tuning Complements Prompting

Fine-tuning and prompt engineering are not mutually exclusive:
- **Prompt engineering** handles the creativity and structure of responses
- **Fine-tuning** can specialize the model's routing behavior
- Together, they create a system where routing is reliable AND responses are high-quality

### RAG Adds a Third Dimension

RAG (Retrieval-Augmented Generation) adds factual grounding:
- **Prompts** define behavior and tone
- **Fine-tuning** improves task-specific performance
- **RAG** ensures factual accuracy with sources

This three-layer approach (prompt → fine-tune → augment) mirrors real-world production AI systems.

---

## 4. Future Improvements

1. **Larger Dataset**: Expand from 100 to 500+ examples, including multi-turn conversations
2. **Human Evaluation**: Add subjective quality ratings alongside automated metrics
3. **Adaptive RAG**: Only retrieve when confidence is low (measured by perplexity threshold)
4. **Multi-Turn Meta-Prompting**: Extend the self-critique to incorporate user feedback loops
5. **A/B Testing Framework**: Deploy multiple technique variants and measure real user satisfaction
6. **Cost Tracking**: Add per-query cost estimation to compare fine-tuning ROI vs. longer prompts

---

## 5. Conclusion

Assignment 4 transformed FITGEN.AI from a prompt engineering demo into a quantitatively evaluated system. Key takeaways:

1. **Measure everything** — Routing accuracy, cosine similarity, perplexity, keyword coverage, and safety compliance provide a multi-dimensional view of system quality.

2. **Prompts are powerful but fragile** — Sensitivity testing revealed that while routing is robust, response quality varies significantly with temperature and technique choice.

3. **RAG is the biggest quality lever** — For factual domains like fitness/nutrition, retrieved evidence dramatically improves accuracy and trustworthiness.

4. **Meta-prompting works** — Self-critique consistently improves response quality, especially for safety and completeness. The cost is 3x the latency and tokens.

5. **Fine-tuning specializes** — Even a small dataset (70 examples) can teach the model domain-specific routing patterns, reducing reliance on complex prompts.

The combination of prompt engineering, fine-tuning, and RAG creates a robust, evidence-based fitness coaching system that handles edge cases gracefully and refuses inappropriate requests safely.
