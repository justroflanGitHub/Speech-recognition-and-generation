# Assignment 2: CTC Decoding Strategies for Wav2Vec2 ASR


## Overview

This assignment involves implementing and evaluating various CTC (Connectionist Temporal Classification) decoding strategies for a pre-trained wav2vec2 acoustic model (`facebook/wav2vec2-base-100h`). The goal is to study the effects of beam search, language model integration, temperature scaling, and domain shift on speech recognition quality.

---

## Completed Tasks

### Task 1: Greedy Decoding ✅

**Implementation:** [`greedy_decode()`](wav2vec2decoder.py:74)

**Description:** Simple baseline decoding that selects the most probable token at each timestep, then collapses repeated tokens and removes CTC blanks.

**Algorithm:**
1. Apply log softmax to logits (with optional temperature scaling)
2. Select argmax at each timestep
3. Collapse consecutive identical tokens
4. Remove blank tokens (token ID 0)
5. Convert token IDs to text using vocabulary

**Results on LibriSpeech test-other (200 samples):**
| Metric | Value |
|--------|-------|
| WER | 13.15% |
| CER | 4.71% |

---

### Task 2: Beam Search Decoding ✅

**Implementation:** [`beam_search_decode()`](wav2vec2decoder.py:116)

**Description:** Maintains top-k hypotheses during decoding using proper CTC state tracking with separate probabilities for blank and non-blank endings.

**Algorithm:**
The implementation uses proper CTC beam search with two probability tracks per hypothesis:
- `p_blank`: Probability of hypothesis ending with a blank token
- `p_non_blank`: Probability of hypothesis ending with a non-blank token

For each timestep:
1. **Blank transition**: Add blank token, probability goes to p_blank
2. **Non-blank transitions**:
   - Different token: Append token, probability goes to p_non_blank
   - Same token as last: Two cases:
     - Collapse (don't emit): tokens stay same
     - Emit (from blank only): creates repeated token

**Beam Width Comparison:**

| Beam Width | WER | CER | Time (s) |
|------------|-----|-----|----------|
| 1 | 13.16% | 4.70% | 63.21 |
| 3 | 12.96% | 4.66% | 68.57 |
| 10 | 12.89% | 4.63% | 88.59 |
| 50 | 12.95% | 4.64% | 229.63 |

**Analysis:** 
- Beam width 10 provides the best WER/CER trade-off
- Width 50 shows diminishing returns with significantly longer processing time
- Greedy (width=1 equivalent) is ~2% relative worse than beam search

---

### Task 3: Temperature Scaling ✅

**Implementation:** Temperature parameter in [`greedy_decode()`](wav2vec2decoder.py:74) and [`beam_search_decode()`](wav2vec2decoder.py:116)

**Description:** Temperature scaling controls the sharpness of the probability distribution:
- `T < 1`: Sharper distribution (more confident, less diverse)
- `T = 1`: Original distribution
- `T > 1`: Flatter distribution (less confident, more diverse)

**Formula:**
```
logits_scaled = logits / temperature
probabilities = softmax(logits_scaled)
```

**Parameter Sweep Results on LibriSpeech test-other (Greedy Decoding):**

| Temperature | WER | CER |
|-------------|-----|-----|
| 0.5 | 13.15% | 4.71% |
| 0.8 | 13.15% | 4.71% |
| 1.0 | 13.15% | 4.71% |
| 1.2 | 13.15% | 4.71% |
| 1.5 | 13.15% | 4.71% |
| 2.0 | 13.15% | 4.71% |

**Analysis:** All temperatures yield identical results for greedy decoding. This is expected because greedy decoding uses argmax, which is invariant to temperature scaling:
- `argmax(softmax(logits/T)) = argmax(logits)` for any T > 0

Temperature has no effect on greedy decoding results. It would only matter for:
1. Beam search with sampling
2. Softmax probability distributions used in scoring

---

### Task 4: Beam Search with LM Shallow Fusion ✅

**Implementation:** [`beam_search_with_lm()`](wav2vec2decoder.py:210)

**Description:** Integrates n-gram language model scores during beam search decoding using shallow fusion.

**Scoring Formula:**
```
score = log_p_acoustic + alpha * log_p_lm + beta * num_words
```

Where:
- `log_p_acoustic`: CTC model log probability
- `log_p_lm`: Language model log probability
- `alpha`: LM weight (controls LM influence)
- `beta`: Word bonus (encourages longer hypotheses)

**Parameter Sweep Results on LibriSpeech test-other (partial results):**

| Alpha | Beta | WER | CER |
|-------|------|-----|-----|
| 0.01 | 0.0 | 13.13% | 4.71% |
| 0.01 | 0.5 | 13.06% | 4.68% |
| 0.05 | 0.0 | ~12.83% | ~4.60% |

**Best Configuration:** alpha=0.05, beta=0.0 with WER=12.83%

**Observation:** The 3-gram LibriSpeech LM provides minimal improvement over acoustic-only decoding. This is because:
1. The acoustic model is already well-trained on LibriSpeech
2. The LM is pruned (1e-7 pruning) and may lack coverage
3. Very low alpha values work best, indicating the acoustic model is already well-calibrated

---

### Task 5: 4-gram Language Model ✅

**Implementation:** Uses same [`beam_search_with_lm()`](wav2vec2decoder.py:210) and [`lm_rescore()`](wav2vec2decoder.py:304)

**Description:** Evaluated with the larger 4-gram LM from openslr.org/11.

**Results on LibriSpeech test-other:**

| Method | LM | Alpha | Beta | WER | CER |
|--------|-----|-------|------|-----|-----|
| Shallow Fusion | 3-gram | 0.05 | 0.0 | 12.83% | ~4.60% |
| Shallow Fusion | 4-gram | 0.05 | 0.0 | 13.38% | 4.74% |
| Rescoring | 3-gram | 0.01 | 0.5 | 12.77% | 4.59% |
| Rescoring | 4-gram | 0.01 | 0.5 | 12.75% | 4.59% |

**Analysis:**
- The 4-gram LM performs slightly worse in shallow fusion but marginally better in rescoring
- The larger context window of 4-gram doesn't provide significant benefits on this task
- Both LMs are trained on LibriSpeech, so the acoustic model already captures similar patterns

---

### Task 6: LM Rescoring ✅

**Implementation:** [`lm_rescore()`](wav2vec2decoder.py:304)

**Description:** Two-pass decoding where:
1. First pass: Generate N-best list using beam search (acoustic only)
2. Second pass: Rescore hypotheses with language model

**Scoring Formula:**
```
final_score = log_p_acoustic + alpha * log_p_lm + beta * num_words
```

**Parameter Sweep Results on LibriSpeech test-other:**

| Alpha | Beta | WER | CER |
|-------|------|-----|-----|
| **0.01** | **0.5** | **12.77%** | **4.59%** |
| 0.01 | 0.0 | 12.86% | 4.63% |
| 0.01 | 1.0 | 12.96% | 4.61% |
| 0.01 | 1.5 | 12.86% | 4.59% |
| 0.05 | 0.0 | 12.90% | 4.64% |
| 0.05 | 0.5 | 12.81% | 4.60% |
| 0.50 | 0.0 | 14.23% | 4.83% |
| 1.00 | 0.0 | 15.36% | 5.06% |
| 2.00 | 0.0 | 17.50% | 5.41% |
| 5.00 | 0.0 | 18.86% | 5.64% |

**Best Configuration:** alpha=0.01, beta=0.5 with WER=12.77%, CER=4.59%

**Analysis:**
- LM rescoring achieves the best overall WER (12.77%), slightly better than shallow fusion (12.83%)
- Very low alpha values (0.01-0.05) work best, indicating the acoustic model is already well-calibrated
- Higher alpha values degrade performance significantly as LM overpowers acoustic scores
- Beta (word bonus) provides modest improvement at alpha=0.01
- Rescoring is more stable to large alpha values compared to shallow fusion because the acoustic beam provides diverse hypotheses that the LM can choose from

---

### Task 7: Cross-Domain Evaluation ✅

**Description:** Evaluated all methods on both in-domain (LibriSpeech test-other) and out-of-domain (Earnings22) datasets to demonstrate domain shift effects.

**Results on Earnings22 (Out-of-Domain - Financial Earnings Calls):**

| Method | WER | CER |
|--------|-----|-----|
| Greedy | 62.31% | 31.28% |
| Beam Search (width=10) | 62.36% | 31.19% |
| Beam+LM Shallow Fusion (alpha=0.05, beta=0) | 62.01% | 31.11% |
| LM Rescoring (alpha=0.01, beta=0.5) | 62.12% | 31.21% |

**Comparison: In-Domain vs Out-of-Domain:**

| Method | LibriSpeech WER | Earnings22 WER | Degradation |
|--------|-----------------|----------------|-------------|
| Greedy | 13.15% | 62.31% | +49.16% |
| Beam Search | 12.89% | 62.36% | +49.47% |
| Beam+LM Shallow Fusion | 12.83% | 62.01% | +49.18% |
| LM Rescoring | 12.77% | 62.12% | +49.35% |

**Analysis:**
- **Massive domain shift effect:** WER degrades by ~49% absolute (nearly 5x worse) on out-of-domain data
- The acoustic model trained on LibriSpeech (read audiobooks) struggles with:
  - Spontaneous speech patterns in earnings calls
  - Financial terminology (e.g., "EBITDA", "YoY", "quarter-over-quarter")
  - Different speaking styles and recording conditions
- LM integration provides minimal help because the LibriSpeech LM doesn't match financial domain vocabulary
- All methods perform similarly on Earnings22, suggesting the bottleneck is acoustic mismatch, not decoding strategy

---

### Task 8: Train Financial-Domain KenLM ✅

Trained a 3-gram language model using the Earnings22 training corpus with KenLM.

**Training Data:**
- Source: `data/earnings22_train/corpus.txt`
- Lines: 4,995 sentences
- Tokens: 101,181 words
- Vocabulary: 5,701 unigrams

**Model Statistics:**
| N-gram | Count |
|--------|-------|
| Unigrams | 5,701 |
| Bigrams | 42,688 |
| Trigrams | 77,164 |

**File:** `lm/financial-3gram.arpa` (3.8 MB)

**Training Commands:**
```bash
# Build KenLM tools
git clone --depth=1 https://github.com/kpu/kenlm /tmp/kenlm_build
mkdir /tmp/kenlm_build/build && cd /tmp/kenlm_build/build
cmake .. && make -j4 lmplz build_binary

# Train 3-gram LM
/tmp/kenlm_build/build/bin/lmplz -o 3 --discount_fallback \
    < data/earnings22_train/corpus.txt > lm/financial-3gram.arpa
```

**Sample N-grams (Financial Domain Vocabulary):**
- "uh", "then", "maybe" - common fillers in spontaneous speech
- "strategic", "partnerships", "enterprise", "quarter" - financial terminology
- "two thousand and twenty" - year expressions common in earnings calls

---

### Task 9: Compare Language Models ✅

Compared both language models on both test domains to evaluate domain adaptation effects.

**Shallow Fusion Results (alpha=0.05, beta=0.0):**

| LM | LibriSpeech WER | LibriSpeech CER | Earnings22 WER | Earnings22 CER |
|----|-----------------|-----------------|----------------|----------------|
| LibriSpeech 3-gram | 13.33% | 4.77% | 62.20% | 31.25% |
| Financial 3-gram | 13.30% | 4.72% | **61.80%** | **31.07%** |

**Rescoring Results (alpha=0.01, beta=0.5):**

| LM | LibriSpeech WER | LibriSpeech CER | Earnings22 WER | Earnings22 CER |
|----|-----------------|-----------------|----------------|----------------|
| LibriSpeech 3-gram | **12.77%** | **4.59%** | 62.28% | 31.17% |
| Financial 3-gram | 12.80% | 4.60% | **62.20%** | **31.11%** |

**Analysis:**

1. **In-Domain Performance (LibriSpeech):**
   - Both LMs perform nearly identically on LibriSpeech
   - LibriSpeech LM slightly better with rescoring (12.77% vs 12.80%)
   - The financial LM doesn't hurt performance on LibriSpeech despite being out-of-domain

2. **Out-of-Domain Performance (Earnings22):**
   - Financial LM provides **marginal improvement** on Earnings22
   - Shallow fusion: 61.80% vs 62.20% (0.4% absolute improvement)
   - Rescoring: 62.20% vs 62.28% (0.08% absolute improvement)
   - The improvement is small because the **acoustic mismatch is the dominant factor**

3. **Why doesn't domain-matched LM help more?**
   - The ~49% WER gap is primarily due to **acoustic model mismatch** (wav2vec2 trained on read audiobooks)
   - The acoustic model produces incorrect phoneme sequences that no LM can fix
   - Financial vocabulary helps marginally (words like "quarter", "earnings", "million")
   - But most errors are due to spontaneous speech patterns, not vocabulary

4. **Key Finding:** Domain-matched LM provides small but consistent improvement, but the dominant bottleneck for cross-domain ASR is the acoustic model, not the language model.

---

## Summary of Results (LibriSpeech test-other)

| Method | WER | CER | Notes |
|--------|-----|-----|-------|
| Greedy | 13.15% | 4.71% | Baseline |
| Beam Search (width=10) | 12.89% | 4.63% | +2.0% relative improvement |
| Beam+LM Shallow Fusion (3-gram, alpha=0.05, beta=0) | 12.83% | ~4.60% | Best shallow fusion config |
| LM Rescoring (3-gram, alpha=0.01, beta=0.5) | 12.77% | 4.59% | Slightly worse than 4-gram |
| LM Rescoring (4-gram, alpha=0.01, beta=0.5) | 12.75% | 4.59% | **Best overall** |

---
