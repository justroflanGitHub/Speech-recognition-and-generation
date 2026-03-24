import math
from typing import List, Tuple

import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# KenLM is optional - only required for LM-based decoding methods
try:
    import kenlm
    KENLM_AVAILABLE = True
except ImportError:
    kenlm = None
    KENLM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Provided utility — do NOT modify
# ---------------------------------------------------------------------------

def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-100h",
            lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=3,
            alpha=1.0,
            beta=1.0,
            temperature=1.0,
        ):
        """
        Args:
            model_name (str): Pretrained Wav2Vec2 model from HuggingFace.
            lm_model_path (str): Path to a KenLM .arpa/.arpa.gz model.
                Pass None to disable LM (Tasks 1–3).
            beam_width (int): Number of hypotheses kept during beam search.
            alpha (float): LM weight used in shallow fusion and rescoring.
                score = log_p_acoustic + alpha * log_p_lm + beta * num_words
            beta (float): Word insertion bonus (see above).
            temperature (float): Scales acoustic logits before softmax.
                T < 1 sharpens the distribution (model more confident).
                T > 1 flattens it (model less confident, giving LM more
                influence). T = 1.0 leaves logits unchanged.
        """
        # Interact with processor/model ONLY here and in decode() to obtain
        # logits — no further model calls are allowed anywhere else.
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        if lm_model_path and not KENLM_AVAILABLE:
            raise ImportError("KenLM is not installed. Please install it from source or using WSL (Ubuntu) with instructions from the README.")
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    # -----------------------------------------------------------------------
    # Provided utility — do NOT modify
    # -----------------------------------------------------------------------

    def _ids_to_text(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs to a decoded string."""
        text = ''.join(self.vocab[i] for i in token_ids)
        return text.replace(self.word_delimiter, ' ').strip().lower()

    # -----------------------------------------------------------------------
    # Tasks 1–4: implement the methods below
    # -----------------------------------------------------------------------

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V).

        Returns:
            str: Decoded transcript.
        """
        # Apply log softmax to get log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Get the most likely token at each timestep
        best_ids = torch.argmax(log_probs, dim=-1).tolist()
        
        # CTC collapsing: remove consecutive duplicates and blanks
        collapsed_ids = []
        prev_id = None
        for token_id in best_ids:
            if token_id != prev_id:  # Remove consecutive duplicates
                if token_id != self.blank_token_id:  # Remove blanks
                    collapsed_ids.append(token_id)
                prev_id = token_id
        
        return self._ids_to_text(collapsed_ids)

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.
            return_beams (bool): Return all beam hypotheses for second-pass
                LM rescoring.

        Returns:
            Union[str, List[Tuple[List[int], float]]]:
                str - best decoded transcript (if return_beams=False).
                List[Tuple[List[int], float]] - list of (token_ids, log_prob)
                    tuples sorted best-first (if return_beams=True).
        """
        # Apply log softmax to get log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        T, V = log_probs.shape
        
        # CTC beam search needs to track two states per hypothesis:
        # - p_blank: probability of ending with blank
        # - p_non_blank: probability of ending with non-blank
        # This is needed to correctly handle repeated tokens separated by blanks
        
        # beams: dict mapping token_sequence -> (p_blank, p_non_blank)
        # Total probability = log(exp(p_blank) + exp(p_non_blank))
        beams = {tuple(): (0.0, float('-inf'))}  # (p_blank, p_non_blank)
        
        for t in range(T):
            new_beams = {}
            frame_log_probs = log_probs[t].cpu().numpy()
            
            for tokens, (p_blank, p_non_blank) in beams.items():
                # Total probability for this hypothesis
                p_total = _log_add(p_blank, p_non_blank)
                
                for v in range(V):
                    token_log_prob = frame_log_probs[v]
                    
                    if v == self.blank_token_id:
                        # Blank: tokens unchanged, probability goes to p_blank
                        new_tokens = tokens
                        new_p_blank = p_total + token_log_prob
                        new_p_non_blank = float('-inf')
                    elif len(tokens) > 0 and tokens[-1] == v:
                        # Same as last token: two cases
                        # 1. Collapse (don't emit): tokens stay same, prob goes to p_non_blank
                        # 2. Emit (only from blank): creates repeated token in output
                        # Handle collapse case first
                        new_tokens = tokens
                        new_p_blank = float('-inf')
                        new_p_non_blank = p_non_blank + token_log_prob
                        
                        # Also handle emit case (from blank state only)
                        if p_blank > float('-inf'):
                            emit_tokens = tokens + (v,)
                            emit_p_non_blank = p_blank + token_log_prob
                            if emit_tokens in new_beams:
                                ex_p_b, ex_p_nb = new_beams[emit_tokens]
                                new_beams[emit_tokens] = (ex_p_b, _log_add(ex_p_nb, emit_p_non_blank))
                            else:
                                new_beams[emit_tokens] = (float('-inf'), emit_p_non_blank)
                    else:
                        # Different token: can extend from any state
                        new_tokens = tokens + (v,)
                        new_p_blank = float('-inf')
                        new_p_non_blank = p_total + token_log_prob
                    
                    # Merge with existing hypothesis if any
                    if new_tokens in new_beams:
                        existing_p_blank, existing_p_non_blank = new_beams[new_tokens]
                        merged_p_blank = _log_add(existing_p_blank, new_p_blank)
                        merged_p_non_blank = _log_add(existing_p_non_blank, new_p_non_blank)
                        new_beams[new_tokens] = (merged_p_blank, merged_p_non_blank)
                    else:
                        if new_p_blank > float('-inf') or new_p_non_blank > float('-inf'):
                            new_beams[new_tokens] = (new_p_blank, new_p_non_blank)
            
            # Keep top beam_width beams by total probability
            def get_total_prob(item):
                tokens, (p_b, p_nb) = item
                return _log_add(p_b, p_nb)
            
            sorted_beams = sorted(new_beams.items(), key=get_total_prob, reverse=True)[:self.beam_width]
            beams = dict(sorted_beams)
        
        # Convert to list of (token_ids, log_prob) sorted by log_prob descending
        result = [(list(tokens), _log_add(p_b, p_nb)) for tokens, (p_b, p_nb) in beams.items()]
        result.sort(key=lambda x: x[1], reverse=True)
        
        if return_beams:
            return result
        
        return self._ids_to_text(result[0][0])

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding with shallow LM fusion.

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.

        Returns:
            str: Decoded transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")
        
        # Apply log softmax to get log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        T, V = log_probs.shape
        
        # For LM fusion, we need to track:
        # - p_blank: acoustic prob ending with blank
        # - p_non_blank: acoustic prob ending with non-blank
        # - lm_state: KenLM state
        # - lm_score: accumulated LM score
        # - num_words: word count for beta bonus
        
        initial_lm_state = kenlm.State()
        self.lm_model.BeginSentenceWrite(initial_lm_state)
        
        # beams: dict mapping tokens_tuple -> (p_blank, p_non_blank, lm_state, lm_score, num_words)
        beams = {tuple(): (0.0, float('-inf'), initial_lm_state, 0.0, 0)}
        
        for t in range(T):
            new_beams = {}
            frame_log_probs = log_probs[t].cpu().numpy()
            
            for tokens, (p_blank, p_non_blank, lm_state, lm_score, num_words) in beams.items():
                p_total = _log_add(p_blank, p_non_blank)
                
                for v in range(V):
                    token_log_prob = frame_log_probs[v]
                    
                    if v == self.blank_token_id:
                        # Blank: tokens unchanged
                        new_tokens = tokens
                        new_p_blank = p_total + token_log_prob
                        new_p_non_blank = float('-inf')
                        new_lm_state = lm_state
                        new_lm_score = lm_score
                        new_num_words = num_words
                    elif len(tokens) > 0 and tokens[-1] == v:
                        # Same as last token
                        # Case 1: collapse (don't emit)
                        new_tokens = tokens
                        new_p_blank = float('-inf')
                        new_p_non_blank = p_non_blank + token_log_prob
                        new_lm_state = lm_state
                        new_lm_score = lm_score
                        new_num_words = num_words
                        
                        # Case 2: emit (from blank only)
                        if p_blank > float('-inf'):
                            char = self.vocab[v]
                            emit_lm_state = kenlm.State()
                            char_score = self.lm_model.BaseScore(lm_state, char, emit_lm_state)
                            emit_tokens = tokens + (v,)
                            emit_p_non_blank = p_blank + token_log_prob
                            emit_lm_score = lm_score + char_score
                            emit_num_words = num_words + (1 if char == self.word_delimiter else 0)
                            
                            # Combined score for comparison
                            emit_combined = emit_p_non_blank + self.alpha * emit_lm_score + self.beta * emit_num_words
                            
                            if emit_tokens in new_beams:
                                ex_p_b, ex_p_nb, ex_lm_state, ex_lm_score, ex_words = new_beams[emit_tokens]
                                ex_combined = _log_add(ex_p_b, ex_p_nb) + self.alpha * ex_lm_score + self.beta * ex_words
                                if emit_combined > ex_combined:
                                    new_beams[emit_tokens] = (float('-inf'), emit_p_non_blank, emit_lm_state, emit_lm_score, emit_num_words)
                                else:
                                    # Merge probabilities
                                    merged_p_nb = _log_add(ex_p_nb, emit_p_non_blank)
                                    new_beams[emit_tokens] = (ex_p_b, merged_p_nb, ex_lm_state, ex_lm_score, ex_words)
                            else:
                                new_beams[emit_tokens] = (float('-inf'), emit_p_non_blank, emit_lm_state, emit_lm_score, emit_num_words)
                    else:
                        # Different token: extend from any state
                        char = self.vocab[v]
                        new_tokens = tokens + (v,)
                        new_p_blank = float('-inf')
                        new_p_non_blank = p_total + token_log_prob
                        new_lm_state = kenlm.State()
                        char_score = self.lm_model.BaseScore(lm_state, char, new_lm_state)
                        new_lm_score = lm_score + char_score
                        new_num_words = num_words + (1 if char == self.word_delimiter else 0)
                    
                    # Combined score for comparison
                    new_combined = _log_add(new_p_blank, new_p_non_blank) + self.alpha * new_lm_score + self.beta * new_num_words
                    
                    if new_tokens in new_beams:
                        ex_p_b, ex_p_nb, ex_lm_state, ex_lm_score, ex_words = new_beams[new_tokens]
                        ex_combined = _log_add(ex_p_b, ex_p_nb) + self.alpha * ex_lm_score + self.beta * ex_words
                        if new_combined > ex_combined:
                            new_beams[new_tokens] = (new_p_blank, new_p_non_blank, new_lm_state, new_lm_score, new_num_words)
                    else:
                        if new_p_blank > float('-inf') or new_p_non_blank > float('-inf'):
                            new_beams[new_tokens] = (new_p_blank, new_p_non_blank, new_lm_state, new_lm_score, new_num_words)
            
            # Keep top beam_width beams by combined score
            def get_combined_score(item):
                tokens, (p_b, p_nb, lm_state, lm_score, words) = item
                return _log_add(p_b, p_nb) + self.alpha * lm_score + self.beta * words
            
            sorted_beams = sorted(new_beams.items(), key=get_combined_score, reverse=True)[:self.beam_width]
            beams = dict(sorted_beams)
        
        # Find best beam by combined score
        best_tokens = None
        best_score = float('-inf')
        for tokens, (p_blank, p_non_blank, lm_state, lm_score, words) in beams.items():
            combined = _log_add(p_blank, p_non_blank) + self.alpha * lm_score + self.beta * words
            if combined > best_score:
                best_score = combined
                best_tokens = tokens
        
        return self._ids_to_text(list(best_tokens))

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs.

        Args:
            beams (List[Tuple[List[int], float]]): List of (token_ids, log_prob)
                tuples from beam_search_decode(logits, return_beams=True).

        Returns:
            str: Best rescored transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")
        
        best_score = float('-inf')
        best_tokens = None
        
        for token_ids, acoustic_score in beams:
            # Compute LM score for this hypothesis
            lm_state = kenlm.State()
            self.lm_model.BeginSentenceWrite(lm_state)
            
            lm_score = 0.0
            num_words = 0
            
            for token_id in token_ids:
                char = self.vocab[token_id]
                new_lm_state = kenlm.State()
                char_score = self.lm_model.BaseScore(lm_state, char, new_lm_state)
                lm_score += char_score
                lm_state = new_lm_state
                
                if char == self.word_delimiter:
                    num_words += 1
            
            # Combined score: acoustic + alpha * lm + beta * num_words
            combined_score = acoustic_score + self.alpha * lm_score + self.beta * num_words
            
            if combined_score > best_score:
                best_score = combined_score
                best_tokens = token_ids
        
        return self._ids_to_text(best_tokens)

    # -----------------------------------------------------------------------
    # Provided — do NOT modify
    # -----------------------------------------------------------------------

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Run the full decoding pipeline on a raw audio tensor.

        Args:
            audio_input (torch.Tensor): 1-D or 2-D audio waveform at 16 kHz.
            method (str): One of "greedy", "beam", "beam_lm", "beam_lm_rescore".

        Returns:
            str: Decoded transcript (lowercase).
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        # Temperature scaling (Task 3): flatten/sharpen the distribution
        # before log_softmax.  T=1.0 is a no-op.  Your decoders must call
        # torch.log_softmax on the logits they receive — do not call it here.
        logits = logits / self.temperature

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose one of: 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'."
            )


# ---------------------------------------------------------------------------
# Quick debug helper — run this file directly to sanity-check your decoder
# on the provided examples/ clips before evaluating on the full test sets.
# ---------------------------------------------------------------------------

def test(decoder: Wav2Vec2Decoder, audio_path: str, reference: str) -> None:
    import jiwer

    audio_input, sr = torchaudio.load(audio_path, backend="soundfile")
    assert sr == 16000, f"Expected 16 kHz, got {sr} Hz for {audio_path}"

    print("=" * 60)
    print(f"REF : {reference}")

    for method in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        try:
            hyp = decoder.decode(audio_input, method=method)
        except NotImplementedError:
            print(f"  [{method}] not yet implemented")
            continue
        except ValueError as e:
            print(f"  [{method}] skipped ({e})")
            continue
        cer = jiwer.cer(reference, hyp)
        wer = jiwer.wer(reference, hyp)
        print(f"  [{method}] {hyp}")
        print(f"           WER={wer:.2%}  CER={cer:.2%}")


if __name__ == "__main__":
    # Reference transcripts are lowercase to match the evaluation manifests.
    # examples/ clips are for quick debugging only — use data/librispeech_test_other/
    # and data/earnings22_test/ for all reported metrics.
    test_samples = [
        ("examples/sample1.wav", "if you are generous here is a fitting opportunity for the exercise of your magnanimity if you are proud here am i your rival ready to acknowledge myself your debtor for an act of the most noble forbearance"),
        ("examples/sample2.wav", "and if any of the other cops had private rackets of their own izzy was undoubtedly the man to find it out and use the information with a beat such as that even going halves and with all the graft to the upper brackets he'd still be able to make his pile in a matter of months"),
        ("examples/sample3.wav", "guess a man gets used to anything hell maybe i can hire some bums to sit around and whoop it up when the ships come in and bill this as a real old martian den of sin"),
        ("examples/sample4.wav", "it was a tune they had all heard hundreds of times so there was no difficulty in turning out a passable imitation of it to the improvised strains of i didn't want to do it the prisoner strode forth to freedom"),
        ("examples/sample5.wav", "marguerite tired out with this long confession threw herself back on the sofa and to stifle a slight cough put up her handkerchief to her lips and from that to her eyes"),
        ("examples/sample6.wav", "at this time all participants are in a listen only mode"),
        ("examples/sample7.wav", "the increase was mainly attributable to the net increase in the average size of our fleets"),
        ("examples/sample8.wav", "operating surplus is a non cap financial measure which is defined as fully in our press release"),
    ]

    decoder = Wav2Vec2Decoder(lm_model_path=None)  # set lm_model_path for Tasks 4+

    for audio_path, reference in test_samples:
        test(decoder, audio_path, reference)
