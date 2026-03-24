#!/usr/bin/env python3
"""
Evaluation script for ASR decoding assignment.
Runs all experiments and generates results for the report.
"""

import csv
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import jiwer
import torch
import torchaudio
from wav2vec2decoder import Wav2Vec2Decoder


def load_manifest(manifest_path: str) -> List[Tuple[str, str]]:
    """Load audio paths and references from a manifest CSV file."""
    samples = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append((row['path'], row['text']))
    return samples


def evaluate_decoder(
    decoder: Wav2Vec2Decoder,
    manifest_path: str,
    method: str = "greedy",
    max_samples: int = None,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Evaluate a decoder on a test set.
    
    Returns:
        Dict with 'wer', 'cer', and 'time' keys.
    """
    samples = load_manifest(manifest_path)
    if max_samples:
        samples = samples[:max_samples]
    
    total_wer = 0.0
    total_cer = 0.0
    total_time = 0.0
    num_samples = len(samples)
    
    for i, (audio_path, reference) in enumerate(samples):
        try:
            audio_input, sr = torchaudio.load(audio_path, backend="soundfile")
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio_input = resampler(audio_input)
            
            start_time = time.time()
            hypothesis = decoder.decode(audio_input, method=method)
            elapsed = time.time() - start_time
            total_time += elapsed
            
            wer = jiwer.wer(reference, hypothesis)
            cer = jiwer.cer(reference, hypothesis)
            total_wer += wer
            total_cer += cer
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{num_samples}, Running WER: {total_wer / (i + 1):.4f}")
                
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            num_samples -= 1
    
    return {
        'wer': total_wer / num_samples if num_samples > 0 else float('inf'),
        'cer': total_cer / num_samples if num_samples > 0 else float('inf'),
        'time': total_time
    }


def run_task1_and_2():
    """Task1: Greedy decode, Task2: Beam search decode on LibriSpeech."""
    print("\n" + "=" * 60)
    print("TASK 1 & 2: Greedy and Beam Search on LibriSpeech test-other")
    print("=" * 60)
    
    manifest_path = "data/librispeech_test_other/manifest.csv"
    
    # Test if manifest exists
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        print("Skipping LibriSpeech evaluation.")
        return None
    
    decoder = Wav2Vec2Decoder(lm_model_path=None, beam_width=10)
    
    # Greedy decoding
    print("\n[Task1] Greedy Decoding...")
    greedy_results = evaluate_decoder(decoder, manifest_path, method="greedy", verbose=True)
    print(f"  Greedy WER: {greedy_results['wer']:.4f} ({greedy_results['wer']*100:.2f}%)")
    print(f"  Greedy CER: {greedy_results['cer']:.4f} ({greedy_results['cer']*100:.2f}%)")
    
    # Beam search with different widths
    beam_widths = [1, 3, 10, 50]
    beam_results = {}
    
    for width in beam_widths:
        print(f"\n[Task2] Beam Search (width={width})...")
        decoder.beam_width = width
        results = evaluate_decoder(decoder, manifest_path, method="beam", verbose=True)
        beam_results[width] = results
        print(f"  Beam (width={width}) WER: {results['wer']:.4f} ({results['wer']*100:.2f}%)")
        print(f"  Beam (width={width}) CER: {results['cer']:.4f} ({results['cer']*100:.2f}%)")
        print(f"  Time: {results['time']:.2f}s")
    
    return {
        'greedy': greedy_results,
        'beam': beam_results
    }


def run_task3():
    """Task3: Temperature sweep on LibriSpeech."""
    print("\n" + "=" * 60)
    print("TASK 3: Temperature Sweep on LibriSpeech (Greedy)")
    print("=" * 60)
    
    manifest_path = "data/librispeech_test_other/manifest.csv"
    
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return None
    
    temperatures = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    results = {}
    
    for temp in temperatures:
        print(f"\nTemperature T={temp}...")
        decoder = Wav2Vec2Decoder(lm_model_path=None, temperature=temp)
        temp_results = evaluate_decoder(decoder, manifest_path, method="greedy", verbose=True)
        results[temp] = temp_results
        print(f"  WER: {temp_results['wer']:.4f} ({temp_results['wer']*100:.2f}%)")
        print(f"  CER: {temp_results['cer']:.4f} ({temp_results['cer']*100:.2f}%)")
    
    return results


def run_task4():
    """Task4: Beam search with LM (shallow fusion) on LibriSpeech."""
    print("\n" + "=" * 60)
    print("TASK 4: Beam Search with LM (Shallow Fusion)")
    print("=" * 60)
    
    manifest_path = "data/librispeech_test_other/manifest.csv"
    
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return None
    
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    betas = [0.0, 0.5, 1.0, 1.5]
    
    results = {}
    
    for alpha in alphas:
        for beta in betas:
            print(f"\nAlpha={alpha}, Beta={beta}...")
            decoder = Wav2Vec2Decoder(
                lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
                beam_width=10,
                alpha=alpha,
                beta=beta
            )
            ab_results = evaluate_decoder(decoder, manifest_path, method="beam_lm", verbose=True)
            results[(alpha, beta)] = ab_results
            print(f"  WER: {ab_results['wer']:.4f} ({ab_results['wer']*100:.2f}%)")
            print(f"  CER: {ab_results['cer']:.4f} ({ab_results['cer']*100:.2f}%)")
    
    # Find best configuration
    best_config = min(results.items(), key=lambda x: x[1]['wer'])
    print(f"\nBest config: alpha={best_config[0][0]}, beta={best_config[0][1]}")
    print(f"Best WER: {best_config[1]['wer']:.4f} ({best_config[1]['wer']*100:.2f}%)")
    
    return results


def run_task6():
    """Task6: LM rescoring on LibriSpeech."""
    print("\n" + "=" * 60)
    print("TASK 6: LM Rescoring")
    print("=" * 60)
    
    manifest_path = "data/librispeech_test_other/manifest.csv"
    
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return None
    
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    betas = [0.0, 0.5, 1.0, 1.5]
    
    results = {}
    
    for alpha in alphas:
        for beta in betas:
            print(f"\nRescoring: Alpha={alpha}, Beta={beta}...")
            decoder = Wav2Vec2Decoder(
                lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
                beam_width=10,
                alpha=alpha,
                beta=beta
            )
            ab_results = evaluate_decoder(decoder, manifest_path, method="beam_lm_rescore", verbose=True)
            results[(alpha, beta)] = ab_results
            print(f"  WER: {ab_results['wer']:.4f} ({ab_results['wer']*100:.2f}%)")
            print(f"  CER: {ab_results['cer']:.4f} ({ab_results['cer']*100:.2f}%)")
    
    # Find best configuration
    best_config = min(results.items(), key=lambda x: x[1]['wer'])
    print(f"\nBest rescoring config: alpha={best_config[0][0]}, beta={best_config[0][1]}")
    print(f"Best WER: {best_config[1]['wer']:.4f} ({best_config[1]['wer']*100:.2f}%)")
    
    return results


def run_task7():
    """Task7: Evaluate on Earnings22 (out-of-domain)."""
    print("\n" + "=" * 60)
    print("TASK 7: Evaluation on Earnings22 (Out-of-Domain)")
    print("=" * 60)
    
    manifest_path = "data/earnings22_test/manifest.csv"
    
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return None
    
    results = {}
    
    # Greedy
    print("\nGreedy on Earnings22...")
    decoder = Wav2Vec2Decoder(lm_model_path=None, beam_width=10)
    results['greedy'] = evaluate_decoder(decoder, manifest_path, method="greedy", verbose=True)
    print(f"  WER: {results['greedy']['wer']:.4f} ({results['greedy']['wer']*100:.2f}%)")
    print(f"  CER: {results['greedy']['cer']:.4f} ({results['greedy']['cer']*100:.2f}%)")
    
    # Beam search
    print("\nBeam Search on Earnings22...")
    results['beam'] = evaluate_decoder(decoder, manifest_path, method="beam", verbose=True)
    print(f"  WER: {results['beam']['wer']:.4f} ({results['beam']['wer']*100:.2f}%)")
    print(f"  CER: {results['beam']['cer']:.4f} ({results['beam']['cer']*100:.2f}%)")
    
    # Beam + LM (shallow fusion) - using reasonable default params
    print("\nBeam + LM (Shallow Fusion) on Earnings22...")
    decoder_lm = Wav2Vec2Decoder(
        lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
        beam_width=10,
        alpha=0.1,
        beta=0.5
    )
    results['beam_lm'] = evaluate_decoder(decoder_lm, manifest_path, method="beam_lm", verbose=True)
    print(f"  WER: {results['beam_lm']['wer']:.4f} ({results['beam_lm']['wer']*100:.2f}%)")
    print(f"  CER: {results['beam_lm']['cer']:.4f} ({results['beam_lm']['cer']*100:.2f}%)")
    
    # Beam + LM (rescoring)
    print("\nBeam + LM (Rescoring) on Earnings22...")
    results['beam_lm_rescore'] = evaluate_decoder(decoder_lm, manifest_path, method="beam_lm_rescore", verbose=True)
    print(f"  WER: {results['beam_lm_rescore']['wer']:.4f} ({results['beam_lm_rescore']['wer']*100:.2f}%)")
    print(f"  CER: {results['beam_lm_rescore']['cer']:.4f} ({results['beam_lm_rescore']['cer']*100:.2f}%)")
    
    return results


def run_task5():
    """Task5: Evaluate with 4-gram LM on LibriSpeech."""
    print("\n" + "=" * 60)
    print("TASK 5: 4-gram LM Evaluation on LibriSpeech")
    print("=" * 60)
    
    manifest_path = "data/librispeech_test_other/manifest.csv"
    
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return None
    
    # Use best params from Task 4: alpha=0.05, beta=0.0
    print("\n4-gram LM with alpha=0.05, beta=0.0...")
    decoder = Wav2Vec2Decoder(
        lm_model_path="lm/4-gram.arpa.gz",
        beam_width=10,
        alpha=0.05,
        beta=0.0
    )
    results_4gram = evaluate_decoder(decoder, manifest_path, method="beam_lm", verbose=True)
    print(f"  4-gram LM WER: {results_4gram['wer']:.4f} ({results_4gram['wer']*100:.2f}%)")
    print(f"  4-gram LM CER: {results_4gram['cer']:.4f} ({results_4gram['cer']*100:.2f}%)")
    
    # Also test rescoring
    print("\n4-gram LM Rescoring with alpha=0.01, beta=0.5...")
    decoder.alpha = 0.01
    decoder.beta = 0.5
    results_4gram_rescore = evaluate_decoder(decoder, manifest_path, method="beam_lm_rescore", verbose=True)
    print(f"  4-gram Rescore WER: {results_4gram_rescore['wer']:.4f} ({results_4gram_rescore['wer']*100:.2f}%)")
    print(f"  4-gram Rescore CER: {results_4gram_rescore['cer']:.4f} ({results_4gram_rescore['cer']*100:.2f}%)")
    
    return {'beam_lm': results_4gram, 'beam_lm_rescore': results_4gram_rescore}


def run_task7b():
    """Task7b: Temperature sweep on Earnings22."""
    print("\n" + "=" * 60)
    print("TASK 7b: Temperature Sweep on Earnings22")
    print("=" * 60)
    
    manifest_path = "data/earnings22_test/manifest.csv"
    
    if not os.path.exists(manifest_path):
        print(f"Manifest not found: {manifest_path}")
        return None
    
    temperatures = [0.5, 1.0, 1.5, 2.0]
    
    greedy_results = {}
    lm_results = {}
    
    for temp in temperatures:
        print(f"\n=== Temperature T={temp} ===")
        
        # Greedy
        print(f"Greedy T={temp}...")
        decoder = Wav2Vec2Decoder(lm_model_path=None, temperature=temp)
        greedy_results[temp] = evaluate_decoder(decoder, manifest_path, method="greedy", verbose=True)
        print(f"  Greedy WER: {greedy_results[temp]['wer']:.4f}")
        
        # Beam + LM
        print(f"Beam + LM T={temp}...")
        decoder_lm = Wav2Vec2Decoder(
            lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=10,
            alpha=0.1,
            beta=0.5,
            temperature=temp
        )
        lm_results[temp] = evaluate_decoder(decoder_lm, manifest_path, method="beam_lm", verbose=True)
        print(f"  LM WER: {lm_results[temp]['wer']:.4f}")
    
    return {'greedy': greedy_results, 'beam_lm': lm_results}


def quick_test():
    """Quick test on examples to verify implementation."""
    print("\n" + "=" * 60)
    print("QUICK TEST on examples/")
    print("=" * 60)
    
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
    
    # Test without LM first
    decoder = Wav2Vec2Decoder(lm_model_path=None, beam_width=10)
    
    for audio_path, reference in test_samples:
        if not os.path.exists(audio_path):
            print(f"Skipping {audio_path} (not found)")
            continue
            
        print(f"\n{'='*60}")
        print(f"REF: {reference}")
        
        audio_input, sr = torchaudio.load(audio_path, backend="soundfile")
        
        for method in ["greedy", "beam"]:
            try:
                hyp = decoder.decode(audio_input, method=method)
                wer = jiwer.wer(reference, hyp)
                cer = jiwer.cer(reference, hyp)
                print(f"  [{method}] {hyp}")
                print(f"           WER={wer:.2%}  CER={cer:.2%}")
            except Exception as e:
                print(f"  [{method}] Error: {e}")
    
    # Test with LM
    print("\n" + "=" * 60)
    print("Testing with LM...")
    decoder_lm = Wav2Vec2Decoder(
        lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
        beam_width=10,
        alpha=0.1,
        beta=0.5
    )
    
    for audio_path, reference in test_samples[:2]:  # Just first 2 for LM test
        if not os.path.exists(audio_path):
            continue
            
        print(f"\n{'='*60}")
        print(f"REF: {reference}")
        
        audio_input, sr = torchaudio.load(audio_path, backend="soundfile")
        
        for method in ["beam_lm", "beam_lm_rescore"]:
            try:
                hyp = decoder_lm.decode(audio_input, method=method)
                wer = jiwer.wer(reference, hyp)
                cer = jiwer.cer(reference, hyp)
                print(f"  [{method}] {hyp}")
                print(f"           WER={wer:.2%}  CER={cer:.2%}")
            except Exception as e:
                print(f"  [{method}] Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate ASR decoder")
    parser.add_argument("--quick", action="store_true", help="Run quick test on examples only")
    parser.add_argument("--task1", action="store_true", help="Run Task 1&2 (Greedy/Beam on LibriSpeech)")
    parser.add_argument("--task3", action="store_true", help="Run Task 3 (Temperature sweep)")
    parser.add_argument("--task4", action="store_true", help="Run Task 4 (LM shallow fusion)")
    parser.add_argument("--task6", action="store_true", help="Run Task 6 (LM rescoring)")
    parser.add_argument("--task7", action="store_true", help="Run Task 7 (Earnings22 eval)")
    parser.add_argument("--task5", action="store_true", help="Run Task 5 (4-gram LM eval)")
    parser.add_argument("--task7b", action="store_true", help="Run Task 7b (Temperature on Earnings22)")
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    
    args = parser.parse_args()
    
    # Check for CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if args.quick:
        quick_test()
    elif args.all:
        run_task1_and_2()
        run_task3()
        run_task4()
        run_task6()
        run_task7()
        run_task7b()
    else:
        if args.task1:
            run_task1_and_2()
        if args.task3:
            run_task3()
        if args.task4:
            run_task4()
        if args.task6:
            run_task6()
        if args.task5:
            run_task5()
        if args.task7:
            run_task7()
        if args.task7b:
            run_task7b()
        
        # Default: run quick test
        if not any([args.task1, args.task3, args.task4, args.task5, args.task6, args.task7, args.task7b]):
            print("Running quick test. Use --all or specific task flags for full evaluation.")
            quick_test()
