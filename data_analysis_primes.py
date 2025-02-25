#!/usr/bin/env python3
"""
Prime Number Analysis using Information Theory

This program analyzes patterns in prime numbers across different base representations
(decimal, binary, octal, and hexadecimal) using concepts from information theory.

The analysis includes:
- Entropy calculations for digit distributions
- Conditional entropy and mutual information analysis
- Cross-base pattern detection
- Markov chain modeling of digit transitions
- Segmentation analysis to find optimal "units"
"""

import os
import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from itertools import product, islice
import multiprocessing as mp
from functools import partial

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class PrimeAnalyzer:
    """Main class for analyzing prime numbers using information theory."""
    
    def __init__(self, max_primes=None, sample_size=10000, n_gram_sizes=(2, 3, 4)):
        """
        Initialize the analyzer.
        
        Args:
            max_primes: Maximum number of primes to analyze (None for all)
            sample_size: Size of random samples for intensive computations
            n_gram_sizes: Tuple of n-gram sizes to analyze
        """
        self.max_primes = max_primes
        self.sample_size = sample_size
        self.n_gram_sizes = n_gram_sizes
        self.bases = {
            'decimal': {'file': 'decimal_primes.txt', 'base': 10, 'symbols': '0123456789'},
            'binary': {'file': 'binary_primes.txt', 'base': 2, 'symbols': '01'},
            'octal': {'file': 'octal_primes.txt', 'base': 8, 'symbols': '01234567'},
            'hexadecimal': {'file': 'hexadecimal_primes.txt', 'base': 16, 
                           'symbols': '0123456789abcdef'}
        }
        
        # Storage for analysis results
        self.entropy_results = {}
        self.transition_matrices = {}
        self.mutual_info_results = {}
        self.cross_base_mi = {}
        self.patterns = {}
        self.segmentation_results = {}
        
        # Check if files exist
        for base_info in self.bases.values():
            if not os.path.exists(base_info['file']):
                raise FileNotFoundError(f"Prime file {base_info['file']} not found")
    
    def load_primes_batch(self, base_name, batch_size=1000, skip=0):
        """
        Load a batch of primes from a file.
        
        Args:
            base_name: Name of the base ('decimal', 'binary', etc.)
            batch_size: Number of primes to load
            skip: Number of primes to skip
        
        Returns:
            List of prime numbers as strings
        """
        file_path = self.bases[base_name]['file']
        primes = []
        
        with open(file_path, 'r') as f:
            # Skip lines if needed
            if skip > 0:
                if base_name == 'binary':
                    # Binary file has one prime per line
                    for _ in range(skip):
                        next(f, None)
                else:
                    # Other files have multiple primes per line
                    lines_to_skip = skip // 15  # 15 primes per line
                    for _ in range(lines_to_skip):
                        next(f, None)
                    
                    # Handle partial line
                    if skip % 15 > 0:
                        line = next(f, "").strip()
                        primes_in_line = line.split()
                        primes_to_keep = primes_in_line[skip % 15:]
                        primes.extend(primes_to_keep)
                        batch_size -= len(primes_to_keep)
            
            # Read the batch
            if base_name == 'binary':
                # Binary file has one prime per line
                for _ in range(batch_size):
                    line = next(f, None)
                    if line is None:
                        break
                    primes.append(line.strip())
            else:
                # Other files have multiple primes per line
                while len(primes) < batch_size:
                    line = next(f, None)
                    if line is None:
                        break
                    primes_in_line = line.strip().split()
                    primes.extend(primes_in_line[:batch_size - len(primes)])
        
        return primes
    
    def count_primes(self, base_name):
        """Count the total number of primes in a file."""
        file_path = self.bases[base_name]['file']
        count = 0
        
        with open(file_path, 'r') as f:
            if base_name == 'binary':
                # Binary file has one prime per line
                for _ in f:
                    count += 1
            else:
                # Other files have multiple primes per line
                for line in f:
                    count += len(line.strip().split())
        
        return count
    
    def calculate_entropy(self, sequence, base=2):
        """
        Calculate Shannon entropy of a sequence.
        
        Args:
            sequence: String or list of symbols
            base: Logarithm base for entropy calculation
        
        Returns:
            Entropy value in specified units (default: bits)
        """
        if not sequence:
            return 0
            
        # Count occurrences of each symbol
        counter = Counter(sequence)
        total = sum(counter.values())
        
        # Calculate entropy
        entropy = 0
        for count in counter.values():
            p = count / total
            entropy -= p * math.log(p, base)
            
        return entropy
    
    def calculate_conditional_entropy(self, sequences, condition_len=1, target_pos=1, base=2):
        """
        Calculate conditional entropy H(X_t | X_{t-k}, ..., X_{t-1}).
        
        Args:
            sequences: List of sequences
            condition_len: Length of the conditioning sequence
            target_pos: Position of the target relative to the end of condition
            base: Logarithm base for entropy calculation
            
        Returns:
            Conditional entropy value
        """
        # Count joint occurrences
        joint_counts = defaultdict(Counter)
        condition_counts = Counter()
        
        for seq in sequences:
            if len(seq) < condition_len + target_pos:
                continue
                
            for i in range(len(seq) - condition_len - target_pos + 1):
                condition = seq[i:i+condition_len]
                target = seq[i+condition_len+target_pos-1]
                
                joint_counts[condition][target] += 1
                condition_counts[condition] += 1
        
        # Calculate conditional entropy
        cond_entropy = 0
        total_observations = sum(condition_counts.values())
        
        for condition, count in condition_counts.items():
            p_condition = count / total_observations
            
            # Calculate entropy of the conditional distribution
            conditional_distribution = joint_counts[condition]
            total = sum(conditional_distribution.values())
            
            entropy = 0
            for target_count in conditional_distribution.values():
                p = target_count / total
                entropy -= p * math.log(p, base)
            
            cond_entropy += p_condition * entropy
            
        return cond_entropy
    
    def calculate_mutual_information(self, sequences, pos1=0, pos2=1, base=2):
        """
        Calculate mutual information between symbols at different positions.
        
        Args:
            sequences: List of sequences
            pos1: First position
            pos2: Second position
            base: Logarithm base for calculation
            
        Returns:
            Mutual information value
        """
        # Ensure pos1 < pos2
        if pos1 > pos2:
            pos1, pos2 = pos2, pos1
            
        # Count occurrences
        joint_counts = defaultdict(Counter)
        pos1_counts = Counter()
        pos2_counts = Counter()
        total = 0
        
        for seq in sequences:
            if len(seq) <= max(pos1, pos2):
                continue
                
            for i in range(len(seq) - max(pos1, pos2)):
                sym1 = seq[i + pos1]
                sym2 = seq[i + pos2]
                
                joint_counts[sym1][sym2] += 1
                pos1_counts[sym1] += 1
                pos2_counts[sym2] += 1
                total += 1
        
        if total == 0:
            return 0
            
        # Calculate mutual information
        mi = 0
        for sym1, counters in joint_counts.items():
            p_sym1 = pos1_counts[sym1] / total
            
            for sym2, count in counters.items():
                p_sym2 = pos2_counts[sym2] / total
                p_joint = count / total
                
                mi += p_joint * math.log(p_joint / (p_sym1 * p_sym2), base)
                
        return mi
    
    def build_transition_matrix(self, sequences, n=1):
        """
        Build n-th order transition matrix from sequences.
        
        Args:
            sequences: List of sequences
            n: Order of the Markov model (1 for first-order)
            
        Returns:
            Dictionary representing transition probabilities
        """
        transitions = defaultdict(Counter)
        
        for seq in sequences:
            if len(seq) <= n:
                continue
                
            for i in range(len(seq) - n):
                state = seq[i:i+n]
                next_sym = seq[i+n]
                transitions[state][next_sym] += 1
        
        # Convert to probabilities
        transition_probs = {}
        for state, counters in transitions.items():
            total = sum(counters.values())
            transition_probs[state] = {sym: count/total for sym, count in counters.items()}
            
        return transition_probs
    
    def find_high_surprisal_transitions(self, trans_matrix, top_n=10):
        """
        Find transitions with highest surprisal (information content).
        
        Args:
            trans_matrix: Transition matrix
            top_n: Number of top transitions to return
            
        Returns:
            List of (state, next_sym, surprisal) tuples
        """
        surprisals = []
        
        for state, transitions in trans_matrix.items():
            for next_sym, prob in transitions.items():
                surprisal = -math.log2(prob)
                surprisals.append((state, next_sym, surprisal))
        
        # Sort by surprisal (descending)
        surprisals.sort(key=lambda x: x[2], reverse=True)
        
        return surprisals[:top_n]
    
    def calculate_cross_base_mutual_information(self, base1, base2, sample_size=1000):
        """
        Calculate mutual information between representations in different bases.
        
        Args:
            base1: First base name
            base2: Second base name
            sample_size: Number of primes to sample
            
        Returns:
            Dictionary of mutual information values for different digit positions
        """
        # Load samples from both bases
        primes1 = self.load_primes_batch(base1, sample_size)
        primes2 = self.load_primes_batch(base2, sample_size)
        
        # Ensure equal length
        min_len = min(len(primes1), len(primes2))
        primes1 = primes1[:min_len]
        primes2 = primes2[:min_len]
        
        # Calculate mutual information for different digit positions
        results = {}
        max_positions = 5  # Limit to first few positions for efficiency
        
        for pos1 in range(max_positions):
            for pos2 in range(max_positions):
                # Extract digits at specified positions (from the end)
                digits1 = []
                digits2 = []
                
                for p1, p2 in zip(primes1, primes2):
                    if len(p1) > pos1 and len(p2) > pos2:
                        # Take positions from the end to align by significance
                        digits1.append(p1[-(pos1+1)] if pos1 < len(p1) else '0')
                        digits2.append(p2[-(pos2+1)] if pos2 < len(p2) else '0')
                
                # Calculate mutual information
                joint_counts = defaultdict(Counter)
                pos1_counts = Counter(digits1)
                pos2_counts = Counter(digits2)
                total = len(digits1)
                
                for d1, d2 in zip(digits1, digits2):
                    joint_counts[d1][d2] += 1
                
                mi = 0
                for d1, counters in joint_counts.items():
                    p_d1 = pos1_counts[d1] / total
                    
                    for d2, count in counters.items():
                        p_d2 = pos2_counts[d2] / total
                        p_joint = count / total
                        
                        if p_joint > 0:  # Avoid log(0)
                            mi += p_joint * math.log2(p_joint / (p_d1 * p_d2))
                
                results[f"{pos1},{pos2}"] = mi
        
        return results
    
    def find_optimal_segmentation(self, sequences, max_segment_len=4):
        """
        Find optimal segmentation of sequences to minimize entropy.
        
        Args:
            sequences: List of sequences
            max_segment_len: Maximum segment length to consider
            
        Returns:
            Dictionary with results for different segment lengths
        """
        results = {}
        
        for segment_len in range(1, max_segment_len + 1):
            # Segment sequences
            all_segments = []
            
            for seq in sequences:
                # Pad sequence if needed
                padded_seq = seq + '0' * (segment_len - (len(seq) % segment_len))
                
                # Extract segments
                segments = [padded_seq[i:i+segment_len] 
                           for i in range(0, len(padded_seq), segment_len)]
                all_segments.extend(segments)
            
            # Calculate entropy of segments
            entropy = self.calculate_entropy(all_segments)
            normalized_entropy = entropy / segment_len  # Normalize by segment length
            
            # Count unique segments
            unique_segments = len(set(all_segments))
            
            results[segment_len] = {
                'entropy': entropy,
                'normalized_entropy': normalized_entropy,
                'unique_segments': unique_segments,
                'total_segments': len(all_segments)
            }
        
        return results
    
    def find_recurring_patterns(self, sequences, pattern_len=3, min_occurrences=5):
        """
        Find recurring patterns in sequences.
        
        Args:
            sequences: List of sequences
            pattern_len: Length of patterns to look for
            min_occurrences: Minimum number of occurrences to consider
            
        Returns:
            Dictionary of patterns and their counts
        """
        pattern_counts = Counter()
        
        for seq in sequences:
            if len(seq) < pattern_len:
                continue
                
            for i in range(len(seq) - pattern_len + 1):
                pattern = seq[i:i+pattern_len]
                pattern_counts[pattern] += 1
        
        # Filter by minimum occurrences
        filtered_patterns = {pattern: count for pattern, count 
                            in pattern_counts.items() if count >= min_occurrences}
        
        return filtered_patterns
    
    def analyze_digit_positions(self, base_name, sample_size=10000):
        """
        Analyze the distribution of digits at different positions.
        
        Args:
            base_name: Name of the base to analyze
            sample_size: Number of primes to sample
            
        Returns:
            Dictionary with position-specific entropy and distribution
        """
        primes = self.load_primes_batch(base_name, sample_size)
        symbols = self.bases[base_name]['symbols']
        max_positions = 10  # Analyze first 10 positions
        
        results = {}
        
        for pos in range(max_positions):
            # Extract digits at specified position (from the end)
            digits = []
            
            for prime in primes:
                if len(prime) > pos:
                    digits.append(prime[-(pos+1)])
            
            # Calculate distribution and entropy
            distribution = Counter(digits)
            for sym in symbols:
                if sym not in distribution:
                    distribution[sym] = 0
            
            entropy = self.calculate_entropy(digits)
            
            results[pos] = {
                'entropy': entropy,
                'distribution': dict(distribution)
            }
        
        return results
    
    def analyze_base(self, base_name):
        """
        Perform comprehensive analysis for a single base.
        
        Args:
            base_name: Name of the base to analyze
            
        Returns:
            Dictionary with analysis results
        """
        print(f"Analyzing {base_name} representation...")
        
        # Load a sample of primes
        sample_size = min(self.sample_size, self.count_primes(base_name))
        primes = self.load_primes_batch(base_name, sample_size)
        
        results = {}
        
        # 1. Basic entropy of digit distribution
        all_digits = ''.join(primes)
        results['overall_entropy'] = self.calculate_entropy(all_digits)
        
        # 2. Position-specific analysis
        results['position_analysis'] = self.analyze_digit_positions(base_name)
        
        # 3. Transition matrices for different n-gram sizes
        results['transition_matrices'] = {}
        for n in self.n_gram_sizes:
            trans_matrix = self.build_transition_matrix(primes, n)
            results['transition_matrices'][n] = trans_matrix
            
            # Find surprising transitions
            surprisals = self.find_high_surprisal_transitions(trans_matrix)
            results[f'high_surprisal_{n}gram'] = surprisals
        
        # 4. Mutual information between positions
        results['mutual_information'] = {}
        max_distance = 5  # Maximum distance between positions
        
        for dist in range(1, max_distance + 1):
            mi = self.calculate_mutual_information(primes, 0, dist)
            results['mutual_information'][dist] = mi
        
        # 5. Segmentation analysis
        results['segmentation'] = self.find_optimal_segmentation(primes)
        
        # 6. Pattern detection
        results['patterns'] = {}
        for pattern_len in range(2, 6):
            patterns = self.find_recurring_patterns(primes, pattern_len)
            results['patterns'][pattern_len] = patterns
        
        return results
    
    def analyze_cross_base_relationships(self):
        """
        Analyze relationships between different base representations.
        
        Returns:
            Dictionary with cross-base analysis results
        """
        print("Analyzing cross-base relationships...")
        results = {}
        
        # Calculate mutual information between bases
        base_pairs = list(product(self.bases.keys(), repeat=2))
        
        for base1, base2 in base_pairs:
            if base1 != base2:
                key = f"{base1}_{base2}"
                results[key] = self.calculate_cross_base_mutual_information(base1, base2)
        
        return results
    
    def run_analysis(self):
        """Run the complete analysis pipeline."""
        start_time = time.time()
        print(f"Starting prime number analysis...")
        
        # Analyze each base
        for base_name in self.bases:
            self.entropy_results[base_name] = self.analyze_base(base_name)
        
        # Analyze cross-base relationships
        self.cross_base_results = self.analyze_cross_base_relationships()
        
        # Generate visualizations
        self.generate_visualizations()
        
        elapsed = time.time() - start_time
        print(f"Analysis completed in {elapsed:.2f} seconds")
        
        # Print key findings
        self.print_key_findings()
    
    def generate_visualizations(self):
        """Generate visualizations of analysis results."""
        print("Generating visualizations...")
        
        # 1. Entropy by position for each base
        plt.figure(figsize=(12, 8))
        
        for base_name in self.bases:
            position_analysis = self.entropy_results[base_name]['position_analysis']
            positions = sorted(position_analysis.keys())
            entropies = [position_analysis[pos]['entropy'] for pos in positions]
            
            plt.plot(positions, entropies, marker='o', label=base_name)
        
        plt.xlabel('Digit Position (from least significant)')
        plt.ylabel('Entropy (bits)')
        plt.title('Entropy by Digit Position Across Different Bases')
        plt.legend()
        plt.grid(True)
        plt.savefig('entropy_by_position.png')
        
        # 2. Mutual information heatmap for cross-base relationships
        base_names = list(self.bases.keys())
        mi_matrix = np.zeros((len(base_names), len(base_names)))
        
        for i, base1 in enumerate(base_names):
            for j, base2 in enumerate(base_names):
                if i != j:
                    key = f"{base1}_{base2}"
                    # Use average MI across positions
                    mi_values = list(self.cross_base_results[key].values())
                    mi_matrix[i, j] = sum(mi_values) / len(mi_values)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(mi_matrix, cmap='viridis')
        plt.colorbar(label='Average Mutual Information (bits)')
        plt.xticks(range(len(base_names)), base_names, rotation=45)
        plt.yticks(range(len(base_names)), base_names)
        plt.title('Cross-Base Mutual Information')
        plt.tight_layout()
        plt.savefig('cross_base_mi.png')
        
        # 3. Normalized entropy by segmentation length
        plt.figure(figsize=(12, 8))
        
        for base_name in self.bases:
            segmentation = self.entropy_results[base_name]['segmentation']
            segment_lengths = sorted(segmentation.keys())
            norm_entropies = [segmentation[length]['normalized_entropy'] 
                             for length in segment_lengths]
            
            plt.plot(segment_lengths, norm_entropies, marker='o', label=base_name)
        
        plt.xlabel('Segment Length')
        plt.ylabel('Normalized Entropy (bits/symbol)')
        plt.title('Normalized Entropy by Segment Length')
        plt.legend()
        plt.grid(True)
        plt.savefig('segmentation_entropy.png')
        
        # 4. Digit distribution for first few positions
        for base_name in self.bases:
            symbols = self.bases[base_name]['symbols']
            position_analysis = self.entropy_results[base_name]['position_analysis']
            
            # Select first 5 positions
            positions = sorted(position_analysis.keys())[:5]
            
            plt.figure(figsize=(15, 10))
            
            for i, pos in enumerate(positions):
                plt.subplot(1, 5, i+1)
                
                distribution = position_analysis[pos]['distribution']
                sorted_symbols = sorted(symbols)
                frequencies = [distribution[sym] for sym in sorted_symbols]
                
                plt.bar(sorted_symbols, frequencies)
                plt.title(f'Position {pos}')
                plt.xlabel('Digit')
                plt.ylabel('Frequency')
            
            plt.suptitle(f'Digit Distribution by Position ({base_name})')
            plt.tight_layout()
            plt.savefig(f'{base_name}_digit_distribution.png')
    
    def print_key_findings(self):
        """Print key findings from the analysis."""
        print("\n===== KEY FINDINGS =====\n")
        
        # 1. Overall entropy comparison
        print("Overall Entropy by Base:")
        for base_name in self.bases:
            entropy = self.entropy_results[base_name]['overall_entropy']
            max_entropy = math.log2(len(self.bases[base_name]['symbols']))
            print(f"  {base_name}: {entropy:.4f} bits (max possible: {max_entropy:.4f})")
        
        print("\n")
        
        # 2. Most surprising transitions
        print("Most Surprising Transitions:")
        for base_name in self.bases:
            print(f"  {base_name}:")
            surprisals = self.entropy_results[base_name]['high_surprisal_2gram'][:3]
            for state, next_sym, surprisal in surprisals:
                print(f"    {state} → {next_sym}: {surprisal:.4f} bits")
        
        print("\n")
        
        # 3. Optimal segmentation
        print("Optimal Segmentation Length by Base:")
        for base_name in self.bases:
            segmentation = self.entropy_results[base_name]['segmentation']
            # Find minimum normalized entropy
            min_entropy = float('inf')
            optimal_length = 0
            
            for length, results in segmentation.items():
                if results['normalized_entropy'] < min_entropy:
                    min_entropy = results['normalized_entropy']
                    optimal_length = length
            
            print(f"  {base_name}: {optimal_length} digits (normalized entropy: {min_entropy:.4f})")
        
        print("\n")
        
        # 4. Strongest cross-base relationships
        print("Strongest Cross-Base Relationships:")
        all_relationships = []
        
        for base1 in self.bases:
            for base2 in self.bases:
                if base1 != base2:
                    key = f"{base1}_{base2}"
                    # Use maximum MI across positions
                    max_mi = max(self.cross_base_results[key].values())
                    max_pos = max(self.cross_base_results[key].items(), key=lambda x: x[1])[0]
                    all_relationships.append((base1, base2, max_pos, max_mi))
        
        # Sort by MI (descending)
        all_relationships.sort(key=lambda x: x[3], reverse=True)
        
        for base1, base2, pos, mi in all_relationships[:5]:
            print(f"  {base1} and {base2} at positions {pos}: {mi:.4f} bits")
        
        print("\n")
        
        # 5. Most common patterns
        print("Most Common Patterns:")
        for base_name in self.bases:
            print(f"  {base_name}:")
            
            # Combine patterns of different lengths
            all_patterns = []
            for length, patterns in self.entropy_results[base_name]['patterns'].items():
                for pattern, count in patterns.items():
                    all_patterns.append((pattern, count, length))
            
            # Sort by count (descending)
            all_patterns.sort(key=lambda x: x[1], reverse=True)
            
            for pattern, count, length in all_patterns[:3]:
                print(f"    '{pattern}' (length {length}): {count} occurrences")
        
        print("\n===== END OF FINDINGS =====\n")


def main():
    """Main entry point for the program."""
    print("Prime Number Analysis using Information Theory")
    print("=============================================")
    
    # Check if prime files exist
    required_files = ['decimal_primes.txt', 'binary_primes.txt', 
                     'octal_primes.txt', 'hexadecimal_primes.txt']
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Error: The following required files are missing: {', '.join(missing_files)}")
        print("Please run primes.py first to generate these files.")
        return
    
    # Create and run analyzer
    analyzer = PrimeAnalyzer(sample_size=50000)
    analyzer.run_analysis()
    
    print("\nAnalysis complete. Visualizations saved to current directory.")


if __name__ == "__main__":
    main()
