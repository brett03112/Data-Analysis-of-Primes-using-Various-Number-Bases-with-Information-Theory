import math
import os

def simple_sieve(limit):
    """
    Generate primes up to limit using the Sieve of Eratosthenes.
    This is used to generate small primes that will be used in the segmented sieve.
    """
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    
    return [i for i in range(limit + 1) if sieve[i]]

def segmented_sieve(n, report_progress=True):
    """
    Generate primes up to n using a segmented sieve approach.
    This is memory efficient for large ranges.
    """
    # Find all primes up to sqrt(n)
    limit = int(math.sqrt(n)) + 1
    print(f"Generating base primes up to {limit}...")
    base_primes = simple_sieve(limit)
    print(f"Found {len(base_primes)} base primes.")
    
    # Write the base primes to the file first
    with open('decimal_primes.txt', 'w') as f:
        pass  # Just create/clear the file
    
    write_primes_to_file(base_primes, 'decimal_primes.txt')
    
    # Use a smaller segment size for better memory efficiency
    # and more frequent progress updates
    # For extremely large n, use a smaller segment size to avoid memory issues
    if n > 10**10:  # If n is greater than 10 billion
        segment_size = 10000000  # 10 million
    else:
        segment_size = min(limit, 1000000)  # 1 million or sqrt(n), whichever is smaller
    
    total_segments = (n - limit) // segment_size + 1
    print(f"Processing {total_segments} segments of size {segment_size}...")
    
    # Process each segment
    for segment_idx, low in enumerate(range(limit + 1, n + 1, segment_size)):
        high = min(low + segment_size - 1, n)
        
        if report_progress:
            if segment_idx % max(1, total_segments // 100) == 0:  # Report progress more frequently for large ranges
                progress = (segment_idx / total_segments) * 100
                print(f"Progress: {progress:.2f}% - Processing segment {segment_idx+1}/{total_segments} [{low} to {high}]")
        
        # Initialize segment array
        segment = [True] * (high - low + 1)
        
        # Mark composites in current segment
        for prime in base_primes:
            # Find the first multiple of prime in the current segment
            start = max(prime * prime, ((low + prime - 1) // prime) * prime)
            
            # Mark all multiples of prime in current segment as composite
            for i in range(start, high + 1, prime):
                segment[i - low] = False
        
        # Collect primes in current segment
        segment_primes = [i + low for i in range(high - low + 1) if segment[i]]
        
        # Write segment primes to file
        write_primes_to_file(segment_primes, 'decimal_primes.txt')
        
        # Flush the file to ensure data is written even if the process is interrupted
        if segment_idx % 10 == 0:
            import gc
            gc.collect()  # Force garbage collection to free memory

def write_primes_to_file(primes, filename, numbers_per_line=15):
    """Write primes to file with specified number of primes per line."""
    with open(filename, 'a') as f:
        line = []
        for prime in primes:
            line.append(str(prime))
            if len(line) == numbers_per_line:
                f.write(' '.join(line) + '\n')
                line = []
        
        # Write any remaining primes
        if line:
            f.write(' '.join(line) + '\n')

def convert_primes():
    """
    Read decimal primes from file and convert to octal, hexadecimal, and binary.
    Write the results to respective files.
    """
    # Clear output files
    open('octal_primes.txt', 'w').close()
    open('hexadecimal_primes.txt', 'w').close()
    open('binary_primes.txt', 'w').close()
    
    # Process primes in batches to avoid loading the entire file into memory
    with open('decimal_primes.txt', 'r') as decimal_file:
        octal_file = open('octal_primes.txt', 'w')
        hex_file = open('hexadecimal_primes.txt', 'w')
        binary_file = open('binary_primes.txt', 'w')
        
        octal_line = []
        hex_line = []
        
        for line in decimal_file:
            primes = line.strip().split()
            
            for prime in primes:
                # Convert to integer
                prime_int = int(prime)
                
                # Convert to octal and hexadecimal
                octal_line.append(oct(prime_int)[2:])  # Remove '0o' prefix
                hex_line.append(hex(prime_int)[2:])    # Remove '0x' prefix
                
                # Write binary (one per line)
                binary_file.write(bin(prime_int)[2:] + '\n')  # Remove '0b' prefix
                
                # Write octal and hex when we have 15 numbers
                if len(octal_line) == 15:
                    octal_file.write(' '.join(octal_line) + '\n')
                    octal_line = []
                
                if len(hex_line) == 15:
                    hex_file.write(' '.join(hex_line) + '\n')
                    hex_line = []
        
        # Write any remaining numbers
        if octal_line:
            octal_file.write(' '.join(octal_line) + '\n')
        
        if hex_line:
            hex_file.write(' '.join(hex_line) + '\n')
        
        # Close files
        octal_file.close()
        hex_file.close()
        binary_file.close()

def main():
    """
    Main entry point for prime number generation and conversion to different bases.

    This program takes command line arguments to control the upper limit for prime generation
    and to choose whether to run with the full limit of 1 trillion (WARNING: This will take a very long time)
    or to skip prime generation and only convert existing primes to different bases.

    The following options are available:

        --limit <int>  : Upper limit for prime generation (default: 1,000,000)
        --full         : Run with the full limit of 1 trillion (WARNING: This will take a very long time)
        --convert-only : Skip prime generation and only convert existing primes to different bases

    The program will print the total execution time and the time taken for each step (prime generation and conversion) to the console.
    """
    
    import argparse
    import time
    import os
    import psutil
    
    parser = argparse.ArgumentParser(description='Generate prime numbers and convert to different bases.')
    parser.add_argument('--limit', type=int, default=1000000, 
                        help='Upper limit for prime generation (default: 1,000,000)')
    parser.add_argument('--full', action='store_true',
                        help='Run with the full limit of 1 trillion (WARNING: This will take a very long time)')
    parser.add_argument('--convert-only', action='store_true',
                        help='Skip prime generation and only convert existing primes to different bases')
    
    args = parser.parse_args()
    
    # Set the upper limit
    if args.full:
        n = 1000000000000  # 1 trillion
        print("WARNING: Calculating all primes up to 1 trillion will take an extremely long time!")
        print("This could take days or weeks depending on your hardware.")
        print("Consider using a smaller limit for testing purposes.")
        
        # Check available memory
        mem = psutil.virtual_memory()
        print(f"Available memory: {mem.available / (1024**3):.2f} GB")
        print(f"Total memory: {mem.total / (1024**3):.2f} GB")
        
        # Ask for confirmation
        confirm = input("Are you sure you want to continue? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled.")
            return
    else:
        n = args.limit
    
    start_time = time.time()
    
    if not args.convert_only:
        print(f"Finding primes up to {n}...")
        segmented_sieve(n)
        sieve_time = time.time() - start_time
        print(f"Sieve completed in {sieve_time:.2f} seconds.")
    else:
        if not os.path.exists('decimal_primes.txt'):
            print("Error: decimal_primes.txt not found. Cannot perform conversion only.")
            return
        print("Skipping prime generation, using existing decimal_primes.txt file.")
    
    print("Converting primes to different number systems...")
    convert_start = time.time()
    convert_primes()
    convert_time = time.time() - convert_start
    
    print(f"Conversion completed in {convert_time:.2f} seconds.")
    print(f"Total execution time: {time.time() - start_time:.2f} seconds.")
    print("Done!")

if __name__ == "__main__":
    main()
