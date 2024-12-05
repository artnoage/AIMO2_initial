import random
from collections import Counter

def sample_n():
    # Calculate normalization constant (sum of 3^(-n) for n=1 to 10)
    norm_const = sum(3**(-i) for i in range(1, 11))
    
    # Generate random number between 0 and 1
    r = random.random()
    cumsum = 0
    n = 1
    
    # Find n where cumulative probability exceeds r
    while n <= 10:
        cumsum += (3**(-n)) / norm_const
        if r <= cumsum:
            break
        n += 1
    return n

def main():
    # Sample 100 times
    samples = [sample_n() for _ in range(100)]
    
    # Count occurrences
    counts = Counter(samples)
    
    # Calculate theoretical probabilities
    norm_const = sum(3**(-i) for i in range(1, 11))
    theoretical = {n: (3**(-n))/norm_const for n in range(1, 11)}
    
    # Print results
    print("\nSampling Results (100 samples):")
    print("-" * 50)
    print("n | Count | Percentage | Theoretical")
    print("-" * 50)
    for n in range(1, 11):
        count = counts[n]
        percentage = count/100
        theo_prob = theoretical[n]
        print(f"{n:2d} | {count:5d} | {percentage:9.1%} | {theo_prob:10.1%}")

if __name__ == "__main__":
    main()
