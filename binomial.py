from math import factorial
from typing import Union, Optional

def binomial(n: int, k: int) -> Optional[int]:
    """
    Calculate the binomial coefficient C(n,k) = n! / (k! * (n-k)!)
    
    Args:
        n (int): Total number of items
        k (int): Number of items to choose
        
    Returns:
        int: The binomial coefficient, or None if inputs are invalid
        
    Examples:
        >>> binomial(5, 2)  # 5C2 = 10
        10
        >>> binomial(10, 3)  # 10C3 = 120
        120
    """
    try:
        # Input validation
        if not isinstance(n, int) or not isinstance(k, int):
            raise ValueError("Both n and k must be integers")
        if n < 0 or k < 0:
            raise ValueError("Both n and k must be non-negative")
        if k > n:
            raise ValueError("k cannot be greater than n")
            
        # Calculate using the factorial formula
        return factorial(n) // (factorial(k) * factorial(n - k))
        
    except Exception as e:
        print(f"Error calculating binomial coefficient: {str(e)}")
        return None

def pascal_triangle(n: int) -> Optional[list[list[int]]]:
    """
    Generate Pascal's triangle up to row n
    
    Args:
        n (int): Number of rows to generate (0-based indexing)
        
    Returns:
        list[list[int]]: Pascal's triangle as a list of rows, or None if input is invalid
        
    Examples:
        >>> pascal_triangle(3)
        [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1]]
    """
    try:
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be a non-negative integer")
            
        triangle = []
        for i in range(n + 1):
            row = []
            for j in range(i + 1):
                row.append(binomial(i, j))
            triangle.append(row)
        return triangle
        
    except Exception as e:
        print(f"Error generating Pascal's triangle: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    print("Binomial coefficient examples:")
    print(f"C(5,2) = {binomial(5, 2)}")
    print(f"C(10,3) = {binomial(10, 3)}")
    
    print("\nPascal's triangle (first 5 rows):")
    triangle = pascal_triangle(4)
    if triangle:
        for row in triangle:
            print(row)
