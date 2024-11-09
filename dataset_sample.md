# Numina-Olympiads Dataset Sample

## Entry 1

**ID**: 0

### Problem

Let \( a, b, c \) be positive real numbers. Prove that

$$
\frac{1}{a(1+b)}+\frac{1}{b(1+c)}+\frac{1}{c(1+a)} \geq \frac{3}{1+abc},
$$

and that equality occurs if and only if \( a = b = c = 1 \).

### Solution

1. Consider the given inequality:

\[
\frac{1}{a(1+b)}+ \frac{1}{b(1+c)}+ \frac{1}{c(1+a)} \geq \frac{3}{1 + abc}
\]

To simplify, we add \( \frac{3}{1 + abc} \) to both sides. The new inequality becomes:

\[
\frac{1}{a(1+b)} + \frac{1}{b(1+c)} + \frac{1}{c(1+a)} + \frac{3}{1 + abc} \geq \frac{6}{1 + abc}
\]

2. Let's analyze each term with an added \( \frac{1}{1 + abc} \):

\[
\frac{1}{a(1+b)} + \frac{1}{1 + abc}, \quad \frac{1}{b(1+c)} + \frac{1}{1 + abc}, \quad \frac{1}{c(1+a)} + \frac{1}{1 + abc}
\]

We can rewrite them as follows:

\[
\begin{aligned}
\frac{1}{a(1+b)} + \frac{1}{1 + abc} &= \frac{1}{1 + abc} \left( \frac{1 + abc}{a(1+b)} + 1 \right), \\
\frac{1}{b(1+c)} + \frac{1}{1 + abc} &= \frac{1}{1 + abc} \left( \frac{1 + abc}{b(1+c)} + 1 \right), \\
\frac{1}{c(1+a)} + \frac{1}{1 + abc} &= \frac{1}{1 + abc} \left( \frac{1 + abc}{c(1+a)} + 1 \right).
\end{aligned}
\]

3. Simplifying inside the parentheses:

\[
\begin{aligned}
\frac{1 + abc}{a(1+b)} + 1 &= \frac{1 + abc + a(1+b)}{a(1+b)} = \frac{1 + abc + a + ab}{a(1+b)} = \frac{1 + a + ab + abc}{a(1+b)}, \\
\frac{1 + abc}{b(1+c)} + 1 &= \frac{1 + abc + b(1+c)}{b(1+c)} = \frac{1 + abc + b + bc}{b(1+c)} = \frac{1 + b + bc + abc}{b(1+c)}, \\
\frac{1 + abc}{c(1+a)} + 1 &= \frac{1 + abc + c(1+a)}{c(1+a)} = \frac{1 + abc + c + ac}{c(1+a)} = \frac{1 + c + ac + abc}{c(1+a)}.
\end{aligned}
\]

4. Combining the simplified terms, we get:

\[
\begin{aligned}
\frac{1}{a(1+b)} + \frac{1}{1 + abc} &= \frac{1}{1 + abc} \left( \frac{1 + a + ab + abc}{a(1+b)} \right), \\
\frac{1}{b(1+c)} + \frac{1}{1 + abc} &= \frac{1}{1 + abc} \left( \frac{1 + b + bc + abc}{b(1+c)} \right), \\
\frac{1}{c(1+a)} + \frac{1}{1 + abc} &= \frac{1}{1 + abc} \left( \frac{1 + c + ac + abc}{c(1+a)} \right).
\end{aligned}
\]

5. Adding these terms together:

\[
\begin{aligned}
\left( \frac{1 + a + ab + abc}{a(1+b)} + \frac{1 + b + bc + abc}{b(1+c)} + \frac{1 + c + ac + abc}{c(1+a)} \right) \cdot \frac{1}{1 + abc}
\end{aligned}
\]

Each term inside the parentheses \( \frac{ 1 + x + xy + xyz}{x(1+y)} \) paired as \( x + \frac{1}{x} \). By AM-GM inequality, for any positive \( x \):

\[
x + \frac{1}{x} \geq 2
\]

Therefore, each addition results in at least 2, and since we have three such terms:

\[
\frac{1 + a + ab + abc}{a(1+b)} + \frac{1 + b + bc + abc}{b(1+c)} + \frac{1 + c + ac + abc}{c(1+a)} \geq 6
\]

Thus, we have:

\[
\frac{6}{1 + abc}
\]

6. From steps above, we conclude:

\[
\frac{1}{a(1+b)} + \frac{1}{b(1+c)} + \frac{1}{c(1+a)} + \frac{3}{1 + abc} \geq \frac{6}{1 + abc}
\]

This demonstrates that:

\[
\boxed{\frac{1}{a(1+b)} + \frac{1}{b(1+c)} + \frac{1}{c(1+a)} \geq \frac{3}{1 + abc}}
\]

Finally, equality holds if and only if \( a = b = c = 1 \).

---

## Entry 2

**ID**: 1

### Problem

A set consists of five different odd positive integers, each greater than 2. When these five integers are multiplied together, their product is a five-digit integer of the form $AB0AB$, where $A$ and $B$ are digits with $A \neq 0$ and $A \neq B$. (The hundreds digit of the product is zero.) In total, how many different sets of five different odd positive integers have these properties?

### Solution


1. **Observe the Structure of \( N \)**:
    Let \( N = AB0AB \) and let \( t \) be the two-digit integer \( AB \).

    We recognize that \( N = 1001 \cdot t \), where \( 1001 = 11 \cdot 91 = 11 \cdot 7 \cdot 13 \).

    Thus, 
    \[
    N = t \cdot 7 \cdot 11 \cdot 13 
    \]

2. **Formulate the Problem**:
    We need to write \( N \) as the product of 5 distinct odd integers, each greater than 2, and count the possible sets \( S \) of such integers.

3. **Case Analysis**:
   
    - **Case 1: \( S = \{7, 11, 13, m, n\} \)**:
        - Here, 
        \[
        N = 7 \cdot 11 \cdot 13 \cdot m \cdot n 
        \]
        This implies \( t = m \cdot n \). Since \( t \) is a two-digit number,
        \[
        m \cdot n < 100
        \]
        Analyzing possible values of \( m \) and corresponding \( n \):
        - If \( m = 3 \), then \( n \) can be \( 5, 9, 15, 17, 19, 21, 23, 25, 27, 29, 31 \) yielding \( mn \) values: \( 15, 27, 45, 51, 57, 63, 69, 75, 81, 87, 93 \).
        - If \( m = 5 \), then \( n \) can be \( 9, 15, 17, 19 \) yielding \( mn \) values: \( 45, 75, 85, 95 \).
        - Higher \( m \) values result in \( mn \geq 135 \), which is invalid.
        
        There are 15 sets in this case.

    - **Case 2: \( S = \{7q, 11, 13, m, n\} \) where \( q \) is an odd integer \( > 1 \)**:
        - Here, 
        \[
        N = 7q \cdot 11 \cdot 13 \cdot m \cdot n 
        \]
        So, \( t = mnq \). This constraint is:
        \[
        mnq \leq 99
        \]
        - If \( q = 3 \), then \( mn \leq 33 \). Possible \( mn \) pairs:
            - If \( m = 3 \), \( n = 5, 9 \), giving 2 potential sets.
        - If \( q = 5 \), then \( mn \leq \frac{99}{5} = 19 \), 
            - Only \( m = 3 \) and \( n = 5 \) satisfy this condition.
        - \( q \geq 7 \) is invalid.
        
        There are 3 sets in this case.

    - **Case 3: \( S = \{7, 11q, 13, m, n\} \)**, similar constraints apply:
        - If \( q = 3 \), \( mn \leq 33 \) with possibilities:
            - If \( m = 3 \), \( n = 5, 9 \), 2 sets are valid.
        - If \( q = 5 \), only valid option is \( m = 3 \) and \( n = 5 \).
        - \( q \geq 7 \) is invalid.

        There are 3 sets in this case.

    - **Case 4: \( S = \{7, 11, 13 q, m, n\} \)**:
        - If \( q = 3 \), valid sets: \( S = \{3,5,7,11,13\} \) with 2 possible sets.
        - If \( q = 5 \), possibility: \( S = \{3,5,7,11,13\} \), only 1 set.
        - \( q \geq 7 \) is invalid.

        There are 3 sets in this case.

    - **Cases 5 and beyond: Combinations involving \( q, r (>1) \equiv \text{odd}\)**:
        - Trying combinations like \( \{7q, 11r, 13, m, n\} \) converge to the realization that the product \( mnr \leq 99 \) can’t hold up with distinct odd integers under errors faced previously.
        
    - **Case 6: \( S = \{77, 13, m, n, l\} \)**:
        - Recognize that \( 77 = 7 \cdot 11 \) and set \( mn \leq 99 \) but \( mnr \geq 105 \). No favorable solutions detected.

Consolidate final conclusion:
There are a total of \( 15 + 3 + 3 + 3 = 24 \) possible sets with distinct odd positive integers greater than \(2\) each.

\[
\boxed{24}
\]

---

## Entry 3

**ID**: 2

### Problem

Given real numbers \( a, b, c \) and a positive number \( \lambda \) such that the polynomial \( f(x) = x^3 + a x^2 + b x + c \) has three real roots \( x_1, x_2, x_3 \), and the conditions \( x_2 - x_1 = \lambda \) and \( x_3 > \frac{1}{2}(x_1 + x_2) \) are satisfied, find the maximum value of \( \frac{2 a^3 + 27 c - 9 a b}{\lambda^3} \).

### Solution


We begin by analyzing the function \( f(x) = x^3 + a x^2 + b x + c \), which has three real roots \( x_1, x_2, x_3 \). We are given the following conditions:
1. \( x_2 - x_1 = \lambda \)
2. \( x_3 > \frac{1}{2} (x_1 + x_2) \)

We aim to find the maximum value of \( \frac{2a^3 + 27c - 9ab}{\lambda^3} \).

1. **Transform the polynomial to remove the quadratic term:**
   Substitute \( x = y - \frac{a}{3} \) into \( f(x) \):
   \[
   \begin{aligned}
   F(y) & = f\left(y - \frac{a}{3}\right) \\
        & = \left(y - \frac{a}{3}\right)^3 + a \left(y - \frac{a}{3}\right)^2 + b \left(y - \frac{a}{3}\right) + c \\
        & = y^3 - \left(\frac{a^2}{3} - b\right)y + \frac{1}{27}(2a^3 + 27c - 9ab).
   \end{aligned}
   \]

2. **Identify the new roots of \( F(y) \):**
   Let the roots of \( F(y) \) be \( y_1, y_2, y_3 \). We know \( y_i = x_i + \frac{a}{3} \) for \( i = 1, 2, 3 \). Using Vieta's formulas:
   \[
   y_1 + y_2 + y_3 = 0 
   \]
   and 
   \[
   y_1 y_2 y_3 = -\frac{1}{27}(2a^3 + 27c - 9ab).
   \]

3. **Utilize the conditions provided:**
   Using \( x_2 - x_1 = \lambda \):
   \[
   y_2 - y_1 = \left(x_2 + \frac{a}{3}\right) - \left(x_1 + \frac{a}{3}\right) = x_2 - x_1 = \lambda.
   \]
   And for \( x_3 \):
   \[
   y_3 = x_3 + \frac{a}{3} > \frac{1}{2}\left(x_1 + x_2\right) + \frac{a}{3} = \frac{1}{2}\left(y_1 + y_2\right) = -\frac{y_3}{2}.
   \]
   Thus,
   \[
   y_3 > 0.
   \]

4. **Express \( y_1 \) and \( y_2 \) in terms of \( y_3 \) and \( \lambda \):**
   From the conditions:
   \[
   \begin{cases}
   y_1 + y_2 + y_3 = 0, \\
   y_2 - y_1 = \lambda,
   \end{cases}
   \]
   we solve:
   \[
   \begin{cases}
   y_1 = -\frac{1}{2}(y_3 + \lambda), \\
   y_2 = -\frac{1}{2}(y_3 - \lambda).
   \end{cases}
   \]

5. **Calculate \( \frac{2a^3 + 27c - 9ab}{\lambda^3} \):**
   \[
   \frac{2a^3 + 27c - 9ab}{\lambda^3} = -\frac{27 y_1 y_2 y_3}{\lambda^3}.
   \]
   Substituting \( y_1 \) and \( y_2 \):
   \[
   y_1 y_2 = \left(-\frac{1}{2}(y_3 + \lambda)\right) \left(-\frac{1}{2}(y_3 - \lambda)\right) = \frac{1}{4}(y_3^2 - \lambda^2).
   \]
   Thus,
   \[
   \frac{2a^3 + 27c - 9ab}{\lambda^3} = -\frac{27}{4} \cdot \frac{y_3^3 - y_3 \lambda^2}{\lambda^3} = -\frac{27}{4} \left(\frac{y_3}{\lambda}^3 - \frac{y_3}{\lambda} \right).
   \]

6. **Define \( z = \frac{y_3}{\lambda} \):**
   Then the expression becomes:
   \[
   -\frac{27}{4} \left(z^3 - z\right).
   \]

7. **Maximize \( g(z) = z^3 - z \) for \( z > 0 \):**
   \[
   g'(z) = 3z^2 - 1 \quad \text{and setting} \quad g'(z) = 0 \quad \text{gives} \quad z = \frac{1}{\sqrt{3}}.
   \]
   The function \( g(z) \) is strictly decreasing for \( z > \frac{1}{\sqrt{3}} \) and strictly increasing for \( 0 < z < \frac{1}{\sqrt{3}} \). Hence, the minimum value of \( g(z) \) is attained at \( z = \frac{1}{\sqrt{3}} \):
   \[
   g\left(\frac{1}{\sqrt{3}}\right) = \left(\frac{1}{\sqrt{3}}\right)^3 - \frac{1}{\sqrt{3}} = -\frac{2\sqrt{3}}{9}.
   \]

8. **Compute the maximum value of the original expression:**
   \[
   \frac{2a^3 + 27c - 9ab}{\lambda^3} = -\frac{27}{4} \left(-\frac{2\sqrt{3}}{9}\right) = \frac{27 \times 2 \sqrt{3}}{4 \times 9} = \frac{3\sqrt{3}}{2}.
   \]

Conclusion:
\[
\boxed{\frac{3\sqrt{3}}{2}}
\]

---

## Entry 4

**ID**: 3

### Problem

In triangle $ABC$, $CA = CB$, and $D$ is the midpoint of $AB$. Line $EF$ passes through point $D$ such that triangles $ABC$ and $EFC$ share the same incenter. Prove that $DE \cdot DF = DA^2$.

### Solution

1. **Identify Key Elements**: Consider \( \triangle ABC \) where \( CA = CB \) and \( D \) is the midpoint of \( AB \). Line \( \mathrm{EF} \) passes through \( D \) such that \( \triangle ABC \) and \( \triangle \mathrm{EFC} \) share the same incircle (inscribed circle).

2. **Given Common Incircle**: Let the shared incircle of \( \triangle ABC \) and \( \triangle \mathrm{EFC} \) be denoted by \( I \). 

3. **Excenter Involvement**: Assume that \( J \) is the \( C \)-excenter of \( \triangle ABC \). Thus, point \( J \) is also the \( C \)-excenter of \( \triangle \mathrm{EFC} \) because these two triangles share the same incircle.

4. **Properties of Excenters and Circumcenters**:
    - \( C, D, I, \) and \( J \) are collinear since \( D \) is the midpoint of \( AB \) and \( I \) is the incenter.
    - Because \( J \) is the excenter for both triangles at vertex \( C \), it lies on line \( \mathrm{CI} \).
    
5. **Midpoint Calculations**: Let \( \mathrm{K} \) be the midpoint of \( \mathrm{IJ} \). By properties of excircles and incenters in isosceles triangles:
    \[
    \mathrm{KI} = \mathrm{KJ} = K \text{A} = K \text{B} = K \text{E} = K \text{F}
    \]
    This relationship shows that all these points are equidistant to \( K \).

6. **Cyclic Hexagon Formation**: Hence, the points \( A, E, I, B, F, J \) form a cyclic hexagon because they lie on a common circle (i.e., they are concyclic).

7. **Power of a Point (for Point \( D \))**:
    - We apply the Power of a Point theorem at \( D \) for this cyclic hexagon:
    \[
    DE \cdot DF = DA \cdot DB
    \]
    Since \( D \) is the midpoint of \( AB \):
    \[
    DA = DB
    \]

8. **Simplification**: Thus, we have:
    \[
    DE \cdot DF = DA \cdot DA = DA^2
    \]

### Conclusion
\boxed{DE \cdot DF = DA^2}

---

## Entry 5

**ID**: 4

### Problem

Let \( p = 2^{3009}, q = 3^{2006}, \) and \( r = 5^{1003} \). Which of the following statements is true?
(A) \( p < q < r \)
(B) \( p < r < q \)
(C) \( q < p < r \)
(D) \( r < p < q \)
(E) \( q < r < p \)

### Solution

Given the values:
\[ p = 2^{3009}, \quad q = 3^{2006}, \quad r = 5^{1003} \]

1. Express \( p \) and \( q \) in terms of powers of the same base:
   \[ p = 2^{3009} = 2^{3 \times 1003} = (2^3)^{1003} = 8^{1003} \]
   \[ q = 3^{2006} = 3^{2 \times 1003} = (3^2)^{1003} = 9^{1003} \]
   Note: \( r = 5^{1003} \) is already expressed as a power.

2. Compare the magnitudes:
   To compare \( p, q, \) and \( r \), we now look at the factors 8, 9, and 5 which are:
   \[ 8^{1003}, \quad 9^{1003}, \quad \text{and} \quad 5^{1003} \]

3. Analyze the base values:
   - Base 8: \( 8 = 2^3 \)
   - Base 9: \( 9 = 3^2 \)
   - Base 5: \( 5 \)

4. Since the common exponent in all terms is \( 1003 \), we compare the bases:
   - Clearly, \( 5 < 8 < 9 \)

5. Therefore, since all exponents are equal, the order of the original terms follows the order of their bases:
   \[ 5^{1003} < 8^{1003} < 9^{1003} \]
   \[ r < p < q \]

### Conclusion:
Thus, the true statements is \( r < p < q \).

\[
\boxed{D}
\]

---

## Entry 6

**ID**: 5

### Problem

When \( a < -1 \), the nature of the roots for the equation
$$
\left(a^{3}+1\right) x^{2}+\left(a^{2}+1\right) x-(a+1)=0
$$
is:
(A) Two negative roots.
(B) One positive root and one negative root, with the absolute value of the negative root being larger.
(C) One positive root and one negative root, with the absolute value of the negative root being smaller.
(D) No real roots.

### Solution


Given the quadratic equation:

\[
(a^3 + 1) x^2 + (a^2 + 1) x - (a + 1) = 0
\]

where \( a < -1 \).

1. **Identify the coefficients and analyze their signs:**

    - \( a^3 + 1 \)
    - \( a^2 + 1 \)
    - \( -(a+1) \)

2. **Signs of the coefficients under the condition \( a < -1 \):**

    - Since \( a < -1 \), we have \( a^3 < -1 \), therefore \( a^3 + 1 < 0 \).
    - Given \( a^2 \geq 0 \) and \( a^2 + 1 \geq 1 \), it follows that \( a^2 + 1 > 0 \).
    - Since \( a < -1 \), then \( a + 1 < 0 \).

3. **Discriminant (\(\Delta\)) of the quadratic equation:**

    The discriminant \(\Delta\) of a quadratic equation \( Ax^2 + Bx + C = 0 \) is given by:
    \[
    \Delta = B^2 - 4AC
    \]

    Here \( A = a^3 + 1 \), \( B = a^2 + 1 \), and \( C = -(a + 1) \). Thus, the discriminant is:
    \[
    \Delta = (a^2 + 1)^2 - 4(a^3 + 1)(-(a + 1))
    \]

4. **Simplify the discriminant:**

    Substituting \( A \), \( B \), and \( C \) into the discriminant formula:
    \[
    \Delta = (a^2 + 1)^2 + 4(a^3 + 1)(a + 1)
    \]

    Since \( a^3 + 1 < 0 \) and \( a + 1 < 0 \), their product is positive. Consequently, the discriminant \(\Delta > 0\), indicating:

    \[
    \Delta > 0
    \]
    
    Therefore, the quadratic equation has two distinct real roots.

5. **Determine the nature of the roots using Vieta's formulas:**

    According to Vieta's formulas:
    - The sum of the roots \( x_1 \) and \( x_2 \) of the equation \( Ax^2 + Bx + C = 0 \) is given by: 
      \[
      x_1 + x_2 = -\frac{B}{A}
      \]
    - The product of the roots is given by:
      \[
      x_1 x_2 = \frac{C}{A}
      \]

    For our equation:
    \[
    x_1 + x_2 = -\frac{a^2 + 1}{a^3 + 1}
    \]
    \[
    x_1 x_2 = \frac{-(a + 1)}{a^3 + 1}
    \]

6. **Sign of the roots' product:**

    Given \( x_1 x_2 = \frac{-(a + 1)}{a^3 + 1} \) and because \( a + 1 < 0 \) and \( a^3 + 1 < 0 \):
    \[
    x_1 x_2 = \frac{-(a + 1)}{a^3 + 1} < 0
    \]

    This implies that the roots \( x_1 \) and \( x_2 \) have opposite signs.

7. **Sum of the roots:**

    Given \( x_1 + x_2 = -\frac{a^2 + 1}{a^3 + 1} \) with \(a^2 + 1 > 0 \) and \( a^3 + 1 < 0 \):
    \[
    x_1 + x_2 = -\frac{a^2 + 1}{a^3 + 1} > 0
    \]

    This implies that the negative root in absolute value is smaller than the positive root.

### Conclusion:

Thus, the correct answer is:
\[
\boxed{C}
\]

---

## Entry 7

**ID**: 6

### Problem

Suppose that $A, B, C, D$ are four points in the plane, and let $Q, R, S, T, U, V$ be the respective midpoints of $AB, AC, AD, BC, BD, CD$. If $QR = 2001$, $SU = 2002$, and $TV = 2003$, find the distance between the midpoints of $QU$ and $RV$.

### Solution

To find the distance between the midpoints of $Q U$ and $R V$, let's break down the given information and analyze each step.

1. **Identify the midpoints**:
    - $Q$, $R$, $S$, $T$, $U$, $V$ are midpoints of segments $AB$, $AC$, $AD$, $BC$, $BD$, $CD$ respectively.

2. **Recall Midpoint theorem**:
    - The Midpoint Theorem states that the segment connecting the midpoints of any two sides of a triangle is parallel to the third side and half as long.
    - Applying this theorem simplifies understanding the relationship and distances between these midpoints.

3. **Given distances**:
    - $Q R = 2001$
    - $S U = 2002$
    - $T V = 2003$

4. **Analyze quadrilateral $QUVR$**:
    - $Q R$ is parallel to $BC$ (because it's a segment joining midpoints of \(AB\) and \(AC\)).
    - $U V$ is also parallel to $BC$ (by a similar argument as $U$ and $V$ are midpoints of segments $BD$ and $CD$ respectively).
    - $Q U$ is parallel to $AD$, which is the segment joining midpoints of \(AB\) and \(BD\).
    - $R V$ is also parallel to $AD$ (as \(R\) and \(V\) are midpoints of \(AC\) and \(CD\)).

5. **Form of $QUVR$**:
    - The quadrilateral $QUVR$ is a parallelogram since opposite sides are both parallel and equal in length.

6. **Distance between midpoints $QU$ and $RV$**:
    - In a parallelogram, the midpoints of opposite sides form another parallelogram.
    - Since $QUVR$ is a parallelogram, the distance between the midpoints of $QU$ and $RV$ is the same as any side length of $QUVR$.
    - Hence, the distance between the midpoints of $QU$ and $RV$ is equal to $Q R$.

Conclusion: The side length $Q R$ is given as 2001. Therefore, the distance between the midpoints of $QU$ and $RV$ is
\[
\boxed{2001}
\]

---

## Entry 8

**ID**: 7

### Problem

Given that \(1 \leq x, y, z \leq 6\), how many cases are there in which the product of natural numbers \(x, y, z\) is divisible by 10?

### Solution

Given the constraints \(1 \leq x, y, z \leq 6\), we are to find the number of natural number combinations \((x, y, z)\) such that their product can be divided exactly by 10. 

To begin, we observe:
1. The total number of combinations of \(x, y, z\) is \(6^3\):
   \[
   6^3 = 216
   \]

2. To be divisible by 10, the product \(xyz\) must include both a factor of 2 and a factor of 5. Therefore, we need to evaluate the ways we can count combinations of \(x, y, z\) that miss these factors.

3. Consider the cases where \(x, y, z\) do not include factors of 2. The numbers within the range that are not multiples of 2 are \{1, 3, 5\}. The number of combinations in this case is:
   \[
   3^3 = 27
   \]

4. Next, consider the cases where \(x, y, z\) do not include factors of 5. The numbers within the range that are not multiples of 5 are \{1, 2, 3, 4, 6\}. The number of combinations in this case is:
   \[
   5^3 = 125
   \]

5. Now consider the cases where \(x, y, z\) include neither factors of 2 nor factors of 5. The only valid numbers in the range are then \{1, 3\}. The number of combinations in this case is:
   \[
   2^3 = 8
   \]

6. Using the principle of inclusion-exclusion, we can calculate the number of combinations where \(xyz\) is not divisible by 10:
   \[
   |A \cup B| = |A| + |B| - |A \cap B|
   \]
   where \(A\) is the set of combinations not including any multiples of 2 (27 in total) and \(B\) is the set not including any multiples of 5 (125 in total), and \(|A \cap B| = 8\):

   \[
   |A \cup B| = 27 + 125 - 8 = 144
   \]

7. Therefore, the number of combinations where \(xyz\) is divisible by 10 is:
   \[
   6^3 - |A \cup B| = 216 - 144 = 72
   \]

Conclusively, the number of natural number combinations \(x, y, z\) such that their product is divisible by 10 is:
\[
\boxed{72}
\]

---

## Entry 9

**ID**: 8

### Problem

Find all prime numbers \( p \) such that for any prime number \( q < p \), if \( p = kq + r \) with \( 0 \leq r < q \), then there does not exist an integer \( a > 1 \) such that \( a^2 \) divides \( r \).

### Solution


**Step 1:** Identify the prime numbers \( p \) that meet the given conditions: 

For every prime \( q < p \), \( p = k q + r \) where \( 0 \leq r < q \). It must hold that no integer \( a > 1 \) exists such that \( a^2 \mid r \).

**Step 2:** Check small prime numbers:

- For \( p = 2 \):
  - \( p = 2 \) and there are no prime \( q < 2 \). Hence, it trivially satisfies the condition.

- For \( p = 3 \):
  - \( p = 3 \) and the only prime \( q < 3 \) is 2.
  - \( 3 = 1 \cdot 2 + 1 \), where the remainder \( r = 1 \).
  - There is no integer \( a > 1 \) such that \( a^2 \mid 1 \).

- For \( p = 5 \):
  - Primes \( q < 5 \) are 2 and 3.
  - For \( q = 2 \), \( 5 = 2 \cdot 2 + 1 \), where the remainder \( r = 1 \).
  - For \( q = 3 \), \( 5 = 1 \cdot 3 + 2 \), where the remainder \( r = 2 \).
  - In both cases, there is no integer \( a > 1 \) such that \( a^2 \mid 1 \) or \( a^2 \mid 2 \).

- For \( p = 7 \):
  - Primes \( q < 7 \) are 2, 3, and 5.
  - \( 7 = 2 \cdot 3 + 1 \), and \( 7 = 1 \cdot 5 + 2 \).
  - The remainders are \( r = 1 \) and \( r = 2 \), and in both cases, there is no integer \( a > 1 \) such that \( a^2 \mid r \).

**Step 3:** Consider \( p = 11 \):

- Primes \( q < 11 \) are 2, 3, 5, 7.
  - For \( q = 2 \), \( 11 = 5 \cdot 2 + 1 \).
  - For \( q = 3 \), \( 11 = 3 \cdot 3 + 2 \).
  - For \( q = 5 \), \( 11 = 2 \cdot 5 + 1 \).
  - For \( q = 7 \), \( 11 = 1 \cdot 7 + 4 \).

  However, since \( 4 = 2^2 \), there is an integer \( a = 2 \) such that \( a^2 \mid 4 \). Thus, \( p = 11 \) does **not** satisfy the condition.

**Step 4:** Investigate \( p > 11 \):

- If \( p > 11 \), the remaining possible remainders \( r \) for \( p \) when divided by a prime \( q \) (where \( q < p \)) are:

  \[ r \neq 1, 2, 3, 5, 6, 7, 10, 11, 12 \]

  Now, considering \( p \) of the forms \( p - 4 \), \( p - 8 \), and \( p - 9 \):

  - Study \( p - 4 \):
    - \( p - 4 \) cannot include any prime factors greater than 3.
    - Suppose \( p - 4 = 3^a \geq 2 \).

  - Study \( p - 9 \):
    - \( p - 9 \) contains the prime factor 2 and possibly 7.
    - Let \( p - 9 = 3^a - 5 \).

**Step 5:** Detailed examination for valid \( p \):

- If \( p = 13 \):
  - Verify calculations:
    - \( p - 8 = 5 \), so \( p - 8 = 5^c \).

  - No contradictions and meets conditions for:
    - \( c = 1 \)

  Since:

  \[ p + 9 = 2 \cdot 7^0 + 1 \]
  
  Which meets given constraints:
    - \( p = 13 \).

**Conclusion:** 

The complete set of primes that satisfy the problem's conditions:

\[
\boxed{2, 3, 5, 7, 13}
\]

---

## Entry 10

**ID**: 9

### Problem

Find all functions \( f: \mathbb{R} \rightarrow \mathbb{R} \) that satisfy for all \( x, y \in \mathbb{R} \):

\[ 
f(y-f(x)) = f(x) - 2x + f(f(y)) 
\]

### Solution


Nous voulons trouver toutes les fonctions \( f: \mathbb{R} \rightarrow \mathbb{R} \) qui satisfont l'équation fonctionnelle suivante pour tous \( x, y \in \mathbb{R} \):

\[
f(y - f(x)) = f(x) - 2x + f(f(y))
\]

1. **Injectivité de \( f \)**:
    
    Pour prouver que \( f \) est injective, supposons que \( f(a) = f(b) \). Nous allons montrer que cela implique \( a = b \).

    Écrivons l'équation fonctionnelle pour \( x = a \) et \( x = b \):

    \[
    f(y - f(a)) = f(a) - 2a + f(f(y))
    \]

    \[
    f(y - f(b)) = f(b) - 2b + f(f(y))
    \]

    Puisque \( f(a) = f(b) \), nous avons:

    \[
    f(y - f(a)) = f(a) - 2a + f(f(y)) = f(b) - 2b + f(f(y)) = f(y - f(b))
    \]

    En comparant les deux équations, nous obtenons:

    \[
    f(a) - 2a = f(b) - 2b
    \]

    Donc,

    \[
    2a = 2b \quad \Rightarrow \quad a = b
    \]

    Donc, \( f \) est injective.

2. **Établir \( f(0) = 0 \)**:

    Supposons \( x = 0 \) et \( y = f(0) \) dans l'équation fonctionnelle:

    \[
    f(f(0) - f(0)) = f(0) - 2 \cdot 0 + f(f(f(0)))
    \]

    Simplifions cette expression:

    \[
    f(0) = f(0) + f(f(f(0)))
    \]

    \[
    f(f(f(0))) = 0
    \]

    Prenons maintenant \( x = f(f(0)) \) et \( y = 0 \):

    \[
    f(0 - f(f(f(0)))) = f(f(f(0))) - 2f(f(0)) + f(f(0))
    \]

    Nous savons que \( f(f(f(0))) = 0 \), donc ceci se simplifie à:

    \[
    f(0) = 0 - 2f(f(0)) + f(f(0))
    \]

    \[
    f(f(0)) = -2f(f(0))
    \]

    \[
    3f(f(0)) = 0 \quad \Rightarrow \quad f(f(0)) = 0
    \]

    Appliquons \( f \) des deux côtés:

    \[
    f(f(f(0))) = f(0)
    \]

    Comme nous avions déjà \( f(f(f(0))) = 0 \), nous obtenons:

    \[
    f(0) = 0
    \]

3. **Prouver \( f(y) = y \)**:
    
    Suppléons \( x = 0 \) dans l'équation initiale:

    \[
    f(y - f(0)) = f(0) - 2 \cdot 0 + f(f(y))
    \]

    Puisque \( f(0) = 0 \):

    \[
    f(y) = f(f(y))
    \]

    Par injectivité de \( f \), cela signifie que:

    \[
    y = f(y)
    \]

    Nous devons vérifier que \( f(y) = y \) est bien une solution de l'équation fonctionnelle:

    Remplaçons \( f(y) = y \) dans l'équation:

    \[
    f(y - f(x)) = f(x) - 2x + f(f(y))
    \]

    LHS:

    \[
    f(y - x) = y - x
    \]

    RHS:

    \[
    x - 2x + y = y - x
    \]

    Les deux côtés sont égaux, donc \( f(y) = y \) satisfait bien l'équation fonctionnelle.

**Conclusion**:
Les seules fonctions \( f: \mathbb{R} \rightarrow \mathbb{R} \) qui satisfont l'équation fonctionnelle donnée sont les fonctions d'identité.

\[
\boxed{f(y) = y}
\]

---

