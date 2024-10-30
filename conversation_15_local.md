# Problem 15 - Solver: LOCAL

## Problem Statement

Adults made up $\frac5{12}$ of the crowd of people at a concert. After a bus carrying $50$ more people arrived, adults made up $\frac{11}{25}$ of the people at the concert. Find the minimum number of adults who could have been at the concert after the bus arrived.

## Dataset Solution

Let $x$ be the number of people at the party before the bus arrives. We know that $x\equiv 0\pmod {12}$, as $\frac{5}{12}$ of people at the party before the bus arrives are adults. Similarly, we know that $x + 50 \equiv 0 \pmod{25}$, as $\frac{11}{25}$ of the people at the party are adults after the bus arrives. $x + 50 \equiv 0 \pmod{25}$ can be reduced to $x \equiv 0 \pmod{25}$, and since we are looking for the minimum amount of people, $x$ is $300$. That means there are $350$ people at the party after the bus arrives, and thus there are $350 \cdot \frac{11}{25} = \boxed{154}$ adults at the party.
~eamo
Since at the beginning, adults make up $\frac{5}{12}$ of the concert, the amount of people must be a multiple of 12.
Call the amount of people in the beginning $x$.Then $x$ must be divisible by 12, in other words: $x$ must be a multiple of 12.
Since after 50 more people arrived, adults make up $\frac{11}{25}$ of the concert, $x+50$ is a multiple of 25. This means $x+50$ must be a multiple of 5.
Notice that if a number is divisible by 5, it must end with a 0 or 5. Since 5 is impossible (obviously, since multiples of 12 end in 2, 4, 6, 8, 0,...), $x$ must end in 0.
Notice that the multiples of 12 that end in 0 are: 60, 120, 180, etc.. By trying out, you can clearly see that $x=300$ is the minimum number of people at the concert.
So therefore, after 50 more people arrive, there are $300+50=350$ people at the concert, and the number of adults is $350*\frac{11}{25}=154$. Therefore the answer is $\boxed{154}$. 
I know this solution is kind of lame, but this is still pretty straightforward. This solution is very similar to the first one, though.
~hastapasta
Let $a$ be the number of adults before the bus arrived and $x$ be the total number of people at the concert. So, $\frac{a}{x}=\frac{5}{12}$. Solving for $x$ in terms of $a$, $x = \frac{12}{5}a$. After the bus arrives, let's say there are an additional $y$ adults out of the 50 more people who enter the concert. From that, we get $\frac{a+y}{x+50}=\frac{11}{25}$. Replacing $x$ with the value of $a$, the second equation becomes $\frac{a+y}{\frac{12}{5}a+50}=\frac{11}{25}$.
By cross-multiplying and simplifying, we get that $25(y-22)=\frac{7a}{5}$.
Observe that we must make sure $y-22$ is positive and divisible by $7$ to have an integer value of $a$. The smallest possible value of $y$ that satisfies this conditions is $29$. Plugging this into the equation, $a = 125$. The question asks for the minimum number of adults that are there after the bus arrives, which is $a+y$. Thus, the answer is simply $125+29=\boxed{154}$.
~mathical8

## Conversation History

### Round 1

#### Input Prompt
```
You are a mathematical problem solver. Your goal is to solve this problem:

Adults made up $\frac5{12}$ of the crowd of people at a concert. After a bus
carrying $50$ more people arrived, adults made up $\frac{11}{25}$ of the people
at the concert. Find the minimum number of adults who could have been at the
concert after the bus arrived.

Then solve the problem step by step, showing your work clearly. Make sure to: -
Explain your reasoning at each step - Show all calculations explicitly - Never
omit calculations for brevity - Highlight any key insights or clever
observations - If some calculations seem hard, think if there is a clever way
around it

Never ask for confirmation. Just provide your final answer as a number at the
end of your response prefixed with 'ANSWER: '.
```

#### Solver's Solution
```
Let's denote the initial number of people at the concert as \( x \). According
to the problem, the number of adults initially is \( \frac{5}{12}x \). After the
bus arrives, the total number of people at the concert becomes \( x + 50 \), and
the number of adults becomes \( \frac{5}{12}x + a \), where \( a \) is the
number of adults on the bus. We are given that after the bus arrives, adults
make up \( \frac{11}{25} \) of the people at the concert. Therefore, we can
write the equation:

\[ \frac{\frac{5}{12}x + a}{x + 50} = \frac{11}{25} \]

To eliminate the fractions, we cross-multiply:

\[ 25\left(\frac{5}{12}x + a\right) = 11(x + 50) \]

Expanding both sides, we get:

\[ \frac{125}{12}x + 25a = 11x + 550 \]

To clear the fraction, we multiply every term by 12:

\[ 125x + 300a = 132x + 6600 \]

Rearranging the terms to isolate \( x \) and \( a \), we get:

\[ 300a - 7x = 6600 \]

Solving for \( x \), we get:

\[ x = \frac{300a - 6600}{7} \]

For \( x \) to be an integer, \( 300a - 6600 \) must be divisible by 7. We can
check this by finding the remainder when 300 and 6600 are divided by 7:

\[ 300 \div 7 = 42 \text{ remainder } 6 \quad \Rightarrow \quad 300 \equiv 6
\pmod{7} \] \[ 6600 \div 7 = 942 \text{ remainder } 6 \quad \Rightarrow \quad
6600 \equiv 6 \pmod{7} \]

Therefore, we have:

\[ 300a - 6600 \equiv 6a - 6 \equiv 6(a - 1) \pmod{7} \]

For \( 6(a - 1) \) to be divisible by 7, \( a - 1 \) must be divisible by 7. Let
\( a - 1 = 7k \) for some integer \( k \). Then \( a = 7k + 1 \). Substituting
\( a = 7k + 1 \) into the equation for \( x \), we get:

\[ x = \frac{300(7k + 1) - 6600}{7} = \frac{2100k + 300 - 6600}{7} = \frac{2100k
- 6300}{7} = 300k - 900 \]

The number of adults after the bus arrives is:

\[ \frac{5}{12}x + a = \frac{5}{12}(300k - 900) + (7k + 1) = \frac{1500k -
4500}{12} + 7k + 1 = 125k - 375 + 7k + 1 = 132k - 374 \]

We need to find the minimum value of \( 132k - 374 \) such that it is a positive
integer. This means:

\[ 132k - 374 > 0 \quad \Rightarrow \quad 132k > 374 \quad \Rightarrow \quad k >
\frac{374}{132} \approx 2.83 \]

Since \( k \) is an integer, the smallest possible value for \( k \) is 3.
Substituting \( k = 3 \) into the equation for the number of adults after the
bus arrives, we get:

\[ 132(3) - 374 = 396 - 374 = 22 \]

Therefore, the minimum number of adults who could have been at the concert after
the bus arrived is:

\[ \boxed{22} \]
```

