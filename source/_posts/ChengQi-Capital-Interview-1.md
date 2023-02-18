---
title: "CQ资产笔试题目"
date: "2020-09-07 22:24:33"
tags: ["Interview"]
---

CQ资产的笔试题目，笔试时间为40分钟，题量为9道。总体感觉笔试难度适中，时间有一点紧。由于笔试题目本身为英文，这里不做额外的翻译，解答为笔者所做，可能存在错误。

个人笔试结果为通过。



1. Given the following function definition: (4 min, score = 3)

   ```
   Define f(x): 
   	result = 1
   	For i from 1 to x (inclusively): 
   		result = result + i
   return result
   ```

   How many additions will take place while evaluating f(f(f(3)))?

*Solution:*
$$
\begin{aligned}
f(x) &= 1 + ( 1+ \ldots+x)
\\
&=1 + \frac{x(1+x)}{2}
\\
&= \frac{x^2 + x + 2}{2}
\end{aligned}
$$
Total additions:$3+f(3) + f(f(3)) = 3+7+29=39$.



2. A stock's price fluctuates every day by going up exactly 5% or going down exactly 5%. Assume that each direction is equally likely. Assume zero trading cost, zero interest rate, and no dividends and splits. What strategy is most likely to be profitable after 100 days?
   (3 min, score = 3)

   A. Buy or sell will produce same profitable

   B. Cannot know / no strategy can be profitable 

   C. Buy the stock

   D. Sell the stock

*Solution:*

D



3. Below is a list of asymptotic complexities of 8 functions, each with length N input:

   1. O(N^3)

   2. O(Log(N))

   3. O(Sqrt(N))

   4. O(N * log(N))

   5. O(2^N)

   6. O(N^N)

   7. O(N!)

   8. O(Log(Log(N)))

   Please sort the functions by order of growth, with slower growing functions first. (your answer shall be a sequence of letters, for example "BACDFHGE􏰀")
   (4 min, score = 3)

*Solution:*

HBDCAEGF



4. What is the maximum possible variance of a random variable taking values in the interval [0, 10]?
   (2 min, score = 2)

*Solution:*

Half is $0$, half is $10$.
$$
Var(x) = E[X-E(X)]^2 = 5^2 = 25
$$


5. How many integers $n$ such that $n^n$ is a perfect square are there in range [100: 400]?
   (5 min, score = 4)

*Solution:*

If $n$ is even, $n = 2k$, then $\sqrt{n^n} = \sqrt{n^{2k}} = n^k$, so it's a perfect square.

If $n$ is odd, $n = 2k+1$, then $\sqrt{n^n} = \sqrt{n^{2k+1}} = n^k\sqrt{n}$, it's a perfect square if and only if $n$ is a perfect square.

We have:
$$
10^2 = 100 \qquad 20^2 = 400
$$
So there are 5 odds that meet the condition, $11^2,13^2,\ldots,19^2$.

Totally, $151 + 5 = 156$ integers.



6. Assume there are three random variables $X, Y, Z$. All pairwise correlations are equal: $corr (x,y)=corr(y,z)=corr(x,z) = r$. What is the range of possible values of $r$? (list a range, like $[-0.3:0.7]$, for example)
   (6 min, score = 12)

*Solution:*
$$
\begin{aligned}
\left|\begin{array}{cccc} 
    1 &   r   & r \\ 
    r &   1   & r\\ 
    r &   r   & 1
\end{array}\right| 
&=
1 -r^2 + r(r^2-r) + r(r^2 - r)
\\
&= 2r^3 - 3r^2 + 1
\\
&= (r - 1)^2 (2r+1)\ge 0
\end{aligned}
$$
So the range of possible values of $r$ is $[-1/2, 1]$.



7. Assume there are three random variables $X, Y, Z$. We would like to use one number to describe their relations, just like the pairwise correlation of 2 variables $corr(x,y)$. We need the number to be normalized. Please list the possible mathematical formulas to calculate such number, the more the better.
   (6 min, score = 12)

*Solution:*
$$
\frac{corr(x,y) + corr(y, z) + corr(x, z)}{3}
$$


8. Triangle ABC has sides of length 45, 60 and 75. A point $X$ is placed randomly and uniformly inside the triangle. What is the expected value of the sum of perpendicular distance from point X to this triangle’s three sides?
   (9 min, score = 10)

*Solution:*
$$
45^2 + 60^2 = 75^2
$$
It's easy to know, triangle ABC is a right triangle.
$$
\begin{aligned}
E[l]&=15 * \frac{1}{6}\int_0^4\int_0^{(12-3x)/4} \left(x+y+\frac{12-3x-4y}{5}\right) dydx 
\\
&= 15 * \frac{1}{6}\int_0^4\int_0^{(12-3x)/4} \left(\frac{12}{5}+\frac{2x}{5}+\frac{y}{5}\right) dydx 
\\
&= 15 * \frac{1}{6}\int_0^4 \left(\frac{12+2x}{5} \frac{12-3x}{4}+\frac{1}{10}(\frac{12-3x}{4})^2\right)dx
\\
&= 15 * \frac{1}{6}\int_0^4 \left(\frac{3}{10}(24-2x-x^2)+\frac{1}{10}\frac{9}{16}(16-8x+x^2)\right)dx
\\
&= 15 * \frac{1}{6}\int_0^4 \left(\frac{81}{10} - \frac{21}{20}x -\frac{39}{160}x^2 \right)dx
\\
&= 15 * \frac{1}{6}\left(\frac{81}{10} * 4 - \frac{21}{20}*\frac{4^2}{2} -\frac{39}{160}*\frac{4^3}{3}\right)
\\
&= 15 * 1/6 * 18.8
\\
&= 47
\end{aligned}
$$


9. Please list one of your most "strange" or "crazy" idea to predict stock's return. You can assume you have all available public data and strong computing power. The answer shall be as "strange" as possible. 
   (6 min, score = 10)

*Solution:*

Use the stock code to select stocks. If the stock code is a prime number, buy it and hold. The strategy is pretty strange because there should be no useful information in the stock code.