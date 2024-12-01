---
layout: post
title: Poisson approximation
katex: True
---

We take a look at the Poisson distribution in this post. Continuing on similar lines as in the previous post, we derive the MGF of this distribution. Then we make the connection between the Poisson and the Binomial. This is a cute derivation from Casella and Berger. 

The Poisson distribution takes on the following form:

$$ 
\begin{aligned}
p(x;\lambda) = \frac {e^{-\lambda} \lambda^x}{x!}
\end{aligned}
$$

We can see that it is a product of a polynomial and exponential decay term, which are sort of competing influences. So while we expect the exponential to dominate at larger values of {{< katex >}}$\lambda${{< /katex >}}, the function is not monotonic, and will have a maximum, rising at first and then decaying to zero. 

The parameter {{<katex>}}$\lambda${{</katex>}} is called the rate. It is often used to model events that occur in periodic intervals, such as accidents or natural disasters like earthquakes. For instance, we could ascertain what the probability of there being 5 snow events in the winter season at a given place - say Montreal. The number would have a peak - say, 10. 1 event might be too low, and 100 might be too high. The parameter {{<katex>}}$\lambda${{</katex>}} determines the rate at which it snows. Naturally, we would expect it to be higher in Montreal than in New York owing to the former being generally much snowier. 

## Poisson MGF
The MGF is given by the expression 

$$ 
\begin{aligned}
M_X(t) = \sum_0^\infty \frac{ e^{tx} e^{-\lambda} \lambda^x}{x!}   = \sum_0^\infty \frac{ e^{-\lambda} e^{x(t + \log \lambda)}}{x!}
\end{aligned}
$$

To simplify this expression, we start with the reasoning that the Poisson distribution sums to unity, and massage the MGF on similar lines. 

$$
\begin{aligned}
\sum_0^\infty \frac{e^{-\lambda} \lambda^x}{x!} = 1 
\end{aligned}
$$ 

or 

$$
\begin{aligned}
\sum_0^\infty \frac{\lambda^x}{x!} = \sum_0^\infty \frac{ e^{x\log \lambda }}{x!}= e^\lambda
\end{aligned}
$$ 

Now, in the expression for the MGF above, let {{<katex>}}$\log u = t + \log \lambda$ {{</katex>}} so that {{<katex>}}$u = \lambda e^t$ {{</katex>}}. The terms in the MGF then become 

$$ 
\begin{aligned}
M_X(t) = \sum_0^\infty \frac{ e^{-\lambda} e^{x(t + \log \lambda)}}{x!} = \sum_0^\infty \frac{ e^{-\lambda} e^{x\log u}}{x!} = e^{-\lambda} e^u = e^{\lambda (e^t - 1)}
\end{aligned}
$$

## Poisson approximation
The binomial MGF is written as follows (see previous post):

$$ 
\begin{aligned}
M_Y(t) = [pe^t + (1-p)]^n 
\end{aligned}
$$

Now, the Poisson approximation says that when we set {{<katex>}}$\lambda = np$ {{</katex>}} with large {{<katex>}}$n$ {{</katex>}}, then if {{<katex>}}$M_X(t) \to M_Y(t)${{</katex>}}

$$ 
\begin{aligned}
P(X=x) \approx P(Y=y) 
\end{aligned}
$$

Or in other words, we can approximate the Poisson with the binomial with large {{<katex>}}$n$ {{</katex>}}. 

In our note here, we show that the MGFs are the same. The Poisson MGF is approximated by the large {{<katex>}}$n$ {{</katex>}} sequence implied by the binomial MGF. Once this is established, one can reason out from the convergence of MGFs theorem (not shown here) that the distributions are the same. 

To show that the MGFs are the same, manipulate the binomial MGF as follows 

$$ 
\begin{aligned}
[pe^t + (1-p)]^n &= [1+ \frac{np(e^t -1)}{n}]^n \\
&= [1+\frac{\lambda(e^t-1)}{n}]^n  
\end{aligned}
$$

At the limit of large {{<katex>}}$n${{</katex>}} we have 

$$ 
\begin{aligned}
\lim_{n \to \infty} [1+\frac{\lambda(e^t-1)}{n}]^n = e^{\lambda(e^t-1)}  
\end{aligned}
$$

We can thus see that the MGFs of the two distributions at the large {{<katex>}}$n${{</katex>}} limit are the same. One can then say that the distributions are also approximately the same from the convergence of MGFs theorem, which says that if one MGF converges to the other at large {{<katex>}}$n${{</katex>}}, then they also converge in distribution at {{<katex>}}$n\to\infty${{</katex>}}. 

In other words, we can approximate the Poisson with the binomial, with {{<katex>}}$\lambda = np$ {{</katex>}} with large {{<katex>}}$n${{</katex>}}. 

## References
1. Statistical Inference - Casella and Berger 