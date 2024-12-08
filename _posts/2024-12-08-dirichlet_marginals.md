---
layout: post
title: Dirichlet Marginals
katex: True
---
Dirichlet distributions are conjugate priors for the multinomial distribution, and arise commonly in Bayesian modeling.In this post, I show that the marginal subvectors of a Dirichlet distribution also form Dirichlet distributions. More specifically, we can aggregate parameters from subvectors when we marginalize over them. 

That is to say that if we assume 

$$ \begin{align}
\theta_1, \theta_2, \cdots, \theta_n \sim Dirichlet(\alpha_1, \alpha_2, \cdots, \alpha_n)
\end{align} $$

then marginalizing over a subset $$ \theta_i, \theta_j $$ can be obtained by aggregating over the respective parameters $$ \alpha_i, \alpha_j $$:

$$ \begin{align}
\theta_1, \theta_2, \cdots, \theta_i + \theta_j, \cdots, \theta_n \sim Dirichlet(\alpha_1, \alpha_2, \cdots, \alpha_i + \alpha_j, \cdots, \alpha_n)
\end{align} $$

## Dirichlet as a conjugate prior of the multinomial distribution
The multinomial or categorical distribution can be written as follows:

$$ \begin{align}
p(y|\theta_1, \theta_2, \cdots, \theta_n) = \prod_{i=1}^n \theta_i^{y_i} 
\end{align} $$

This would be for a one-hot vector $y={y_1, y_2, \cdots, y_n}$ where $y_i \in {0, 1}$ and $\sum_{i=1}^n y_i = 1$. 

The posterior distribution can be written as follows:

$$ \begin{align}
p(\theta|y) \propto p(y|\theta) p(\theta)
\end{align} $$

with the prior $$ \theta \sim Dirichlet(\alpha_1, \alpha_2, \cdots, \alpha_n) $$ having the form:

$$\begin{align}
p(\theta|\alpha) \propto \Pi_{i=1}^n \theta_i^{\alpha_i - 1}
\end{align} $$

The posterior distribution is then

$$ \begin{align}
p(\theta|y) \propto \Pi_{i=1}^n \theta_i^{y_i + \alpha_i - 1}
\end{align} $$

which is also a Dirichlet distribution with parameters $y_i + \alpha_i$. 

We will return to this in another post. 

## Marginalization Constraints
We would first like to clarify the problem before we get into the machinery - admittedly, the machinery makes us happy, but it is easy to get lost in the details not knowing what we are doing. 

In the Dirichlet distribution written above, we would like to marginalize over some of the parameters. Consider the simplest case with three parameters:

$$ \begin{align}
\theta \sim Dirichlet(\alpha_1, \alpha_2, \alpha_3) \propto \theta_1^{\alpha_1 - 1} \theta_2^{\alpha_2 - 1} \theta_3^{\alpha_3 - 1}
\end{align} $$

with $\theta_1 + \theta_2 + \theta_3 = 1$. 

Our goal is to compute the marginal of the parameter $$\theta_3$$, so that we marginalize over $$\theta_1$$ and $$\theta_2$$. We can loosely pose the problem as integrating over parameters that are NOT $$\theta_3$$ - let us call them $$\theta_{rest}$$. 

$$ \begin{align} 
\int_{\theta_{rest}} \theta_1^{\alpha_1 - 1} \theta_2^{\alpha_2 - 1} \theta_3^{\alpha_3 - 1} d\theta_{rest}
\end{align} $$

However, we need to account for the fact that here $$ d\theta_{rest} \neq d\theta_1 d\theta_2$$. This is a constrained problem, and so integrating over $$\theta_1, \theta_2$$ also integrates over $$\theta_3$$ (!). 

The key - I rationalized, after much rumination - is that varying one parameter actually varies two. In this case, marginalization amounts to integrating over $$\theta_1$$, keeping $$\theta_1+\theta_2$$ constant. We are thus dropping the dimensionality of the simplex by one, but it is still the same problem. Formally, we can write:

$$ \begin{align} 
I = \int_{\theta_1} \theta_1^{\alpha_1 - 1} \theta_2^{\alpha_2 - 1} \theta_3^{\alpha_3 - 1} d\theta_1
\end{align} $$

with the constraint 

$$ \begin{align} 
\theta_1 + \theta_2 = c
\end{align} $$

Now that we have established the problem, it is quite easy to go over the machinery, which we do so in the next section. 

## Dirichlet Marginals 
Let us integrate over $$\theta_{rest}$$ to get the marginals. 

$$ \begin{align}
I &=& \int_{\theta_1} \theta_1^{\alpha_1 - 1} (c-\theta_1)^{\alpha_2 - 1} \theta_3^{\alpha_3 - 1} d\theta_1 \\
&=& \int_{\theta_1} c^{\alpha_1} (\theta_1/c)^{\alpha_1 - 1} [c(1-\theta_1/c)]^{\alpha_2 - 1} \theta_3^{\alpha_3 - 1} d(\theta_1/c) \\
&=& \int_{\theta_1/c} c^{\alpha_1 + \alpha_2 -1} \theta_3^{\alpha_3-1}(\theta_1/c)^{\alpha_1 - 1} (1-\theta_1/c)^{\alpha_2-1} d(\theta_1/c) \\
&=& c^{\alpha_1 + \alpha_2 -1} \theta_3^{\alpha_3-1} \int_t t^{\alpha_1 - 1} (1-t)^{\alpha_2 - 1} dt \\
&\propto& (\theta_1+\theta_2)^{\alpha_1 + \alpha_2 -1} \theta_3^{\alpha_3-1} \\
&=& \theta_3^{\alpha_3-1} (1-\theta_3)^{\alpha_0-\alpha_3-1}
\end{align} $$

where $$\alpha_0 = \alpha_1 + \alpha_2 + \alpha_3$$.

Or in other words we see that the marginal can be obtained as the Dirichlet distribution formed by gathering the parameters $$\theta_{rest}=\theta_1+\theta_2$$ and $$\theta_3$$ (we can normalize it to get the actual Dirichlet).

$$ \begin{align} 
I = Dirichlet(\alpha_1 + \alpha_2, \alpha_3)
\end{align}
$$ 

In this case with three parameters, it reduces to the beta distribution, of which the Dirichlet is a multivariate extension. 

$$ \begin{align} 
I = Beta(\alpha_3, \alpha_0-\alpha_3)
\end{align} $$

## Extending to more parameters
In order to extend to more than 3 parameters, we can build on the result above. Gather parameters of interest (over which we want to marginalize) sequentially. Again, start by gathering 2 groups $$ \theta_1, \theta_2 $$ and $$ \theta_3, \theta_4, \cdots $$. We can prove that marginalizing over $$\theta_1, \theta_2$$ gives 

$$ \begin{align} 
p(\theta_3, \theta_4, \cdots, \theta_n) = \theta_3^{\alpha_3} \theta_4^{\alpha_4} \cdots \theta_n^{\alpha_n}  (\theta_1+\theta_2)^{\alpha_1+\alpha_2}
\end{align} $$

We can group more variables by continuing in this vein. This is equivalent to the stick breaking construction of the Dirichlet distribution, wherein we start with lets say three variables, and break up the internal bits into smaller pieces. 

