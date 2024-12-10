---
layout: post
title: Dirichlet marginals part 2 (BDA problem 3.1)
katex: True
---
In our [previous](https://pravn.github.io/blog/2024/12/08/dirichlet_marginals.html) post, we went over general machinery for marginalizing over Dirichlet distributions. This time, we move on to its sequel. We examine problem 3.1 from the BDA3 book, which adds a twist to the marginalization exercise alluded to above. Given a multinomial (eh, categorial) distribution, we are asked to get the marginal posterior, but expressed in terms of a new, reparameterized set of parameters.

Consider the n-dimensional variable $$ y = (y_1, y_2, \cdots y_n) $$, following a multinomial distribution with parameters $$ \theta = (\theta_1, \theta_2, \cdots, \theta_n) $$. Assuming that the parameters $$ \theta $$ follow a Dirichlet prior distribution with parameters $$ \alpha_1, \alpha_2, \cdots, \alpha_n $$, we would like to obtain the marginal posterior of the variable $$ \alpha = \frac{\theta_1}{\theta_1+\theta_2} $$. 

## Posterior and marginal of subvector 
We first recall the definitions of the posterior and marginal subvectors from our [previous](https://pravn.github.io/blog/2024/12/08/dirichlet_marginals.html) post.

Posterior:

$$ \begin{align}
p(\theta|y) \propto \Pi_{i=1}^n \theta_i^{y_i + \alpha_i - 1}
\end{align} $$

Marginal of subvectors $$ \theta_1, \theta_2 $$:

$$ \begin{align}
p(\theta_1, \theta_2|y) &\propto& Dir(\alpha_1+y_1, \alpha_2+y_2, \alpha_0-\alpha_1-\alpha_2+1-y_1-y_2) \\
&\propto&  \theta_1^{\alpha_1+y_1-1} \theta_2^{\alpha_2+y_2-1} (1-\theta_1-\theta_2)^{\alpha_0-\alpha_1-\alpha_2 + 1-y_1-y_2}
\end{align} $$

where $$ \alpha_0 = \sum_{i=1}^n \alpha_i $$.

As we saw in the previous post, we can group parameters we are marginalizing over so that 

$$ p(\theta_1, \theta_2|y) = Dir(\theta_1+\alpha_1+y_1, \theta_2+\alpha_2+y_2, \theta_{rest}+\alpha_{rest}+y_{rest}) $$

where $$ \theta_{rest} = 1- \theta_1-\theta_2 $$ and $$ y_{rest} =  \sum_{i=1}^n y_i -y_1-y_2 $$.

## Reparameterization 
We need to reparameterize in terms of the new variable $$ \alpha = \frac{\theta_1}{\theta_1+\theta_2} $$ for the problem. Note though that we only have one variable $$\alpha$$ and so a second variable must be chosen. In the [solutions](http://www.stat.columbia.edu/~gelman/book/solutions3.pdf) to the BDA3 book, they choose $$ \beta = (\theta_1+\theta_2)$$. However, that choice is arbitrary. 

The reparameterization is as follows:

$$ \begin{align}
p(\alpha, \beta) = p(\theta_1, \theta_2) |J^{-1}|
\end{align} $$

where $$ J $$ is the Jacobian matrix of the transformation:

$$ \begin{align}
\alpha &= \frac{\theta_1}{\theta_1+\theta_2} \\
\beta &= \theta_1 + \theta_2
\end{align} $$

The Jacobian transformation of $\alpha, \beta$ with respect to $\theta_1, \theta_2$ is given by:

$$ \begin{align}
J &= \begin{vmatrix} 
\frac{\partial \alpha}{\partial \theta_1} & \frac{\partial \alpha}{\partial \theta_2} \\
\frac{\partial \beta}{\partial \theta_1} & \frac{\partial \beta}{\partial \theta_2} 
\end{vmatrix} \\
&= \begin{vmatrix} 
\frac{\theta_2}{(\theta_1+\theta_2)^2} & -\frac{\theta_1}{(\theta_1+\theta_2)^2} \\
1 & 1 
\end{vmatrix} \\
&= \frac{1}{\theta_1+\theta_2} \\
&= \frac{1}{\beta}
\end{align} $$

Before we finish writing the full transformation, let us also rewrite $$ \theta_1, \theta_2 $$ in terms of $$ \alpha $$ and $$ \beta $$:

$$ \begin{align}
\theta_1 &= \alpha\beta \\
\theta_2 &= \beta(1-\alpha)
\end{align} $$

## Reparameterized posterior
We can now write the full transformation: 

$$ \begin{align}
p(\alpha, \beta|y) &= (\alpha\beta)^{\alpha_1+y_1-1} \beta^{\alpha_2+y_2-1} (1-\alpha)^{\alpha_2+y_2-1} (1-\beta)^{\alpha_{rest}+y_{rest}  -1} \\
&= \alpha^{\alpha_1+y_1-1} \beta^{y_1+y_2+\alpha_1+\alpha_2-1} (1-\alpha)^{\alpha_2+y_2-1} (1-\beta)^{\alpha_{rest}+y_{rest}  -1} 
\end{align} $$

This is a product of Beta distributions with parameters $$ (y_1+\alpha_1, y_2+\alpha_2)$$ and $$ (y_1+y_2+\alpha_1+\alpha_2, y_{rest}+\alpha_{rest}) $$. 

$$ \begin{align}
p(\alpha, \beta|y) = Beta(\alpha|y_1+\alpha_1, y_2+\alpha_2)Beta(\beta|y_1+y_2+\alpha_1+\alpha_2, y_{rest} + \alpha_{rest})
\end{align} $$

## References

[1] Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). Bayesian Data Analysis, Third Edition. CRC Press.

[2] BDA3 solutions: [pdf](http://www.stat.columbia.edu/~gelman/book/solutions3.pdf)

[3] Dirichlet marginals: [link](https://pravn.github.io/blog/2024/12/08/dirichlet_marginals.html)

