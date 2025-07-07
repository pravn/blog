---
layout: post
title: Marginals of the gaussian posterior
katex: True
---
*Unguent, powdered or liquid, troubled, confused, and drowned their sense in odours.* 

This is how I feel for the most part about developments in our field today. It is all fantastically, mindbogglingly complex, probably fun too if you are a practitioner. As a consumer, I find them useful. LLMs have made life much easier, an example demonstration of which point is this blog which wouldn't have been possible without an LLM assistant helping me with the ghastly details of filling out css sheets and reconciling versions of various dependencies. But softly now, let us get to the point.

In this post, we take a look at the Gaussian posterior distribution, and the marginalization machinery for the same. It is all properly laid out in the BDA book. This gives us the means to categorize data through their distributional parameters, quantify uncertainty and make draws from posterior distributions. 

## Formulation
Suppose that we are given some data $$ y = (y_1, y_2, \cdots, y_n) $$, from a gaussian distribution $$ p(y|\mu, \sigma^2)$$. We would like to get the marginal posterior distribution for the paramerers $$ p(\mu|y)$$ and $$ p(\sigma^2|y)$$. Naturally, we would also like to get the joint $$ p(\mu, \sigma^2|y) $$. 

The general idea is to start with the joint posterior (which we will compute next) and from there to marginalize. 

$$ \begin{align} 
p(\sigma^2|y) &= \int p(\mu, \sigma^2|y) d\mu \\
p(\mu|y) &= \int p(\mu, \sigma^2|y) d\sigma^2 \\
\end{align} $$

We first recapitulate the Gaussian posterior distribution, with two parameters $$\mu$$ and $$\sigma^2$$. Assume a non-informative prior $$ p(\mu, \sigma^2) \propto \sigma^{-2}$$.

$$ \begin{align}
    p(\mu, \sigma^2|y) &\propto& p(\mu, \sigma^2) \Pi_{i=1}^n p(y_i|\mu, \sigma^2) \\
    &=& \sigma^{-2} \Pi_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(y_i-\mu)^2}{2\sigma^2}} \\
    &=& \sigma^{-n-2} \exp(\sum_{i=1}^n -\frac{(y_i-\mu)^2}{2\sigma^2}) \\
\end{align} $$

We can simplify this by completing the square inside the quadrartic term $$ \sum (y-y_i)^2 $$. To do this, define the sample mean and variance.

$$ \begin{align}
\bar{y} &= \frac{1}{n} \sum_{i=1}^n y_i \\
s^2 &= \frac{1}{n-1} \sum_{i=1}^n (y_i-\bar{y})^2
\end{align} $$

After using the definitions for the sample mean and variance, the term inside the exponential becomes:

$$\begin{align}
\sum_{i=1}^n (y_i-\mu)^2 &= \sum_{i=1}^n [(y_i-\bar{y})+ (\bar{y} -\mu)]^2 \\
&= \sum_{i=1}^n (y_i-\bar{y})^2 + n (\bar{y}-\mu)^2 \\
&= (n-1)s^2 + n (\bar{y}-\mu)^2 
\end{align} $$

Note that the cross term containing $$ \sum (\bar{y}-\mu) (y_i-\bar{y}) $$ cancels out since $$n\bar{y}=\sum y_i$$.


With these manipulations, we write the joint posterior as follows:

<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
$$ \begin{align}
p(\mu, \sigma^2|y) &\propto& \sigma^{-n-2} \exp(-\frac{(n-1)s^2 + n (\bar{y}-\mu)^2}{2\sigma^2}) 
\end{align} $$
</div>

## Marginal posterior of variance
Let us now integrate the the above equation to marginalize over the mean $$\mu$$. 

$$\begin{align} 
p(\sigma^2|y) \propto \int \sigma^{-n-2} \exp(-\frac{(n-1)s^2 + n (\bar{y}-\mu)^2}{2\sigma^2}) d\mu
\end{align} $$

The dependence in $$\mu$$ can be seen as the integral over a gaussian distribution, and can therefore be factored out since $$ \int \exp(-(\bar{y}-\mu)^2/2\sigma^2) d\mu = \sqrt{2\pi\sigma^2}$$. 

The resulting expression, after some algebra, can then be written as follows. 

<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
$$ \begin{align}
p(\sigma^2|y) &\propto \sigma^{-(n+1)/2} \exp\left(-\frac{(n-1)s^2}{2\sigma^2}\right) \\
&= Inv_{\chi^2}(n-1, s^2)
\end{align} $$
</div>


## Marginal posterior of mean

$$ \begin{align}
p(\mu|y) &\propto \int p(\mu, \sigma^2|y) d\sigma^2 \\
&= \int \sigma^{-n-2} \exp\left(-\frac{(n-1)s^2 + n (\bar{y}-\mu)^2}{2\sigma^2}\right) d\sigma^2 \\
& = \int \sigma^{-n-2} \exp\left(-\frac{A}{2\sigma^2}\right) d\sigma^2 \\
\end{align} $$

where $$ A = (n-1)^2 + n(\bar{y}-\mu)^2 $$.

Here, we can see that we can make another substitution $$ z = \frac{A}{2\sigma^2} $$ to take the dependence of $$ A $$ outside the integral, which is then treated as a constant. 

$$ \begin{align}
p(\mu|y) &\propto A^{-n/2} \int_0^\infty z^{\frac{n-2}{2}}\exp(-z) dz 
&\propto [(n-1)s^2 + n(\mu-\bar{y})^2]^{-n/2} \\
\end{align} $$

Factor out $$(n-1)s^2$$ to manipulate in terms of the t-distribution of order $$n-1$$. 

<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
$$ \begin{align}
p(\mu|y) \propto [1 + \frac{n(\mu-\bar{y})^2}{(n-1)s^2}]^{-n/2} 
\end{align} $$
</div>


The posterior distribution of $$ \mu $$ under a non-informative prior on $$(\mu, log \sigma )$$ is the $$t_{n-1}(\bar{y}, s^2/n)$$ distribution.

$$ \begin{align}
\frac{\bar{y}-\mu}{s/\sqrt{n}}|y \sim t_{n-1}
\end{align} $$

## Conjugate Priors 
Instead of the non-informative prior, we can use a conjugate prior on $$\mu$$ and $$\sigma^2$$.

$$ \begin{align}
\mu &\sim N(\mu_0, \sigma_0^2/\kappa_0) \\
\sigma^2 &\sim Inv-\chi^2(\nu_0, \sigma_0^2)
\end{align} $$

so that we can write the joint prior as follows:

$$ \begin{align}
p(\mu, \sigma^2) &\propto& \sigma^{-1}( \sigma^2 )^{-(\nu_0/2+1)} \exp \left( -\frac{\nu_0\sigma_0^2+\kappa_0(\mu_0-\mu)^2}{2\sigma^2}\right ) 
\end{align} $$

## Posterior distributions - and the influence of prior vs data terms
When we combine the prior with the likelihood term spelled out earlier, we get the posterior $$p(\mu, \sigma^2|y) \propto p(\mu, \sigma^2) p(y|\mu, \sigma^2)$$.

$$ \begin{align}
p(\mu, \sigma^|y) &\propto \sigma^{-1}( \sigma^2 )^{-(\nu_0/2+1)} \exp \left( -\frac{\nu_0\sigma_0^2+\kappa_0(\mu_0-\mu)^2}{2\sigma^2}\right ) \\ 
& \times \sigma^{-n/2} \exp(-\frac{(n-1)s^2 + n (\bar{y}-\mu)^2}{2\sigma^2}) \\
&= N-Inv-\chi^2(\mu, \sigma^2|\mu_n, \sigma_n^2/\kappa_n; \nu_n, \sigma_n^2)
\end{align} $$

where 

$$ \begin{align}
\mu_n &= \frac{\kappa_0}{\kappa_0 + n} \mu_0 + \frac{n}{\kappa_0 + n} \bar{y} \\
\kappa_n &= \kappa_0 + n \\
\nu_n &= \nu_0 + n \\
\nu_n \sigma_n^2 &= \nu_0 \sigma_0^2 + (n-1)s^2 + \frac{\kappa_0 n}{\kappa_0+n} (\mu_0-\bar{y})^2
\end{align} $$

This posterior mean emerges as the weighted average of the prior mean and the data mean, and the variance term is the sum of prior variance, sample variance and the difference between the sample and prior mean. 

Next, we look at the conditional and marginal posteriors.
We notice that the conditional posterior is the weighted sum of the prior and data terms, and the influence of the data term increases as $$n$$ increases.

$$ \begin{align}
\mu|\sigma^2, y &\sim N(\mu_n, \sigma_n^2/\kappa_n) \\
& = N(\frac{\frac{\kappa_0}{\sigma^2}\mu_0 + \frac{n}{\sigma^2}\bar{y}}{\frac{\kappa_0}{\sigma^2} + \frac{n}{\sigma^2}}, \frac{1}{\frac{\kappa_0}{\sigma^2} + \frac{n}{\sigma^2}})
\end{align} $$

The marginal posterior also follows a similar pattern as before giving a t-distribution. 


<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
$$\begin{align}
p(\mu|y) &\propto \left( 1 + \frac{ \kappa_n(\mu-\mu_n)^2}{\nu_n\sigma_n^2} \right)^{-(\nu_n+1)/2}\\
&= t_{\nu_n}(\mu|\mu_n, \sigma_n^2/\kappa_n)
\end{align} $$
</div>


