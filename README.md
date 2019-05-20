# Portfolio Optimization
Implementation & modification of a set of papers of portfolio optimization problem. 

Course project for **OIT 604 Spring19** of Graduate School of Business@Stanford.

The codes are on the basis of Yihao Kao's [RPCA code](http://www.yhkao.com/RPCA-code.zip).

### Problem Formulation

Consider an optimization problem:
$$
\max_{\mathbf{z \in \mathbb{R}^p}} \mathbf{c^Tz -(z^Ty)^2}
$$
Here $\mathbf{y} \in \mathbb{R}^p​$ could be considered as centered stock returns, $\mathbf{z}\in\mathbb{R}^p​$ are stock holding decisions for $p​$ stocks and $\mathbf{c} \in \mathbb{R}^p​$ are combinations of expected stock returns and other costs. The term $\mathbf{(z^Ty)^2}​$ is the total risk and the term $\mathbf{c^Tz}​$ is the expected revenue. By maximizing the weighted sum (the weight could be merged into $\mathbf{c}​$, so it is ignored), we can get a decision $\mathbf{z}​$.

With history data $\{\mathbf{y_1, y_2,\dots,y_N}\}$, the problem could be solved via efficient data driven decision making approaches such as PPCA <sup>1</sup>, RPCA<sup>2</sup>and DPCA<sup>3</sup>. These methods could be considered  as sample average approximation (SAA) approaches.

Furthermore, inspired by kernel methods<sup>456</sup>, suppose we can access a set of additional history feature data $\{\mathbf{x_1, x_2,\dots,x_N}\}$ along with $\mathbf{y}$s, given a new feature observation $\mathbf{x}​$, we try to improve the out-of-sample performance with additional information.

### References

1. Tipping, M. E. and Bishop, C. M. (1999), Probabilistic Principal Component Analysis. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 61: 611-622. ([link](https://doi.org/10.1111/1467-9868.00196))

2. Kao, Y.H. and Van Roy, B., 2013. Learning a Factor Model via Regularized PCA. Machine Learning, Volume 91, Number 3, pp. 279-303. ([link](https://doi.org/10.1007/s10994-013-5345-8))
3. Kao, Y.H. and Van Roy, B., 2014. Directed principal component analysis. *Operations Research*, *62*(4), pp.957-972. ([link](https://pubsonline.informs.org/doi/abs/10.1287/opre.2014.1290))
4. Bertsimas, D. and Kallus, N., 2014. From predictive to prescriptive analytics. *arXiv preprint arXiv:1402.5481*. ([link](https://arxiv.org/abs/1402.5481))
5. Nadaraya, Elizbar., 1964. On estimating regression. Theory Probab. Appl. 9(1) 141-142.
6. Watson, Geoffery., 1964. Smooth regression analysis. Sankhya A 359-372.



