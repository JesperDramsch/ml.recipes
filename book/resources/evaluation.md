# Model Evaluation Resources

Here are additional resources that take evaluation further, beyond these notebooks:

{cite:t}`Raschka2018` focuses on the importance of proper model evaluation, selection, and algorithm selection in both academic and industrial machine learning research. The article covers common methods like the holdout data and introduces alternatives like bootstrapping for estimating performance uncertainty. Cross-validation techniques like leave-one-out and k-fold are discussed, along with tips for choosing the optimal value of k based on the bias-variance trade-off. The article also covers statistical tests for comparing algorithms and strategies for dealing with multiple comparisons, as well as, alternative methods for algorithm selection on small datasets.

{cite:t}`Dramsch2021` provides an introductory resource walking through different aspects of generalization and reproducibility of machine learning on real-world data.

Making sure results are solid and trustworthy when presented and published is a challenge in machine learning research.
Reproducibility, which entails getting comparable results with the same code and data, is essential for confirming the validity of research findings and encouraging open and accessible research.
NeurIPS launched a reproducibility program in 2019 to raise the bar for machine learning research. It consists of a code submission policy, a community-wide reproducibility challenge, and the use of a machine learning reproducibility checklist when submitting papers.
{cite:t}`Pinneau2020` aims to lower accidental errors and raise the bar for performing, sharing, and assessing machine learning research. 

{cite:t}`Lones2021` was originally developed for research students with a fundamental understanding of machine learning. The preprint lists typical pitfalls to avoid and provides remedies.
It covers five stages of the machine learning process: preparation before model building, reliable model building, robust model evaluation, fair model comparison, and results reporting. It focuses on issues relevant to academic research, such as the need for rigorous comparisons and valid conclusions. 

## Bibliography

```{bibliography}
:filter: docname in docnames
```