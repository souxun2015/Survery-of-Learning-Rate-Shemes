
# Survery-of-Learning-rate-shemes

## Introduction
This project mainly introduces the learning rate schemes provided by tensorflow and observes their influences on convolutional neural networks. The problem about how they work is not included as it is difficult to explain. Maybe in the future, I will post it once I get them straight. So, there are 15 learning rate schemes we will talk about:
- 1. exponential_decay
- 2. piecewise_constant_decay
- 3. polynominal_decay
- 4. inverse_time_decay
- 5. cosine_decay
- 6. cosine_decay_restarts
- 7. linear_cosine_decay
- 8. noisy_linear_cosine_decay
- 9. tf.train.AdadeletaOptimizer
- 10. tf.train.AdagradOptimizer
- 11. tf.train.MomentumOptimizer
- 12. tf.train.AdamOptimizer
- 13. tf.train.FtrlOptimizer
- 14. tf.train.RMSPropOptimizer
- 15. AMSGradOptimizer
We conduct experiments on Cifar10 with these shemes, and then make analyses on different combinations among them.

## Comparable Analyses
- 1. 
