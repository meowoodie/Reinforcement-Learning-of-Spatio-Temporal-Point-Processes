Reinforcement Learning of Spatio-Temporal Point Processes
===

Introduction
---
Spatio-temporal event data is ubiquitous in various applications, such as social media, crime events, and electronic health records. Spatio-temporal point processes offer a versatile framework for modeling such event data, as it can jointly capture spatial and temporal dependency. This repository provides a general framework for learning complex spatio-temporal dependencies using two different learning strategies: *Maximum Likelihood Estimate* and *Reinforcement learning*. See details at [reference paper](https://arxiv.org/abs/1906.05467). 

Usage
---
In this repository, there are two kinds of spatio-temporal point process trainer: *MLE_Hawkes_Generator* defined in `ppgmle.py` and *RL_Hawkes_Generator* defined in `ppgrl.py`, which are using two different learning frameworks, respectively. 

###### Construct and train a point process trainer
```Python
# load your dataset (n_sample, seq_len, 3)
data = np.load('../Spatio-Temporal-Point-Process-Simulator/data/rescale.ambulance.perday.npy')
data = data[:320, 1:51, :] # truncate your samples in case the sample size is too large and
                           # remove the first element in each seqs in case t = 0 for the first element of each sequence
# space limit
S    = [[-1., 1.], [-1., 1.]]
# time limit
T    = [0., 10.]
# normalize data to specific space-time region. 
da   = utils.DataAdapter(init_data=data, S=S, T=T)
seqs = da.normalize(data)

# training model
with tf.Session() as sess:
    batch_size = 32
    epoches    = 10
    layers     = [5]
    n_comp     = 5
    # define point process trainer
    ppg = MLE_Hawkes_Generator(
        T=T, S=S, layers=layers, n_comp=n_comp,
        batch_size=batch_size, data_dim=3, 
        keep_latest_k=None, lr=1e-1, reg_scale=0.)
    # training
    ppg.train(sess, epoches, seqs)
    # save parameters
    ppg.hawkes.save_params_npy(sess, 
        path="../Spatio-Temporal-Point-Process-Simulator/data/rescale_ambulance_mle_gaussian_mixture_params.npz")
```

###### Generate points from a well-trained point process generator

First you have to construct a `stppg` defined in `stppg.py` by loading the well-trained parameters.

```Python
params = np.load('../Spatio-Temporal-Point-Process-Simulator/data/rescale_ambulance_mle_gaussian_mixture_params.npz')
mu     = params['mu']
beta   = params['beta']
kernel = GaussianMixtureDiffusionKernel(
    n_comp=5, layers=[5], C=1., beta=beta, 
    SIGMA_SHIFT=.05, SIGMA_SCALE=.2, MU_SCALE=.01,
    Wss=params['Wss'], bss=params['bss'], Wphis=params['Wphis'])
lam    = HawkesLam(mu, kernel, maximum=1e+3)
pp     = SpatialTemporalPointProcess(lam)
```

Then generate points and visualize the intensity function as an animation.
```Python
# generate points
points, sizes = pp.generate(
    T=[0., 10.], S=[[-1., 1.], [-1., 1.]], 
    batch_size=500, verbose=True)

# plot intensity of the process over the time
plot_spatial_intensity(lam, points[0], S=[[0., 10.], [-1., 1.], [-1., 1.]],
    t_slots=1000, grid_size=50, interval=50)
```

Simulation Results
---

 | Linear spatial pattern | Nonlinear spatial pattern
:----------------------------:|:----------------------------:|:----------------------------:
Simulated parameters | ![](https://github.com/meowoodie/Reinforcement-Learning-of-Spatio-Temporal-Point-Processes/blob/master/imgs/kernel-svgau-a.png) | ![](https://github.com/meowoodie/Reinforcement-Learning-of-Spatio-Temporal-Point-Processes/blob/master/imgs/kernel-svgau-b.png)
Learned parameters | ![](https://github.com/meowoodie/Reinforcement-Learning-of-Spatio-Temporal-Point-Processes/blob/master/imgs/learned-kernel-svgau-a.png) | ![](https://github.com/meowoodie/Reinforcement-Learning-of-Spatio-Temporal-Point-Processes/blob/master/imgs/learned-kernel-svgau-b.png)

Numerical Results
---

We test our approach on two real datasets that contains complex spatial dependency. Such dependency is highly related to geographic features:

- **Atlanta 911 calls-for-service data**. The 911 calls-for-service data in Atlanta from the end of 2015 to the middle of 2017 is provided by the Atlanta Police Department.
- **Northern California seismic data**. The [Northern California Earthquake Data Center (NCEDC)](https://www.ncedc.org/index.html) provides public time series data that comes from boardband, short period, strong motion seismic sensors, and GPS, and other geophysical sensors.

To interpret the spatial dependency learned using our model, we visualize the progression of the conditional intensity through times on the map:

 Robbery in Atlanta           | Earthquake in North California
:----------------------------:|:----------------------------:
![](https://github.com/meowoodie/Imitation-Learning-for-Point-Process/blob/master/imgs/atl-robbery-1.gif)  |  ![](https://github.com/meowoodie/Imitation-Learning-for-Point-Process/blob/master/imgs/cal-earthquake-1.gif)

References
---
- [Spatio-Temporal Point Process Simulator (STPP)](https://github.com/meowoodie/Spatio-Temporal-Point-Process-Simulator)
- [Shixiang Zhu, Yao Xie. "Reinforcement Learning of Spatio-Temporal Point Processes."](https://arxiv.org/abs/1906.05467)
