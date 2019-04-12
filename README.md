Imitation-Learning-for-Point-Process
===

Introduction
---
PPG (Point Process Generator) is a highly-customized RNN (Recurrent Neural Network) Model that would be able to produce actions (a point process) by imitating expert sequences. (**Shuang Li's ongoing work**)

How to Train a PPG
---
Before training a PPG, you have to organize and format the training data and test data into numpy arrays, which have shape (`num_seqs`, `max_num_actions`, `num_features`) and (`batch_size`, `max_num_actions`, `num_features`) respectively. Also you have to do paddings (zero values) for those sequences of actions whose length are less than `max_num_actions`. For the time being,
`num_features` has to be set as 1 (time).

Then you can initiate a session by tensorflow, and do the training process like following example:
```python
max_t        = 7
seq_len      = 10
batch_size   = 3
state_size   = 5
feature_size = 1
with tf.Session() as sess:
	# Substantiate a ppg object
	ppg = PointProcessGenerator(
		t_max=t_max,     # max time for all learner & expert actions
		seq_len=seq_len, # length for all learner & expert actions sequences
		batch_size=batch_size,
		state_size=state_size,
		feature_size=feature_size,
		iters=10, display_step=1, lr=1e-4)
	# Start training
	ppg.train(sess, input_data, test_data, pretrained=False)
```
You can also omit parameter `test_data`, which is set `None` by default, if you don't have test data for training.

The details of the training process will be logged into standard error stream. Below is testing log information.
```shell
2018-01-23 15:50:25.578574: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2018-01-23 15:50:25.578595: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2018-01-23 15:50:25.578601: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2018-01-23 15:50:25.578605: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
[2018-01-23T15:50:46.381850-05:00] Iter: 64
[2018-01-23T15:50:46.382798-05:00] Train Loss: 12.07759
[2018-01-23T15:51:01.708265-05:00] Iter: 128
[2018-01-23T15:51:01.708412-05:00] Train Loss: 10.53193
[2018-01-23T15:51:16.116164-05:00] Iter: 192
[2018-01-23T15:51:16.116308-05:00] Train Loss: 6.10489
[2018-01-23T15:51:26.816642-05:00] Iter: 256
[2018-01-23T15:51:26.816785-05:00] Train Loss: 3.17384
[2018-01-23T15:51:36.808152-05:00] Iter: 320
[2018-01-23T15:51:36.808301-05:00] Train Loss: 2.29386
[2018-01-23T15:51:46.169030-05:00] Iter: 384
[2018-01-23T15:51:46.169334-05:00] Train Loss: 1.71573
[2018-01-23T15:51:55.244403-05:00] Iter: 448
[2018-01-23T15:51:55.244538-05:00] Train Loss: 1.82563
[2018-01-23T15:52:04.491172-05:00] Iter: 512
[2018-01-23T15:52:04.491308-05:00] Train Loss: 2.59895
[2018-01-23T15:52:13.234096-05:00] Iter: 576
[2018-01-23T15:52:13.234250-05:00] Train Loss: 1.99338
[2018-01-23T15:52:21.904642-05:00] Iter: 640
[2018-01-23T15:52:21.904949-05:00] Train Loss: 1.20168
[2018-01-23T15:52:30.511283-05:00] Iter: 704
[2018-01-23T15:52:30.511429-05:00] Train Loss: 0.94646
[2018-01-23T15:52:39.296644-05:00] Iter: 768
[2018-01-23T15:52:39.296788-05:00] Train Loss: 0.88800
[2018-01-23T15:52:47.973522-05:00] Iter: 832
[2018-01-23T15:52:47.973675-05:00] Train Loss: 0.70098
[2018-01-23T15:52:56.207430-05:00] Iter: 896
[2018-01-23T15:52:56.207577-05:00] Train Loss: 0.68432
[2018-01-23T15:53:04.548818-05:00] Iter: 960
[2018-01-23T15:53:04.548964-05:00] Train Loss: 0.66598
[2018-01-23T15:53:04.549057-05:00] Optimization Finished!
```

How to Generate Actions
---
By simply running following code, fixed size (number and length) of sequences with indicated time frame will be generated automatically without input data. What needs to be noted is the length and the number of the generated sequence have been specified by the same input parameters when you initialize the `ppg` object.
```python
with tf.Session() as sess:

	# Here is the code for training a new ppg or loading an existed ppg

	# Generate actions
	actions, states_history = ppg.generate(sess, pretrained=False)
	print actions
```
Below are generated test actions.
```shell
(array([[ 0.63660634,  1.12912512,  0.39286253],
        [ 1.64375508,  1.60563707,  1.77609217],
        [ 3.08153439,  2.41127753,  2.59949875],
        [ 3.91807413,  3.74258327,  3.54215193],
        [ 4.97372961,  4.49850368,  4.98060131],
        [ 5.73539734,  5.15121365,  5.43891001],
        [ 6.24749708,  5.667624  ,  6.38705158],
        [ 6.60757065,  6.88907528,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ]], dtype=float32)
```

Experimental results
---

 Robbery in Atlanta           | Earthquake in North California
:----------------------------:|:----------------------------:
![](https://github.com/meowoodie/Imitation-Learning-for-Point-Process/blob/master/imgs/atl-robbery-1.gif)  |  ![](https://github.com/meowoodie/Imitation-Learning-for-Point-Process/blob/master/imgs/cal-earthquake-1.gif)

References
---
- [Shuang Li, Shuai Xiao, Shixiang Zhu, Nan Du, Yao Xie, Le Song. "Learning Temporal Point Processes via Reinforcement Learning
"](https://arxiv.org/abs/1811.05016)
