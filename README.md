# q
***********************
ML basics
***********************

调参除了grid search还有其他什么自动的调参方法
Answer: random search, bayesian-optimization (when there are many parameters)
Reference:
https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f

Sampling methods:
Bernolli sampling, Reservoir sampling

2）what is p-value, 怎么计算
3）Gradient descent 解释原理，什么是 mini batch GD, stachastic GD, Adam
4）NN 里面 gradient descent怎么计算，是convex的吗，能保证最优解吗，（不能保证）怎么解决
5）Regression 用什么loss? Classification 用什么loss， 多分类呢？分类的loss是convex的吗
6）Bias & variance trade off
7）Random forest hyperparameter 怎么选
8）Validation set 都用来干嘛
9）classification data imbalance 怎么解决
10）CNN
11）介绍LSTM原理

https://www.1point3acres.com/bbs/thread-857873-1-1.html
(1) 如果改变了网页上的一些东西，比如字的颜色，怎么测试这个改变？
Follow up: null hypothesis 和alternative hypothesis分别是什么？
(2) 什么是p value? 如果p-value 是 0.04, 预先设定的threshold是 0.05, 你会得出什么结论？
(3) 如果testing结果远低于training performance，而且你确定不是overfitting，那是什么原因？
Followup: 你怎么判断training和testing distribution不同？
如何知道两个sample sets from same distribution (ks testing)
(4) 在logistic regression模型中，你有好几百个feature，怎么减少feature？
(5) 在amazon shopping中，设计一个ML系统给每个物品分类。注意每个物品可能多类别，比如运动服可以是clothing也可以是fitness类。怎么设计feature？怎么设计model？
(6) 如果在上面的模型中，有些类别数据特别少，怎么处理？用什么metrics evaluate？
(7) 什么是transfer learning? 
(8) 什么是overfitting? 什么是viarance bias trade-off? 怎么处理overfitting（分别描述neural networks 和 traditional ML (比如 regression)）?
Followup:  L1 and L2 regularization是‍‍‌‌‍‍‌‍‌‌‍‌‌‌‍‍什么?区别在哪里。
(9) Deep learning中用哪些activation functions? 什么时候用哪个? Tahn and ReLu区别是什么? 分别适合什么情况？
(10) 给你bag of words, 怎么vectorize?
(11) 描述k-means? 怎么选择K?

  (3) KL divergence？
（4）我答的是feature selection或者feature reduction. Feature selection举例了L1或者sklearn里的mutual information based selection，feature reduction比如PCA。
（5）我答的是multilabel classifier。我之前刚好有项目里用过。
（9）我讲了一下tanh，sigmoid和reLu。她其实想引导我回答tahn和sigmoid用于classification，reLu用在regression problem上，答到这句话就没再继续问了。你说的gradient vanishing的问题我也提了
（11）我是说先根据经验值选一个比较可能的范围，然后在training set上做grid search或者random search选最优的K

https://www.1point3acres.com/bbs/thread-814762-1-1.html
explain bagging and boosting, compare autoregressive and non-autoregressive models, explain CTC and RNN-T's loss computation and when to use which
explain self-attention mechanism, compare Conformer and Transformer, explain CTC and RNN-T, how to do ASR with a very large vocabulary set.

https://www.1point3acres.com/bbs/thread-796805-1-1.html

Which one/ones is convex optimization : linear regression, logistic regression, SVM, NN?
What's downside of decision tree, any other tree based model, what's GBM, how it works, what's sampling method of GBM? what hyperparam you know about GBM?

描述 logistic regression的loss是什么意思，和cross entropy联系，和MLE的联系

1. 分类和回归区别是什么，以及逻辑回归怎么做分类，再如果是多分类怎么办
2. 给一个句子做分类，如果不准使用神经网络，怎么做（tf-idf抽特征然后用贝叶斯/svm等）
4. 为什么lstm能解决梯度爆炸

t-test、covariance
confusion matrix related, when to use which metric, cross validation, reinforcement learning, supervised and unsupervised algorithms comparison
negative sampling何时使用
binary classification时imbalance dataset怎么办，metrics, PR, ROC
l1与sparsity的关系
Random forest

不平衡分类问题怎么解决，我说了一个cost sensitive，一个对负样本下采样然后ensemble，然后追问ensemble的基本原理，和boosting的区别，简单说rf和gdbt的区别，问了我很多xgboost的细节
不平衡问题选什么metric，为什么
聊regularization，我举了L0, L1, L2, L_infinity，让我说了首尾两个的定义，对比了一下中间两个，问我什么情况下用什么
dropout在inference的时候怎么实现
DL怎么调参，如果要调学习率怎么调，怎么防止很差的局部最优
Q-Learning的具体原理
difference between discriminative and generating models, examples, how they work, advantages

凸优化和非凸优化分别是什么，难点，如何解决
NLP的general knowledge, BERT的各种细节，self-attention的优劣等，pre-train language model有关的各种问题

SVM，核，为什么用核，你都知道怎样的核

what kind of optimization techniques do you use? apart from gradient descent?
the difference between sgd and adam
Design a simple text classification model.dataset is review sentiment(10k): 4 classes (50% neutral, 30% negative, 15% positive and 5% mixed)

discriminative 和generative model的区别, AutoEncoder以及VAE, GAN, Pre-trained model等等

几个基础的classifier
brainstorm features
large dataset上的一些算法

给了一个场景，大概意思就是说有一堆弱分类器，你不能对弱分类器和数据做修改，你改怎么改进？我一拍脑袋这不是boosting吗，然后一顿给说。之后问boosting了一些比较深的内容，比如Gradient boost 和xgboost的种种具体问题。
问了一些graphic model的问题，比如马尔科夫毯，马尔科夫链一些概念的问题

问自然语言模型里面，我们预测输出，也要加标点怎么办，因为nlp一般都是去掉标点，然后embedding。如果需要model怎么办。我也不知道，我就随便扯了一些，后来看了一些资料，有人说先去标点得到一个
模型，然后在加上更好的学习标点，我也不知道对不对。大家可以讨论下。我nlp做的很少。。
然后就是一些ml basic，什么是正则化，正则的意义是啥，他想问的就是l1 norm 说明weight属于laplace distribution，l2属于高斯。

bayesian和lr的区别，联系。batch dropout等基本知识点，如何做feature selection，pca等

***********************
ML implementation
***********************
1. 手写Logistic regression
2. 用numpy手写RNN神经网络
3. 用numpy写一下dense层，让我先写初始化w和b什么的
4. 手写CNN(卷积神经网络) 和怎么算梯度
5. LSTM解决梯度爆炸
6. normalization公式
7. 为什么transformer好，好在哪
8. 手写K-means
9. 手写一个deep learning training+testing的framework。我是基于pytorch写的。这个题就没什么好说的了，总之如果是做dl的话，应该比较顺利就能完成。最后会问到ba‍‍‍‌‍‌‍‌‌‍‌‍‌‌‍‌‍‌‌ckward()函数和optimizer.step()函数在底层分别做了什么。

************************
Coding
************************
1. 	String to Integer
2. 	Lc20, lc 27
3. 	找到全部最长的连续偶数或者奇数，输出list集合。
4. 	Max path sum
5. 	Design twitter
6. 	LRU Cache (lc 146)
7. 	Meeting room 1, meeting room 2
8. 	Lc 323, lc 735, lc 347
9. 	一个string是不是全都是distinct char
10.  给一个 list of list, 返回长度最长的list的第一个元素
11.  Hashmap是怎么实现的, 数据太多内存放不下怎么办
12.  如何实现heap
13.  设计一个迷宫游戏
14.  设计类似Linux的find命令，支持flexible的参数传入（getopts实现）
https://www.mkyong.com/java/search-directories-recursively-for-file-in-java/
15.  Count prime
16.  Java factory design, create interface, inheritance, override, overload, ood. Such as search file based on file name.
17.  Java gc collections, abstract class, interface,
18.  OOD设计思路
19.  Parking lot, 用到哪种数据结构，如何实现OOD
20.  Lucene internals, elasticsearch internals, inverted indexes
21.  Autocomplete system design
22.  tinyURL
