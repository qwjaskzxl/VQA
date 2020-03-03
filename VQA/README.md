## hands on Visual Question Answering
env：python3.6 + pytorch1.0

1. motivation 
	- 熟悉CV与NLP领域常用模型：如cnn、rnn、tf etc.
	- 提升实践能力（复现论文）
2. work
	1. 自己从头写一遍VQA1的multichoice baseline（选1000个常见ans，其余的归为其他ans，在预测时，结果为others时可认为是正确or错误；看到另一个做法是先得到1000个常见ans，然后只取这些ans对应的q&v来用）：预处理、模型CNN+COW->concat->fc->softmax、评估(0.4746)
	2. 参考[他人代码](https://github.com/Cyanogenoid/pytorch-vqa)学习复现[A Strong Baseline For Visual Question Answering](https://arxiv.xilesou.top/pdf/1704.03162.pdf)：resnet152+lstm+attn (my openended_acc: ~0.60，后来我加了glove_300d到0.62+
### 资料
	1. [VQA2.0模型排行榜](https://paperswithcode.com/sota/visual-question-answering-on-vqa-v2)
 
