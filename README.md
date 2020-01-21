## hands on Visual Question Answering
env：python3.6 + pytorch1.0

1. motivation 
	- 熟悉CV与NLP领域常用模型：如cnn、rnn、tf etc.
	- 提升实践能力（复现论文）
2. work
	1. 自己从头写一遍VQA1的baseline：预处理、模型CNN+COW->concat->fc->softmax、评估(my multichoice_acc:0.4746)
	2. 参考[他人代码](https://github.com/Cyanogenoid/pytorch-vqa)学习复现[A Strong Baseline For Visual Question Answering](https://arxiv.xilesou.top/pdf/1704.03162.pdf)：resnet152+lstm+attn (my openended_acc:0.4986) 
