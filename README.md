## homework_1
这次作业实现了词嵌入向量的预训练，使用的是Skip-Gram模型，并用Negative Sampling进行近似计算来降低计算成本，最后用UMAP降维并可视化了前500个词的空间关系；
本项目使用的数据集是Penn Tree Bank(PTB)数据集，该语料库取自《华尔街日报》的文章，已经被划分为训练集、验证集和测试集,使用的代码是在学习李沐《深度学习》代码的基础上对小bug进行了修正；
本项目最终训练好的词嵌入向量和对应于PTB语料库的词汇表数据被保存在文件vocab_and_embedding.pkl中；
本项目可视化词嵌入向量的代码放在了UMAP_visualize.py文件中，而关于模型构建和训练的细节则放在了NLP_Word2Vec.ipynb中。
## homework_3
这次作业是训练一个Transformer完成英法翻译任务，我选择的Transformer结构是在论文的基础上做了一点针对性修改：
* 相较于论文中的post-norm我使用的是目前更为常用的pre-norm；
* 针对本次作业的小训练集，我使用了较小的头数和堆叠数，降低模型的复杂度；
* 此外，我还使用了早停、weight decay这些正则化手段来防止过拟合；
* 由于最后的模型需要放在GPU上训练，考虑到GPU的显存，我在训练中使用自动混合精度来节省显存，并配合使用GradScalar来避免梯度下溢为0。
在本次作业中我采用的是**点积注意力**，设置的轮次是60个epoch，由于训练集相对比较小，所以设置了早停，忍耐度是5个epoch没有improve，但是直到60个epoch结束，也没有因为早停而结束训练，这表明模型应该还可以获得更好的性能，但是出于性能与成本的权衡，最终只训练了60个epoch，没有继续增加epoch，最后在验证集上评估得到的**最佳模型的BLEU是44.679**；考虑到GPU训练的费用和时间，我没有再采用加性注意力机制进行对比。
## homework_4
20 Newsgroups是一个非常经典，专门用来做文本分类的英文新闻文档数据集，它来源于Usenet新闻组帖子，包含约18,800篇新闻文档，一共有20个主题类别。本次作业选取的是该数据集中**关于无神论和基督教主题的新闻文档**来训练模型完成文本分类任务。

下面是我完成作业过程中记录的一些反思与总结：
* 使用BERT处理文本分类任务的时候，需要使用其配套的**Tokenizer**，它会自动处理词汇映射、padding和特殊符号的添加（如[CLS]，[SEP]）。
* 在使用fetch_20newsgroups下载20newsgroups的时候如果用远程服务器的网络一直下载失败，可以现在本地电脑上面把数据集下载，传到远程服务器的相应路径上面，这样远程服务器就不需要用网络下载数据集了。
* 运行代码的时候如果终端提示版本太低，需要更新 accelerate / 要装 transformers [torch]，一定要检查是否已经缓存了旧版模型文件（model.safetensors 等），如果缓存了，需要删掉从而让代码重新运行的时候重新加载新版模型文件；否则就会出现旧版模型参数名称跟新版模型参数名称对不上的情况，虽然无伤大雅，但是最好还是避免。

下面是先下载预训练的BERT模型参数，然后在20 Newsgroups训练集上进行微调，最后在20 Newsgroups测试集上进行评估：
tokenizer_config.json: 100%|████████████████████████████████████████████████| 48.0/48.0 [00:00<00:00, 193kB/s]
vocab.txt: 232kB [00:00, 524kB/s]
tokenizer.json: 466kB [00:00, 1.60MB/s]
config.json: 570B [00:00, 2.38MB/s]
model.safetensors: 100%|███████████████████████████████████████████████████| 440M/440M [00:37<00:00, 11.6MB/s]
Loading weights: 100%|████████████████████████████████████████████████████| 199/199 [00:00<00:00, 8348.42it/s]
BertForSequenceClassification LOAD REPORT from: bert-base-uncased
Key                                        | Status     | 
-------------------------------------------+------------+-
cls.predictions.transform.LayerNorm.weight | UNEXPECTED | 
cls.predictions.transform.dense.bias       | UNEXPECTED | 
cls.predictions.transform.LayerNorm.bias   | UNEXPECTED | 
cls.seq_relationship.bias                  | UNEXPECTED | 
cls.predictions.transform.dense.weight     | UNEXPECTED | 
cls.predictions.bias                       | UNEXPECTED | 
cls.seq_relationship.weight                | UNEXPECTED | 
classifier.weight                          | MISSING    | 
classifier.bias                            | MISSING    | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
`logging_dir` is deprecated and will be removed in v5.2. Please set `TENSORBOARD_LOGGING_DIR` instead.
开始微调模型...
{'loss': '0.734', 'grad_norm': '4.624', 'learning_rate': '4.5e-06', 'epoch': '0.1471'}                        
{'loss': '0.7061', 'grad_norm': '6.522', 'learning_rate': '9.5e-06', 'epoch': '0.2941'}                       
{'loss': '0.6698', 'grad_norm': '2.645', 'learning_rate': '1.45e-05', 'epoch': '0.4412'}                      
{'loss': '0.6351', 'grad_norm': '6.849', 'learning_rate': '1.95e-05', 'epoch': '0.5882'}                      
{'loss': '0.6063', 'grad_norm': '5.944', 'learning_rate': '2.45e-05', 'epoch': '0.7353'}                      
{'loss': '0.5484', 'grad_norm': '7.366', 'learning_rate': '2.95e-05', 'epoch': '0.8824'}                      
{'eval_loss': '0.4129', 'eval_accuracy': '0.8257', 'eval_f1': '0.8336', 'eval_precision': '0.8867', 'eval_recall': '0.7864', 'eval_runtime': '0.5384', 'eval_samples_per_second': '1332', 'eval_steps_per_second': '22.29', 'epoch': '1'}
Writing model shards: 100%|█████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.30it/s]
{'loss': '0.4562', 'grad_norm': '6.357', 'learning_rate': '3.45e-05', 'epoch': '1.029'}                       
{'loss': '0.3915', 'grad_norm': '4.034', 'learning_rate': '3.95e-05', 'epoch': '1.176'}                       
{'loss': '0.2938', 'grad_norm': '8.919', 'learning_rate': '4.45e-05', 'epoch': '1.324'}                       
{'loss': '0.3268', 'grad_norm': '3.521', 'learning_rate': '4.95e-05', 'epoch': '1.471'}                       
{'loss': '0.3859', 'grad_norm': '9.315', 'learning_rate': '4.567e-05', 'epoch': '1.618'}                      
{'loss': '0.3006', 'grad_norm': '4.104', 'learning_rate': '4.087e-05', 'epoch': '1.765'}                      
{'loss': '0.3137', 'grad_norm': '6.447', 'learning_rate': '3.606e-05', 'epoch': '1.912'}                      
{'eval_loss': '0.527', 'eval_accuracy': '0.7992', 'eval_f1': '0.8326', 'eval_precision': '0.7749', 'eval_recall': '0.8995', 'eval_runtime': '0.5382', 'eval_samples_per_second': '1332', 'eval_steps_per_second': '22.3', 'epoch': '2'}
Writing model shards: 100%|█████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.30it/s]
{'loss': '0.2373', 'grad_norm': '8.084', 'learning_rate': '3.125e-05', 'epoch': '2.059'}                      
{'loss': '0.1363', 'grad_norm': '12.64', 'learning_rate': '2.644e-05', 'epoch': '2.206'}                      
{'loss': '0.1523', 'grad_norm': '1.707', 'learning_rate': '2.163e-05', 'epoch': '2.353'}                      
{'loss': '0.1313', 'grad_norm': '0.7928', 'learning_rate': '1.683e-05', 'epoch': '2.5'}                       
{'loss': '0.1762', 'grad_norm': '2.242', 'learning_rate': '1.202e-05', 'epoch': '2.647'}                      
{'loss': '0.08413', 'grad_norm': '19.62', 'learning_rate': '7.212e-06', 'epoch': '2.794'}                     
{'loss': '0.1058', 'grad_norm': '11.48', 'learning_rate': '2.404e-06', 'epoch': '2.941'}                      
{'eval_loss': '0.5091', 'eval_accuracy': '0.8326', 'eval_f1': '0.8492', 'eval_precision': '0.8492', 'eval_recall': '0.8492', 'eval_runtime': '0.5455', 'eval_samples_per_second': '1314', 'eval_steps_per_second': '22', 'epoch': '3'}
Writing model shards: 100%|█████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.26it/s]
{'train_runtime': '17.8', 'train_samples_per_second': '181.9', 'train_steps_per_second': '11.46', 'train_loss': '0.365', 'epoch': '3'}
100%|███████████████████████████████████████████████████████████████████████| 204/204 [00:17<00:00, 20.55it/s]
There were missing keys in the checkpoint model loaded: ['bert.embeddings.LayerNorm.weight', 'bert.embeddings.LayerNorm.bias', 'bert.encoder.layer.0.attention.output.LayerNorm.weight', 'bert.encoder.layer.0.attention.output.LayerNorm.bias', 'bert.encoder.layer.0.output.LayerNorm.weight', 'bert.encoder.layer.0.output.LayerNorm.bias', 'bert.encoder.layer.1.attention.output.LayerNorm.weight', 'bert.encoder.layer.1.attention.output.LayerNorm.bias', 'bert.encoder.layer.1.output.LayerNorm.weight', 'bert.encoder.layer.1.output.LayerNorm.bias', 'bert.encoder.layer.2.attention.output.LayerNorm.weight', 'bert.encoder.layer.2.attention.output.LayerNorm.bias', 'bert.encoder.layer.2.output.LayerNorm.weight', 'bert.encoder.layer.2.output.LayerNorm.bias', 'bert.encoder.layer.3.attention.output.LayerNorm.weight', 'bert.encoder.layer.3.attention.output.LayerNorm.bias', 'bert.encoder.layer.3.output.LayerNorm.weight', 'bert.encoder.layer.3.output.LayerNorm.bias', 'bert.encoder.layer.4.attention.output.LayerNorm.weight', 'bert.encoder.layer.4.attention.output.LayerNorm.bias', 'bert.encoder.layer.4.output.LayerNorm.weight', 'bert.encoder.layer.4.output.LayerNorm.bias', 'bert.encoder.layer.5.attention.output.LayerNorm.weight', 'bert.encoder.layer.5.attention.output.LayerNorm.bias', 'bert.encoder.layer.5.output.LayerNorm.weight', 'bert.encoder.layer.5.output.LayerNorm.bias', 'bert.encoder.layer.6.attention.output.LayerNorm.weight', 'bert.encoder.layer.6.attention.output.LayerNorm.bias', 'bert.encoder.layer.6.output.LayerNorm.weight', 'bert.encoder.layer.6.output.LayerNorm.bias', 'bert.encoder.layer.7.attention.output.LayerNorm.weight', 'bert.encoder.layer.7.attention.output.LayerNorm.bias', 'bert.encoder.layer.7.output.LayerNorm.weight', 'bert.encoder.layer.7.output.LayerNorm.bias', 'bert.encoder.layer.8.attention.output.LayerNorm.weight', 'bert.encoder.layer.8.attention.output.LayerNorm.bias', 'bert.encoder.layer.8.output.LayerNorm.weight', 'bert.encoder.layer.8.output.LayerNorm.bias', 'bert.encoder.layer.9.attention.output.LayerNorm.weight', 'bert.encoder.layer.9.attention.output.LayerNorm.bias', 'bert.encoder.layer.9.output.LayerNorm.weight', 'bert.encoder.layer.9.output.LayerNorm.bias', 'bert.encoder.layer.10.attention.output.LayerNorm.weight', 'bert.encoder.layer.10.attention.output.LayerNorm.bias', 'bert.encoder.layer.10.output.LayerNorm.weight', 'bert.encoder.layer.10.output.LayerNorm.bias', 'bert.encoder.layer.11.attention.output.LayerNorm.weight', 'bert.encoder.layer.11.attention.output.LayerNorm.bias', 'bert.encoder.layer.11.output.LayerNorm.weight', 'bert.encoder.layer.11.output.LayerNorm.bias'].
There were unexpected keys in the checkpoint model loaded: ['bert.embeddings.LayerNorm.beta', 'bert.embeddings.LayerNorm.gamma', 'bert.encoder.layer.0.attention.output.LayerNorm.beta', 'bert.encoder.layer.0.attention.output.LayerNorm.gamma', 'bert.encoder.layer.0.output.LayerNorm.beta', 'bert.encoder.layer.0.output.LayerNorm.gamma', 'bert.encoder.layer.1.attention.output.LayerNorm.beta', 'bert.encoder.layer.1.attention.output.LayerNorm.gamma', 'bert.encoder.layer.1.output.LayerNorm.beta', 'bert.encoder.layer.1.output.LayerNorm.gamma', 'bert.encoder.layer.2.attention.output.LayerNorm.beta', 'bert.encoder.layer.2.attention.output.LayerNorm.gamma', 'bert.encoder.layer.2.output.LayerNorm.beta', 'bert.encoder.layer.2.output.LayerNorm.gamma', 'bert.encoder.layer.3.attention.output.LayerNorm.beta', 'bert.encoder.layer.3.attention.output.LayerNorm.gamma', 'bert.encoder.layer.3.output.LayerNorm.beta', 'bert.encoder.layer.3.output.LayerNorm.gamma', 'bert.encoder.layer.4.attention.output.LayerNorm.beta', 'bert.encoder.layer.4.attention.output.LayerNorm.gamma', 'bert.encoder.layer.4.output.LayerNorm.beta', 'bert.encoder.layer.4.output.LayerNorm.gamma', 'bert.encoder.layer.5.attention.output.LayerNorm.beta', 'bert.encoder.layer.5.attention.output.LayerNorm.gamma', 'bert.encoder.layer.5.output.LayerNorm.beta', 'bert.encoder.layer.5.output.LayerNorm.gamma', 'bert.encoder.layer.6.attention.output.LayerNorm.beta', 'bert.encoder.layer.6.attention.output.LayerNorm.gamma', 'bert.encoder.layer.6.output.LayerNorm.beta', 'bert.encoder.layer.6.output.LayerNorm.gamma', 'bert.encoder.layer.7.attention.output.LayerNorm.beta', 'bert.encoder.layer.7.attention.output.LayerNorm.gamma', 'bert.encoder.layer.7.output.LayerNorm.beta', 'bert.encoder.layer.7.output.LayerNorm.gamma', 'bert.encoder.layer.8.attention.output.LayerNorm.beta', 'bert.encoder.layer.8.attention.output.LayerNorm.gamma', 'bert.encoder.layer.8.output.LayerNorm.beta', 'bert.encoder.layer.8.output.LayerNorm.gamma', 'bert.encoder.layer.9.attention.output.LayerNorm.beta', 'bert.encoder.layer.9.attention.output.LayerNorm.gamma', 'bert.encoder.layer.9.output.LayerNorm.beta', 'bert.encoder.layer.9.output.LayerNorm.gamma', 'bert.encoder.layer.10.attention.output.LayerNorm.beta', 'bert.encoder.layer.10.attention.output.LayerNorm.gamma', 'bert.encoder.layer.10.output.LayerNorm.beta', 'bert.encoder.layer.10.output.LayerNorm.gamma', 'bert.encoder.layer.11.attention.output.LayerNorm.beta', 'bert.encoder.layer.11.attention.output.LayerNorm.gamma', 'bert.encoder.layer.11.output.LayerNorm.beta', 'bert.encoder.layer.11.output.LayerNorm.gamma'].
100%|███████████████████████████████████████████████████████████████████████| 204/204 [00:17<00:00, 11.38it/s]
在测试集上评估...
100%|█████████████████████████████████████████████████████████████████████████| 12/12 [00:00<00:00, 25.29it/s]

**评估结果**:{'eval_loss': 0.41219010949134827, 'eval_accuracy': 0.8228730822873083, 'eval_f1': 0.8313413014608234, 'eval_precision': 0.8816901408450705, 'eval_recall': 0.7864321608040201, 'eval_runtime': 0.55, 'eval_samples_per_second': 1303.523, 'eval_steps_per_second': 21.816, 'epoch': 3.0}