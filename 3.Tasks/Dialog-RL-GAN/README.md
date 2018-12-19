### 下载代码
> 基于 [MWransky/dialogue-gan](https://github.com/MWransky/dialogue-gan) 重构，数据集请到该仓库下载

链接：https://github.com/imageslr/dialogue-gan-new

首先跳转到存储代码的文件夹下，下载此仓库:
```
git clone https://github.com/imageslr/dialogue-gan-new.git
```

**【必须】** 然后用自己的名字创建一个分支：
```
git checkout -b 分支名
# 举个例子：git checkout -b dev-zhu
```

这样修改代码不会影响到主分支。

修改完代码后，可以发送 Pull request，最终我们共同维护这一份代码。具体操作可以等需要的时候再讨论。

### 文件结构
```
├── adversarial_train.py # 对抗训练
├── debug.py # 测试生成器的预训练
├── debug_disc_model.py # 测试判别器的构建
├── debug_gen_model.py # 测试生成器的构建
├── disc
│   ├── hier_rnn_data.py # 数据
│   ├── hier_rnn_model.py # 模型
│   └── hier_rnn_train.py # 训练
├── gen
│   ├── gen_data.py # 数据
│   ├── gen_model.py # 模型
│   ├── gen_train.py # 训练
│   └── seq2seq.py
├── disc_data # 直接使用旧版的数据即可，不上传到git仓库
├── gen_data # 直接使用旧版的数据即可，不上传到git仓库
└── utils
    ├── conf.py # 配置文件
    ├── data_utils.py
    └── utils.py
```

### 代码说明
#### 配置模型最多训练多少次
`utils/conf.py`：

```python
class disc_config(object):
    max_pre_train_step = 30000 # 预训练的时候最多训练多少次
    
class gen_config(object):
    max_pre_train_step = 30000 # 预训练的时候最多训练多少次
    
class adver_config(object):
    max_train_step = 30000 # 对抗训练最多训练多少次
```

#### 命令行打印日志
`utils/utils.py` 的 `just_message(str)`方法，参数是一个字符串，作用是在该字符串左右添加“=======”。

```python
from utils.utils import just_message as just

print(just("Update Discriminator: %d" % current_step))
```

输出类似于这种样式：
```
============================================================== Update Discriminator: 2 ===============================================================
================================================================== mc_search: False ==================================================================
================================================================ Update Generator: 2 =================================================================
================================================================== mc_search: True ===================================================================
step_reward:  0.710699575700346
gen_step_loss:  2.106717
gen_step_adjusted_loss:  1.4972429
t_step_loss:  2.0420232
t_adjusted_loss 2.0420232
```

### 未解决的问题
对于代码中未解决的问题，我在注释中用“TODO(Zhu)”进行了标注。大家在读代码的时候可以帮我解决这些问题，也可以把自己的问题用“TODO(XXX)”标注。

修改代码后，可以发送 Pull Request，一起维护代码、讨论代码中的问题、解决问题。

#### 一个比较重要的问题：这个代码是怎么实现强化学习的？
读代码发现：
* 在`adversarial_train.py`的 117 行，判别器最终得到的 reward 是**它自己生成的所有负例得到的 reward_i 的平均值**
* 在`adversarial_train.py`的 112 行，把这个 reward 传给判别器的 step(..) 进行训练，同时参数`up_reward`设置为`True`
* 在`gen.gem_model.py`的`_build_graph()`方法的 119 行发现，如果采用`up_reward`为真，最终的 loss 就要乘上 reward，否则只有 loss

是这样实现强化学习的吗？