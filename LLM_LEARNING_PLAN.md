# LLM 学习计划(基于 Kronos 代码实现)

> 学习方式:每个阶段都遵循同一个循环 —— **先懂概念(不看代码)→ 对照 Kronos 代码看它怎么实现 → 动手改代码/加 print 验证理解 → 补充这个概念在真实 LLM 里的差异点**。不要跳过"动手改代码"这一步,光读不改,理解不会扎实。

---

## 前置基础知识

正式开始前,建议先具备(或边学边补)以下基础,按优先级排列:

### 1. 编程基础(必须,优先级最高)
- **Python**:类与对象(`class`、`__init__`、继承)、列表/字典操作、基本的函数式写法(lambda、列表推导)。
- **PyTorch 基础**:`nn.Module` 怎么定义一个网络、`forward` 方法、`torch.Tensor` 的基本操作(`shape`、`view`/`reshape`、广播机制 broadcasting)、`.to(device)`、`with torch.no_grad()`。
  - 不需要精通,能看懂 [model/module.py](model/module.py) 里的类定义即可上手,遇到不懂的API现查即可。

### 2. 数学基础(不需要精通,建立直觉即可)
- **线性代数**:矩阵乘法是什么、向量的点积(dot product)的几何含义(衡量两个向量的相似度)。这是理解注意力机制(`Q·K^T`)的基础。
- **概率与统计**:什么是概率分布、softmax函数的作用(把一组数值变成加起来等于1的概率分布)、期望值的直觉。这是理解采样(temperature/top-k/top-p)和交叉熵损失的基础。
- **微积分/梯度**:知道"导数是变化率"、"梯度下降是沿着让loss变小的方向调整参数"即可,**不需要会手推反向传播的数学推导**——现代深度学习框架(PyTorch的autograd)会自动帮你算,理解直觉比会推公式更重要。

### 3. 机器学习基础概念
- 什么是"训练"(用数据反复调整模型参数,使得损失函数变小)。
- 什么是"损失函数"(衡量模型预测和真实答案差多少的指标)。
- 什么是"过拟合"(模型在训练数据上表现很好,但在没见过的数据上表现差)。
- 什么是"batch"、"epoch"(一批数据、一轮完整遍历训练集)。

### 4. 深度学习基础概念
- 神经网络的基本结构:线性层(`nn.Linear`)+ 激活函数(非线性,如 ReLU/SiLU)的堆叠。
- 为什么需要激活函数(没有它,多层线性层等价于一层线性层,无法拟合复杂函数)。
- 优化器(Adam)的直觉:比普通梯度下降更"聪明"地自适应调整每个参数的更新步长,不需要懂内部公式。

### 5. 可以完全跳过、不需要提前掌握的
- 反向传播的数学推导细节(链式法则的具体公式)——用到时知道"框架自动算梯度"就够了。
- CUDA/GPU 底层编程——只需要知道"GPU并行算矩阵乘法比CPU快很多"即可。
- 任何具体的 NLP 传统方法(如 n-gram、HMM、传统词向量 word2vec)——对理解 Transformer/LLM 不是必需前提。

> **一句话总结**:如果你能看懂"矩阵乘法"、"softmax是把数字变成概率"、"训练就是不断调整参数让预测更准"这三句话,就可以直接开始阶段0了,其余细节在过程中遇到再补。

---

## 学习路线总览

| 阶段 | 主题 | 预计时长 | 核心代码位置 |
|---|---|---|---|
| 0 | 环境准备 | 1-3天 | `examples/prediction_example.py` |
| 1 | Token化与Embedding | 3-5天 | `model/kronos.py` (KronosTokenizer), `model/module.py` (HierarchicalEmbedding) |
| 2 | 自注意力机制 | 5-7天(重点) | `model/module.py` (MultiHeadAttentionWithRoPE, RotaryPositionalEmbedding) |
| 3 | Transformer Block完整结构 | 3-4天 | `model/module.py` (RMSNorm, FeedForward, TransformerBlock) |
| 4 | 训练目标与损失函数 | 2-3天 | `model/module.py` (DualHead.compute_loss), `model/kronos.py` (Kronos.forward) |
| 5 | 推理与自回归生成 | 3-4天 | `model/kronos.py` (sample_from_logits, auto_regressive_inference) |
| 6 | 补齐Kronos未覆盖的LLM技术 | 1-2周(泛读) | 无(外部资料) |

总计约 4-6 周(每天1-2小时),不要求严格按天数,以"能不能给自己讲清楚"作为每阶段的过关标准。

---

## 阶段 0:准备工作(1-3天)

**目标**:能跑起来、能读懂 Python/PyTorch 基本语法。

- 确保能在本地跑通仓库的一次推理(比如 `examples/prediction_example.py`),打断点/加print,观察张量的 `shape`。这是贯穿整个学习过程最重要的调试手段。
- 如果 PyTorch 基础薄弱,先花半天过一遍 PyTorch 官方 60分钟入门教程,不需要精通,能看懂类的写法即可。

**验收标准**:能自己在 `Kronos.forward`([model/kronos.py:239](model/kronos.py#L239))里加一行 `print(x.shape)` 并看懂输出。

---

## 阶段 1:Token 化与 Embedding(3-5天)

**先懂概念**:
- 神经网络不能直接吃"文字"或"数字序列",必须先变成离散的整数ID(token),再查表映射成向量(embedding)。
- 为什么要离散化?——这样模型可以用"分类"(预测下一个token是词表里的哪一个)而不是"回归"(直接预测一个连续数值),分类问题用交叉熵损失训练更稳定。

**对照代码看实现**:
- `BSQuantizer`(通过 `KronosTokenizer`,[model/kronos.py:13](model/kronos.py#L13)):把连续的K线向量,量化成离散的 s1/s2 token id。这一步对应文本LLM里的**分词器(tokenizer)**,只是文本用的是 BPE 算法,这里用的是"把每个维度push到±1"的二值量化。
- `HierarchicalEmbedding`([model/module.py:400](model/module.py#L400)):拿到 token id 后查 `nn.Embedding` 表得到向量,这一步和文本LLM的 word embedding 一模一样。

**动手练习**:
1. 在 `KronosTokenizer.encode`([model/kronos.py:142](model/kronos.py#L142))里打印 `z_indices` 的取值范围,验证它确实落在 `[0, 2^s1_bits)` 和 `[0, 2^s2_bits)` 之间。
2. 自己写一个10行的小脚本,用 `nn.Embedding(1000, 16)` 随便查几个id,打印出向量,理解"embedding 就是一张查找表,训练时这张表的数值会被梯度更新"。

**补充真实LLM的差异**:
- 去了解一下 BPE(Byte-Pair Encoding)算法的原理(不需要精通实现,能讲清楚"如何从字符逐步合并出子词"即可)。推荐直接跑一遍 [tiktokenizer.vercel.app](https://tiktokenizer.vercel.app) 这种在线工具,输入英文/中文句子看它怎么被切分——这是理解真实LLM tokenizer最快的方式。

---

## 阶段 2:自注意力机制(Self-Attention)——最核心的一周

**先懂概念(建议花2-3天只研究这一个机制)**:
- 核心问题:一个序列里,每个位置的向量如何"看到"其他位置的信息?
- QKV 三个矩阵的直觉:Query(我在找什么)、Key(我是什么)、Value(我能提供什么信息),`attention_weight = softmax(Q·K^T / sqrt(d))`,输出 `= attention_weight · V`。
- 为什么叫"自"注意力:Q、K、V 都来自同一个输入序列(区别于"交叉注意力")。
- 为什么要除以 `sqrt(d)`:防止点积数值过大导致softmax梯度消失。
- Causal Mask(因果掩码):为什么GPT类模型在预测第i个token时,只能看到第1~i个token,不能看到未来——这是"自回归"能成立的前提。

**对照代码看实现**:
- `MultiHeadAttentionWithRoPE`([model/module.py:315](model/module.py#L315)):逐行对照公式看代码——`q_proj/k_proj/v_proj` 就是QKV矩阵,`F.scaled_dot_product_attention(..., is_causal=True)` 就是因果掩码,`n_heads` 拆分就是"多头"(把d_model切成多份并行算注意力,让模型同时关注不同的模式)。
- 位置编码:`RotaryPositionalEmbedding`(RoPE,[model/module.py:284](model/module.py#L284))——这是目前LLaMA/Qwen等主流开源LLM都在用的位置编码方式,比GPT-2最初用的绝对位置编码更好地处理长序列外推。

**动手练习(重要,强烈建议做)**:
1. 手写一个最简化版本的attention(不用 `F.scaled_dot_product_attention`,自己用 `torch.matmul`+`softmax`+手动加mask实现),对比输出和官方函数是否一致。可以参考 Andrej Karpathy 的 nanoGPT/`makemore` 系列视频。
2. 去掉 `is_causal=True`,观察模型训练/推理会发生什么变化(理论上会"看到未来"作弊,训练loss会不正常地低)。

**补充真实LLM的差异**:
- 这一块和真实LLM几乎没有差异,是可以100%迁移的知识,唯一要留意的是有些模型(如BERT)用的是双向注意力(无因果掩码),GPT系列(包括Kronos)是单向因果注意力。

---

## 阶段 3:Transformer Block 的完整结构(3-4天)

**先懂概念**:
- 一个Transformer层 = 注意力子层 + 前馈网络(FFN)子层,每个子层外面包一层"残差连接+归一化"。
- 为什么要残差连接(`x = x + sublayer(x)`):防止深层网络梯度消失/爆炸,让信息能"抄近路"直接传到后面层。
- 为什么要归一化(Normalization):稳定每层输出的数值分布,加速训练收敛。
- Pre-Norm vs Post-Norm:现代LLM几乎都用Pre-Norm(先归一化再进子层)。
- FFN的作用:注意力负责"信息在不同位置间流动",FFN负责"对每个位置单独做非线性变换",两者配合才是一层完整的处理。

**对照代码看实现**:
- `RMSNorm`([model/module.py:257](model/module.py#L257)):对比它和标准 `LayerNorm` 的区别(RMSNorm去掉了减均值这一步,只做缩放,计算更快),这是LLaMA系列的标配选择。
- `FeedForward`([model/module.py:271](model/module.py#L271)):这是 SwiGLU 结构(`w1`、`w2`、`w3` 三个矩阵 + SiLU激活),是当前主流开源LLM(LLaMA/Mistral/Qwen)的标准FFN设计,比早期GPT-2用的两层ReLU-FFN更好。
- `TransformerBlock`([model/module.py:465](model/module.py#L465)):看它的 `forward` 方法,清晰地体现了 `norm→attn→残差`、`norm→ffn→残差` 这个标准结构。

**动手练习**:
1. 打印一层 `TransformerBlock` 前后的张量shape,确认输入输出维度不变(这是"可以叠加N层"的前提)。
2. 试着把 `n_layers` 从12改成2,重新跑一次微调,观察loss曲线/效果变化,直觉感受"层数"对模型能力的影响。

---

## 阶段 4:训练目标与损失函数(2-3天)

**先懂概念**:
- 自回归语言模型的训练目标:给定前面的token,预测下一个token是词表中的哪一个——这是一个多分类问题,用交叉熵损失(Cross-Entropy Loss)。
- Teacher Forcing:训练时用真实的历史token作为输入(而不是模型自己生成的),让训练更稳定、可并行。

**对照代码看实现**:
- `DualHead.compute_loss`([model/module.py:494](model/module.py#L494)):看 `F.cross_entropy` 是怎么在 s1/s2 两个token上分别计算损失并相加平均的。
- `Kronos.forward` 里的 `use_teacher_forcing` 参数([model/kronos.py:239](model/kronos.py#L239)):对照"Teacher Forcing"概念看它具体怎么实现的(用真实的 `s1_targets` 而不是模型采样出来的结果去算 s2)。

**动手练习**:
1. 自己写一个玩具例子:随便定义一个 `[batch, seq_len, vocab_size]` 的logits张量和一个 `[batch, seq_len]` 的target,手动调用 `F.cross_entropy`,理解为什么要 `reshape(-1, vocab_size)`。

---

## 阶段 5:推理与生成(自回归采样)(3-4天)

**先懂概念**:
- 训练时是"整个序列一次性并行算loss",推理时必须"一个token一个token地生成",因为要生成的token本身还不存在。
- Temperature、Top-k、Top-p(Nucleus)采样:控制生成的"随机性/多样性 vs 确定性"。
- KV Cache(重要但Kronos代码里没有直接体现的优化):真实LLM推理时会缓存已经算过的Key/Value,避免每生成一个新token就重新计算整个历史序列的注意力——这是推理提速的关键工程手段,建议之后单独找资料补上这一课(Kronos的 `auto_regressive_inference` 用滑动窗口重算,没有做KV Cache优化,这是和真实LLM推理引擎的一个差异点,值得注意)。

**对照代码看实现**:
- `sample_from_logits` / `top_k_top_p_filtering`([model/kronos.py:373](model/kronos.py#L373)、[331](model/kronos.py#L331)):逐行对照理解 top-k(只保留概率最高的k个候选)和 top-p(累积概率超过p就截断)分别怎么实现。
- `auto_regressive_inference`([model/kronos.py:389](model/kronos.py#L389)):这是完整的自回归生成循环,重点看 `for i in range(pred_len)` 这个循环——每一步:预测→采样→把采样结果拼回输入序列→重复。这和 ChatGPT/Claude 生成一段回复的底层循环逻辑完全一致(只是token含义不同)。

**动手练习**:
1. 把 `T`(temperature)分别设成 0.1 和 2.0 跑同一段推理,观察生成结果的多样性差异,直觉感受"温度"的作用。
2. 把 `sample_logits=False`(贪心解码,直接取概率最大的token)和 `True`(随机采样)对比,理解两种解码策略的区别。

---

## 阶段 6:补齐 Kronos 没有覆盖的真实LLM技术(1-2周,泛读为主)

这部分不需要动手实现,理解原理、知道关键词、能讲清楚"是什么、为什么需要"就够了:

| 主题 | 为什么Kronos没有 | 建议资料 |
|---|---|---|
| BPE/SentencePiece真实分词算法 | 用的是BSQ量化替代 | 读一遍 [minbpe](https://github.com/karpathy/minbpe)(Karpathy写的教学版BPE实现,~100行代码) |
| 大规模分布式训练(数据并行/张量并行/ZeRO) | 模型太小不需要 | 了解"为什么千亿参数模型不能塞进一张卡"的直觉即可,不用深挖工程细节 |
| KV Cache / 推理加速工程 | 代码里没做 | 搜"LLM KV cache explained",理解一次就够 |
| 指令微调(Instruction Tuning) | Kronos是预测任务,没有"指令遵循"这个概念 | 了解 Alpaca/FLAN 数据格式的直觉即可 |
| RLHF / DPO(对齐技术) | 完全在预训练之外的后训练阶段 | 读一遍 InstructGPT 论文摘要+博客解读,不用啃数学细节 |
| Mixture of Experts(MoE) | Kronos是稠密模型 | 了解"多个专家网络+路由"的直觉即可 |

---

## 推荐的辅助资料

1. **Andrej Karpathy 的 "Let's build GPT: from scratch"(YouTube/B站都有)**——公认最好的从零手写GPT视频,和本计划高度互补,建议在阶段2-3之间看。
2. **"Attention Is All You Need" 论文**——不用一开始就啃,建议做完阶段2的动手练习后再读原论文,会容易理解很多。
3. **[nanoGPT](https://github.com/karpathy/nanoGPT)** 仓库——比Kronos更"纯"的语言模型实现(真的是文本),代码风格和Kronos高度相似,读完Kronos后再读一遍nanoGPT,交叉验证会加深理解。

---

## Kronos 与真实 LLM 的关键映射表(速查)

| 概念 | Kronos 实现 | 真实 LLM 对应 |
|---|---|---|
| Tokenization | `BSQuantizer` | BPE / SentencePiece |
| Token Embedding | `HierarchicalEmbedding` | 词嵌入表 |
| 位置编码 | `RotaryPositionalEmbedding`(RoPE) | LLaMA/Qwen 同款 RoPE |
| 因果自注意力 | `MultiHeadAttentionWithRoPE` | GPT式 causal self-attention |
| FFN结构 | `FeedForward`(SwiGLU) | LLaMA/Mistral/Qwen 标准FFN |
| 归一化 | `RMSNorm` | LLaMA系列标配 |
| 残差+Pre-Norm | `TransformerBlock.forward` | 主流LLM block结构 |
| 自回归生成 | `auto_regressive_inference` | next-token-prediction循环 |
| 采样策略 | `sample_from_logits` | ChatGPT/Claude同款解码策略 |
