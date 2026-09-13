# Relation-Space：有预算上限的小模型实验与多卡训练

**训练前不能保证同质量加速。先做计算开销检查，再决定是否花小额训练预算。**
这里训练的是约 1090 万参数的双向 Transformer，词表为 256 个 UTF-8 byte 加一个
边界符，MASK 在输出词表外。没有使用、修改或继续训练 LLaDA-8B，也没有证明
byte 级结果能迁移到 BPE 或数学/代码任务。测试成功仅代表实验工具可运行。

## 三道独立关卡

| 阶段 | 消耗 | 输出与判断 |
| --- | --- | --- |
| `prepare` | CPU，下载 WikiText-2 的 train/validation | 划分、训练集拟合的冻结关系码、哈希；不读取 test |
| `preflight` | 单卡，无训练、无预训练权重 | 相同随机模型的 1/2/4/8/16 步生成 + 逆变换时间；只有开销判断 |
| `pilot` | 两个模型各最多 200 次更新、600 秒训练循环 | 原 token 对关系码的少步验证损失、生成耗时；仍是短预算筛查 |

每个阶段单独启动，不会自动进入下一阶段。训练时间限制在每次 optimizer update 后
检查；一次更新、初始化、存盘和评估时间可能使总作业时间超过该限制。
碰到上限时保存 checkpoint，停止当前 campaign，拒绝继续用不相等的更新数比较。

`preflight` 中“原 token 16 步 / 关系码 4 步”时间比至少 1.5 只表示**有计算空间**，
不是模型学会了用 4 步完成生成。随机模型本来就会生成无意义文本。
200 更新训练也不能证明方案成败：若模型还明显欠拟合，需要先看学习曲线；
不得因欠训练的小模型暂时打平就宣称理论被否定。

真正继续投入的证据，应当是相同训练预算下，真实文本的关系模型在更少步数有稳定的
原文空间分布质量优势，并在多个种子下重复，扣除逆变换后仍节省时间。
本轮没有自动调参、自动扩展到 8B 或自动运行长训练的逻辑。

## 1. 拉代码与准备数据（在服务器执行）

先进入一个控制用的 tmux：

```bash
tmux new-session -A -s relation-console
```

在里面执行：

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fastdllm311
cd /home/xuyouwen/REFRAME-dLLM
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
git pull --ff-only
CUDA_VISIBLE_DEVICES="" python -m pytest relation_diffusion/tests -q

export DATA_DIR=/home/xuyouwen/hf_home_local/relation_diffusion/wikitext2_byte128_v1
SESSION=relation-prepare bash relation_diffusion/scripts/launch_tmux.sh prepare
```

下载的是 [WikiText](https://huggingface.co/datasets/Salesforce/wikitext) 的
`wikitext-2-raw-v1`，只读取 train/validation。原始 HF 数据缓存继续使用
`/home/xuyouwen/hf_home_local/datasets`，Hub 缓存继续使用
`/home/xuyouwen/hf_hub_local`，不需要下载新权重。
准备脚本使用 `https://hf-mirror.com`，清除失效的反向隧道代理；训练/评估离线执行。

语料按官方划分，逐行 UTF-8 编码并插入边界符，打包成固定 128-symbol 样本；
前 32 symbols 作为可见前缀，后 96 symbols 为生成目标。末尾不足一块的内容丢弃。
跨块可能切断 UTF-8 字符，因此第一轮衡量的是固定 symbol 分布，不是完善的文本产品。
从训练集移除与验证集完全相同的块；最多抽 8192 个训练块拟合关系码。
准备耗时、采样索引、数据和编码哈希均保存，不能拿验证/测试答案拟合编码。

准备还输出 `independent_diagnostic.json`：用训练块估计每个位置的平滑类别频率，
在相同验证块上比较原 token、重编号、随机和统计关系码的独立生成分布 NLL。
这不训练神经网络，可以先看编码在简单统计模型下是否有改善迹象。它忽略前缀内容，
也不代表 Transformer 能获得相同收益；不是未来模型性能的上下界。
如果统计关系码连这一项都没有改善，就更值得先检查编码设计，而非直接启动大训练。

脚本会打印具体日志和 tmux 名称。完成标记：`DATA_DIR/manifest.json` 和作业
`exit_code=0`。若准备失败，请换新的 DATA_DIR 后重跑；默认拒绝覆盖已有数据。

## 2. 先运行“完全不训练”的开销检查

确认物理 GPU 3 是分配给自己的空闲卡后：

```bash
export GPU_IDS=3
export RUN_DIR="$PWD/relation_diffusion/runs/preflight_$(date +%Y%m%d_%H%M%S)"
SESSION=relation-preflight bash relation_diffusion/scripts/launch_tmux.sh preflight
echo "$RUN_DIR/preflight.json"
```

这一步只用单卡。在输出里看 `ratio_identity16_to_relation4`、`cost_gate_pass`、
每个步数的生成时间与 `mean_decode_seconds`。同时检查可逆性和单码扰动影响范围。
即使显示 `cost_gate_pass=true`，也不自动启动训练，更不能把时间比称为同质量加速比。

## 3. 手动决定后，跑限定预算的两模型 pilot

只改 GPU_IDS 就能选卡。单卡示例：

```bash
export GPU_IDS=3
export RUN_DIR="$PWD/relation_diffusion/runs/pilot_$(date +%Y%m%d_%H%M%S)"
STEPS=200 MAX_SECONDS=600 GLOBAL_BATCH=48 MICRO_BATCH=8 \
  SESSION=relation-pilot bash relation_diffusion/scripts/launch_tmux.sh pilot
echo "$RUN_DIR/comparison.json"
```

若已获得六张卡的使用权，改成实际分配的编号，例如：

```bash
export GPU_IDS=0,1,2,3,4,5
export RUN_DIR="$PWD/relation_diffusion/runs/pilot6_$(date +%Y%m%d_%H%M%S)"
STEPS=200 MAX_SECONDS=600 GLOBAL_BATCH=48 MICRO_BATCH=8 \
  SESSION=relation-pilot6 bash relation_diffusion/scripts/launch_tmux.sh pilot
```

**示例卡号不代表这些卡当前空闲或归你使用。** 启动器把 nvidia-smi 的物理编号
转成 UUID 绑定，拒绝重复编号；每张卡默认要求至少 24 GiB 空闲。
它不会停止其它进程，也不会自动换卡。

两种命令的每个模型都训练 200 次更新 × 48 样本 × 128 symbols = 1,228,800
个输入 symbols，优化器更新数和数据噪声一致。六卡不会暗中多看六倍数据。
两模型串行运行，每个模型的训练循环各有 600 秒上限，评估另外计时。

默认 1090 万参数模型很小，六卡 DDP 的通信开销可能让提速有限。
为了提高整体实验产出，后续也可以在不同 GPU 上独立跑不同编码/种子；
这种做法提高的是实验吞吐，不能报告为单模型训练加速。

## 多卡怎样工作

使用 [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
和 `torchrun`，一张卡一个进程，每张卡保存完整的小模型和优化器。
全局 batch 与梯度累积满足：

```
global_batch = GPU 数量 × micro_batch × accumulation
48 = 1 × 8 × 6
48 = 2 × 8 × 3
48 = 3 × 8 × 2
48 = 6 × 8 × 1
```

不能整除时直接报错。例如 4 卡可用 `MICRO_BATCH=4`、累积 3 次。
前几个 microbatch 使用 `no_sync()`，最后一次才同步梯度；归一化保持全局目标不变。
每个更新的样本和噪声由 seed + update 决定，先建立同一个全局 batch 再切给各 rank。
不使用 dropout，续训可恢复相同的样本/噪声序列，卡数变化只允许浮点求和顺序差异。

这不是把六张卡显存拼成一张 192 GB 卡。未来如果训练 LLaDA-8B，需要另外设计
FSDP/ZeRO 等分片方案；当前工具没有实现或验证 8B 多卡训练。

## 训练目标与公平对照

所有模型采用同一个双向 attention 网络，使用 PyTorch SDPA，没有 KV cache。
训练先把完整干净目标编码成关系码，前缀保持原文，再随机遮盖目标码。
损失只作用于被遮盖位置，以遮盖概率的倒数加权，并按全部目标位置归一化。
这是参照 [LLaDA 训练指南](https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md)
的 masked-denoising 思路；这里没有冒充完整 LLaDA 预训练复现。

| CODEC / OBJECTIVE | 用途 |
| --- | --- |
| identity / diffusion | 原 token 坐标基线 |
| rename / diffusion | 只对 byte 重编号，排除单纯 label 更换 |
| random / diffusion | 两层随机稀疏条件排列，排除随便混合就有效 |
| relation1 / diffusion | 一层训练统计选择的关系码 |
| relation2 / diffusion | 两层训练统计选择的关系码，首轮候选 |
| relation2 / one-step | 相同关系码、全部目标遮盖的一步生成器，检查收益是否依赖多步 diffusion |

当前统计拟合只是“条件频率排序 + 稀疏排列补全”的启发式，并非已经学出的最优关系。
GPU 使用 257×257 的小查表实现每层并行变换；BPE/LLaDA 词表不能照搬平方表。

`pilot` 只跑 identity 与 relation2。通过初筛后，手动启动其它控制，例如：

```bash
export GPU_IDS=3
export RUN_DIR="$PWD/relation_diffusion/runs/random_$(date +%Y%m%d_%H%M%S)"
CODEC=random OBJECTIVE=diffusion SEED=1234 STEPS=200 MAX_SECONDS=600 \
  SESSION=relation-random bash relation_diffusion/scripts/launch_tmux.sh train
```

同一比较组保持 WIDTH/LAYERS/HEADS、全局 batch、计划更新数、学习率、dtype、数据
和 seed 相同。编码额外耗时包含在训练计时中；准备/拟合总成本也单独记录。
跨种子实验另设 `SEED=2345`、`SEED=3456`，不能只挑表现最好的 seed。

## 怎么读质量—时间结果

`evaluation.json` 对固定揭示顺序的 1/2/4/8/16 步分别记录：

- `path_bits_per_symbol`：原文空间诱导分布的固定路径负对数似然，越低越好。
  编码双射允许映回原文比较；不是传统 AR perplexity、准确率或所有路径边缘化后的 diffusion 似然。
- `nll_per_example_nats`：逐样本数值，便于做配对分析；这里只提供均值，不声称统计显著。
- `timing.mean_seconds`：单卡、batch 1，单条固定前缀的完整生成与逆变换时间；
  两次预热、默认五次测量。第一轮仅为开销诊断，后续还需多 prompt 延迟统计。
- `error_spread`：单个码被改动后原文受影响的位置数量。
- `sample_text`：一个生成示例，辅助发现格式/编码异常，不能替代文本质量评估。

评分使用 teacher forcing，仅揭示之前组的真实码，当前/未来组保持 MASK。
实际采样只获得前缀，不会使用真实答案或将中间残缺的码偷偷解码成完整原文。
一步模型只评一步；不能把它放到没训练过的多步路径上后比较输赢。

`compare.py` 检查实际更新数、模型、数据哈希、样本数等一致后才汇总。
重点看关系 4 步能否达到原 token 8/16 步的原文路径分布质量，同时生成更快。
**不能比较两种编码各自随机遮盖的训练 loss 就宣称质量改善。**

## 中断与恢复

预算用完会保留 `checkpoint.pt` 和 `status.json`。只有同样的训练配置允许恢复，
尤其 `--steps` 是原定训练计划，不能改大后声称是同一条学习率曲线。
需要延长训练计划时，应为所有对照重新设定相同计划，或明确记录为另一组实验。

在控制 tmux 内、卡号与模型配置按原实验设置，以下示例恢复默认的单卡 relation2：

```bash
export CUDA_VISIBLE_DEVICES="$(nvidia-smi -i 3 --query-gpu=uuid --format=csv,noheader)"
python -m relation_diffusion.train \
  --data "$DATA_DIR" --output /填写原来的运行目录/relation2-diffusion-seed1234 \
  --codec relation2 --objective diffusion --steps 200 \
  --global-batch 48 --micro-batch 8 --max-seconds 600 --resume
```

恢复只续训这个模型，不自动恢复整个 campaign。之后可以单独调用 `evaluate.py`，
必须选择新的输出文件名。训练输出有 loss、实测 symbols/s、ETA、峰值 allocated
显存；ETA 来自本机实际 update 时间，不能提前保证六卡耗时。

## 已验证与尚未验证

本地覆盖可逆性、保护前缀/特殊码、NumPy/GPU 一致性、固定路径分布归一化与防标签泄漏、
空 mask 梯度、真实训练/存盘/精确续训、真实 checkpoint 的评估与 preflight。
另有双进程 CPU/Gloo 与单进程匹配测试，以及本地 GPU 的 BF16 训练和评估短测。
具体通过数量见 [VALIDATION.md](VALIDATION.md)。

按照本轮要求，没有在实验室服务器启动训练；六卡 NCCL/5090 的吞吐尚未实测。
本地验证使用测试语句而非正式 WikiText 训练，不能据此宣称关系扩散有语言质量优势。
