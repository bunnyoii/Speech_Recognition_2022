# 语音识别

同济大学 2022级 计算机科学与技术学院 软件工程专业 机器智能方向 语音识别课程作业

授课教师：沈莹

授课学期：2024-2025年度 秋季学期

# 任务: Viterbi Algorithm

有一周，老师布置了以下家庭作业：

| 星期一 | 星期二 | 星期三 | 星期四 | 星期五 |
| :---: | :---: | :---: | :---: | :---: |
| A | C | B | A | C |

那一周他的情绪曲线最可能是怎样的？请给出完整的计算过程。

我们希望分析学生的情绪状态变化趋势，即在这五天内最可能的情绪曲线。我们假设有三种可能的情绪状态：
1. Good
2. Neutral
3. Bad

情绪转移和观察概率已知，可以使用隐马尔可夫模型（HMM）和维特比算法来找到给定作业类型序列下最可能的情绪曲线。

# 模型参数设定

**状态**：情绪状态集 states = ['Good', 'Neutral', 'Bad']

**观测**：作业类型集 observations = ['A', 'B', 'C']

**转移概率矩阵**：表示各情绪状态之间的转换概率：

```python
transition_prob = {
    'Good': {'Good': 0.2, 'Neutral': 0.3, 'Bad': 0.5},
    'Neutral': {'Good': 0.2, 'Neutral': 0.2, 'Bad': 0.6},
    'Bad': {'Good': 0.0, 'Neutral': 0.2, 'Bad': 0.8}
}
```

**发射概率矩阵**：表示在特定情绪状态下观察到特定作业类型的概率：

```python
emission_prob = {
    'Good': {'A': 0.7, 'B': 0.2, 'C': 0.1},
    'Neutral': {'A': 0.3, 'B': 0.4, 'C': 0.3},
    'Bad': {'A': 0.0, 'B': 0.1, 'C': 0.9}
}
```

**初始状态概率**：各情绪状态的初始概率（均匀分布）：

```python
start_prob = {'Good': 1/3, 'Neutral': 1/3, 'Bad': 1/3}
```

# 解题步骤

**定义观测序列和初始条件**

观测序列 `obs_sequence = ['A', 'C', 'B', 'A', 'C']` 对应五天的作业类型。`viterbi` 和 `path` 表将分别存储每个时间步下各状态的最大概率和最优路径。

**初始化**

在第一个时间步 `t=0`：

对每个情绪状态，根据其初始概率和观察到的第一个作业 `A` 计算其最大概率。
例如，对于状态 `Good`，计算其概率为：

```scss
viterbi['Good', 0] = start_prob['Good'] * emission_prob['Good']['A'] = (1 / 3) * 0.7 ≈ 0.2333
```

依次计算每个状态的初始概率。

**迭代计算**

从 `t=1` 到 `t=4`，依次填充 `viterbi` 表的每一列：

对每个情绪状态，找到所有可能的前一状态，并计算从该状态转移到当前状态的最大概率。
例如，对于 `t=1` 且状态为 `Neutral`，考虑所有前一状态的可能性：
从 `Good` 转移到 `Neutral` 的概率为：
```scss
viterbi['Good', 0] * transition_prob['Good']['Neutral'] * emission_prob['Neutral']['C']
```
找到最大概率并更新 `viterbi` 和 `path` 表。

重复该过程，逐步填充所有时间步的 `viterbi` 表。

**回溯找到最优路径**

在最后一个时间步（t=4），找到最大概率的状态，回溯 path 表找到前一状态，依次回溯至 t=0，得到完整的最优情绪曲线。

# 结果与情绪曲线

**初始状态 (t=0)：**

在起始时间点 `t=0` ，教师的情绪状态“良好”具有最高概率 (0.233333)，其次是“中等” (0.100000)，而“糟糕”状态的概率为0。这表明教师在一开始的情绪更可能是良好或中等，而不太可能是糟糕。

**时间步 t=1, 观察值 C：**

根据观测值“C”，状态转移后，“糟糕”状态的概率变得最高 (0.105000)，这是从“良好”状态转移而来的。
“中等”状态的概率为0.021000（从“良好”状态转移而来），其次是“良好”状态的概率为0.004667（从“良好”状态转移而来）。
这一变化反映了教师在t=1时刻情绪有所转变，偏向于较差的情绪状态。

**时间步 t=2, 观察值 B：**

当观测值为“B”时，“中等”状态和“糟糕”状态的概率相等，为0.008400，两者都是从“糟糕”状态转移而来。
此外，“良好”状态的概率稍低，为0.000840，来自于从“中等”状态的转移。
这个时间步显示出情绪状态不再集中于一个状态，可能性在“中等”和“糟糕”状态之间均分。

**时间步 t=3, 观察值 A：**

在t=3，观察值变为“A”，此时“良好”状态的概率较高 (0.001176)，来源于从“中等”状态的转移。
“中等”状态的概率下降到0.000504，从“糟糕”状态转移而来，而“糟糕”状态的概率为0。
这一结果暗示出教师的情绪状态从“中等”逐渐向“良好”过渡。

**时间步 t=4, 观察值 C：**

最后一个时间步的观察值为“C”，此时“糟糕”状态的概率最高 (0.000529)，来自于“良好”状态的转移。
“中等”状态的概率为0.000106，而“良好”状态的概率最低，为0.000024。
最终的状态倾向于“糟糕”，表明在这一周的最后，教师的情绪状态逐渐趋于消极。

根据以上步骤得到的 `viterbi` 表和 `path` 表，学生的最可能情绪曲线为：

| 星期一 | 星期二 | 星期三 | 星期四 | 星期五 |
| :---: | :---: | :---: | :---: | :---: |
| Good | Neutral | Bad | Good | Neutral |

这个情绪曲线表示，学生在完成这周的作业时可能经历的情绪状态分别是 Good（星期一）、Neutral（星期二）、Bad（星期三）、Good（星期四）、和 Neutral（星期五）。

# 总结

根据以上分析，教师的情绪状态在一周内经历了从“良好”到“中等”，并逐渐转变为“糟糕”的趋势。维特比算法的计算表明，情绪状态的变化是由观察到的家庭作业类型驱动的，这为预测教师情绪的变化趋势提供了洞察。

# 附录

**代码运行结果**

Starting Viterbi Calculation:

Initialization:
t=0 | State=Good | Observation='A' | Initial Probability = Start Prob(0.33) * Emission(0.70) = 0.233333
t=0 | State=Neutral | Observation='A' | Initial Probability = Start Prob(0.33) * Emission(0.30) = 0.100000
t=0 | State=Bad | Observation='A' | Initial Probability = Start Prob(0.33) * Emission(0.00) = 0.000000

Time Step t=1 | Current Observation = 'C'
State=Good | Max Probability=0.004667 from previous state 'Good' (Transition Prob=0.20, Emission Prob=0.10)
State=Neutral | Max Probability=0.021000 from previous state 'Good' (Transition Prob=0.30, Emission Prob=0.30)
State=Bad | Max Probability=0.105000 from previous state 'Good' (Transition Prob=0.50, Emission Prob=0.90)

Time Step t=2 | Current Observation = 'B'
State=Good | Max Probability=0.000840 from previous state 'Neutral' (Transition Prob=0.20, Emission Prob=0.20)
State=Neutral | Max Probability=0.008400 from previous state 'Bad' (Transition Prob=0.20, Emission Prob=0.40)
State=Bad | Max Probability=0.008400 from previous state 'Bad' (Transition Prob=0.80, Emission Prob=0.10)

Time Step t=3 | Current Observation = 'A'
State=Good | Max Probability=0.001176 from previous state 'Neutral' (Transition Prob=0.20, Emission Prob=0.70)
State=Neutral | Max Probability=0.000504 from previous state 'Bad' (Transition Prob=0.20, Emission Prob=0.30)
State=Bad | Max Probability=0.000000 from previous state 'Bad' (Transition Prob=0.80, Emission Prob=0.00)

Time Step t=4 | Current Observation = 'C'
State=Good | Max Probability=0.000024 from previous state 'Good' (Transition Prob=0.20, Emission Prob=0.10)
State=Neutral | Max Probability=0.000106 from previous state 'Good' (Transition Prob=0.30, Emission Prob=0.30)
State=Bad | Max Probability=0.000529 from previous state 'Good' (Transition Prob=0.50, Emission Prob=0.90)

Completed Viterbi Table:

t=0 | State='Good' | Probability=0.233333
t=0 | State='Neutral' | Probability=0.100000
t=0 | State='Bad' | Probability=0.000000
t=1 | State='Good' | Probability=0.004667
t=1 | State='Neutral' | Probability=0.021000
t=1 | State='Bad' | Probability=0.105000
t=2 | State='Good' | Probability=0.000840
t=2 | State='Neutral' | Probability=0.008400
t=2 | State='Bad' | Probability=0.008400
t=3 | State='Good' | Probability=0.001176
t=3 | State='Neutral' | Probability=0.000504
t=3 | State='Bad' | Probability=0.000000
t=4 | State='Good' | Probability=0.000024
t=4 | State='Neutral' | Probability=0.000106
t=4 | State='Bad' | Probability=0.000529