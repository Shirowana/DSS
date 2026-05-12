# Gradient Importance Notes

## 背景

当前 DSS stage1 需要为候选位置构造一个重要性分数，然后按该分数筛选 elite。

我们最初使用的分数是：

```text
score_abs_mean = abs(mean(g))
```

其中 `g` 表示同一候选位置在若干个 step 上收集到的梯度。

这个定义的优点是：

- 偏好“净更新方向稳定”的参数
- 如果一个参数在多个 step 上始终朝相同方向推动 loss，它的分数会高

缺点是：

- 很容易被 step 间的正负震荡抵消
- 即使某个参数每一步的梯度幅值都很大，只要方向经常翻转，`abs(mean(g))` 也可能很小


## 三种值得比较的量

### 1. `abs(mean(g))`

```text
score_abs_mean = abs(mean(g))
```

含义：

- 看的是“净方向”
- 更像是在问：这个参数在这一小段窗口里，是否有一致的更新趋势

优点：

- 更偏好方向稳定的参数
- 不容易选到来回震荡但净效果很小的参数

缺点：

- 容易错杀“高敏感但方向有冲突”的参数
- 在多任务/多样本混合训练里，容易出现抵消


### 2. `mean(abs(g))`

```text
score_mean_abs = mean(abs(g))
```

含义：

- 看的是“平均强度”
- 更像是在问：这个参数是否经常被当前任务强烈调用

优点：

- 不会被正负抵消
- 与很多梯度驱动的 sparse/PEFT 方法更接近
- 更贴近“梯度敏感度”直觉

缺点：

- 会偏爱方向不稳但幅值很大的参数
- 这些参数未必适合长期稳定训练


### 3. `mean(abs(g)) * consistency`

推荐构造一个折中量：

```text
consistency = abs(mean(g)) / (mean(abs(g)) + eps)
score_consistent = mean(abs(g)) * consistency
```

化简后它与 `abs(mean(g))` 在数值上接近，但从解释上更清晰，因为它显式拆成了：

- 强度：`mean(abs(g))`
- 一致性：`abs(mean(g)) / mean(abs(g))`

解释：

- `consistency ≈ 1`
  - 说明梯度方向很稳定
- `consistency ≈ 0`
  - 说明梯度正负强烈抵消

这个拆法的价值在于：

- 即使最终仍然使用 `abs(mean(g))` 作为排序分数，我们也能额外记录：
  - 这个参数是因为“强度大且稳定”被选中
  - 还是因为“强度小但偶然净方向大”被选中


## 我当前推荐比较的量

如果后面要做系统对照，我推荐比较下面三种：

1. `score_abs_mean = abs(mean(g))`
2. `score_mean_abs = mean(abs(g))`
3. `score_consistent = mean(abs(g)) * consistency`

其中：

- `score_abs_mean`
  - 是当前方法的原型
- `score_mean_abs`
  - 是更稳的梯度强度指标
- `score_consistent`
  - 是兼顾强度和方向一致性的折中版本


## 为什么当前 zero-threshold 排查里，`mean(abs(g))` 仍然是 0

这次对两个异常层：

- `model.layers.10.mlp.up_proj`
- `model.layers.12.mlp.up_proj`

做了临时调试，观察了三段量：

1. hook 阶段
   - `grad_delta_abs_mean`
   - `grad_delta_abs_max`
   - `candidate_grads_abs_mean`
   - `candidate_grads_abs_max`
2. collect 阶段
   - `grad_cache_abs_mean`
   - `grad_cache_abs_max`
3. refresh 阶段
   - `abs(mean(g))`
   - `mean(abs(g))`
   - `max(abs(g))`

实验结论是：

- 在 hook 阶段，这两个层的 `grad_delta` 就已经是 0
- 因此 `candidate_grads` 也是 0
- collect 阶段累计的自然也还是 0
- 到 refresh 阶段，无论是：
  - `abs(mean(g))`
  - `mean(abs(g))`
  - `max(abs(g))`
  都还是 0

这说明：

- 这次 zero-threshold 现象**不是**因为 `abs(mean(g))` 的正负抵消
- 也**不是**因为我们用错了统计量
- 更早的问题是：这两个层在这些 step / 这些 batch 上，整条 `delta_weight` 分支拿到的梯度本身就是 0

因此：

- 把 `abs(mean(g))` 改成 `mean(abs(g))`
  - **并不能自动修复这次已经观察到的 0 梯度现象**
- 它能解决的是：
  - “非零梯度但被正负抵消后分数塌掉”的问题
- 它不能解决的是：
  - “从 hook 开始原始梯度就是 0”的问题


## 当前阶段的判断

所以要分清两类问题：

### 问题 A：统计量是否稳健

这个问题上，`mean(abs(g))` 或显式分解出的 `consistency` 都是值得尝试的。

### 问题 B：某些层 candidate 分支在某些 step 上是否根本拿不到梯度

这个问题上，改统计量无效，因为输入给统计量的就是全 0。

因此后续实验里要分别处理：

- 一方面比较不同 score 的筛选效果
- 另一方面继续排查为什么某些 `up_proj` 模块会出现整段时间的 `grad_delta = 0`

