# 模型框架图交付说明

本文件夹给出一张模型总图、三张创新模块图。`*.svg` 是**结构准确、可编辑的论文图稿**；同名 `*.png` 是便于预览和插入文稿的导出图。`AI生成草图/` 保留 imagegen 生成的论文风格探索稿，其中局部箭头可能不符合最终计算图；**论文正文请以顶层 SVG/PNG 为准**。

## 统一设计语言

- **珊瑚红：**RGB 当前帧及其特征；**青绿：**热红外及其特征；**蓝色：**融合特征与检测主流；**紫色虚线：**可靠性或控制信号；**灰色：**过去帧缓存。
- **⊕：**逐元素 Add，含保底残差；**⊗：**可靠性门控的逐元素乘法；**C：**Concat；**虚线箭头：**参数/权重控制，不代表图像或特征直接流动；**叠层片：**特征图。
- 图中 `P2/P3/P4` 表示三个尺度，`Q full` 表示不压缩的密集 Query，`K,V sparse` 表示压缩的 Key/Value，`M past` 仅含过去帧。输出 `Boxes_t` 采用 **RGB 坐标系**。

## 四张图要表达的计算逻辑

1. **Overall：**两路编码器及双模态候选汇合 → OCR 判断热红外对应是否可靠 → TPSI 在高分辨率特征上做非对称稀疏交互 → RGCM 有条件地使用历史候选 → 检测头预测 RGB 坐标的目标框 → 预测后更新缓存。RGB 的 P3/P4 作为检测头的粗尺度上下文；微小目标重点保留 P2。
2. **OCR：**RGB/热红外特征与双模态候选 → 粗视野映射 → 局部对应搜索 → 相关性分布与有效性 → 可靠性 `r` 和热红外候选；融合时 RGB 特征作为残差保底。热红外候选可在 RGB 很暗时提供种子，但必须落入可解释的 RGB 坐标与标注范围。
3. **TPSI：**完整 RGB 高分辨率 Query；热红外候选及局部/全局上下文经预算路由取稀疏 Key/Value；OCR 的可靠性控制采样和门控，输出仍与原 RGB 特征图同分辨率。
4. **RGCM：**当前融合特征与 `t−1…t−k` 历史候选匹配；运动、尺度、外观、可见性和 OCR 可靠性共同确定历史证据权重；仅可靠历史进入当前特征，预测后将当前候选写入 FIFO 缓存。**在线推理不访问未来帧。**

## 三个创新点在图中的一一对应

| 创新 | 图中主要符号 | 应在后续实验中证明 |
|---|---|---|
| **OCR：对应可靠性先于融合** | `FoV → Search → r/valid`；紫色控制箭头 | 真实错位、无重叠/单模态弱可见情况下，比强制配准融合减少误检和漏检 |
| **TPSI：微小目标保护的稀疏交互** | `Q full` 对 `K,V sparse`；高分辨率残差 | 微小目标 AP/召回率与实际延迟、显存的 Pareto 改善 |
| **RGCM：可靠性约束的因果记忆** | `M past → Match → Gate → ⊕` | 与同帧数普通时序融合相比，提升弱目标检测并减少错误传播 |

## 建议用于论文的精简图注

**Figure 1.** Overall architecture of the proposed reliability-guided RGB–T video tiny-object detector. OCR estimates cross-modal correspondence reliability, TPSI retains dense queries while sparsifying contextual key–value tokens, and RGCM selectively incorporates past-frame evidence. Predictions are made in RGB coordinates.

**Figure 2(a).** OCR: dual-modal seeds guide coarse-to-local correspondence search. Confidence, uncertainty and overlap validity produce a reliability weight for residual thermal fusion.

**Figure 2(b).** TPSI: full-resolution RGB queries interact with a reliability-aware sparse thermal/context key–value bank, preserving tiny-object localization positions.

**Figure 2(c).** RGCM: matched past candidates are filtered by motion, appearance, visibility and correspondence reliability before temporal fusion; memory is updated after prediction.

## 使用前的学术检查

- 这些图表示**拟研究架构**，尚不能代表已经训练、验证或达到指标。论文正式投稿时，须使图、方法公式、代码和消融实验完全一致。
- 图中“可靠性”是检测有用性的估计，不能在没有对应真值的真实数据上称作精确物理配准概率。
- 若实验表明某模块无独立收益，应从正式模型图和贡献声明中删除，而不是仅保留在图中。
