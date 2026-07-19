---
hide:
  - toc
---

# pyGWRx 中文手册

<div class="model-hero" markdown>

**面向空间异质性、时空关系与局部统计分析的 Python 研究工具。**

当前公开版本包含 **19 个模型族、174 个公开 API、45 个可运行示例**，并将模型、数值底座、诊断、绘图与数据接口组织在一套统一文档中。

[五分钟开始](../getting-started/quickstart.md){ .md-button .md-button--primary }
[选择模型](guides/model-selection.md){ .md-button }
[模型手册](models/index.md){ .md-button }
[API 与示例](guides/api-and-examples.md){ .md-button }

</div>

## 从你的研究问题开始

| 研究需求 | 建议起点 | 进一步考虑 |
|---|---|---|
| 连续变量的空间非平稳关系 | [GWR](models/gwr.md) | MGWR、RGWR、LCR-GWR |
| 计数或二分类响应 | [GWGLM](models/gwglm.md) | 分布族与暴露量诊断 |
| 逐观测时空变化 | [GTWR](models/gtwr.md) | SGTWR、MGTWR |
| 多阶段历史信息 | [STWR](models/stwr.md) | 阶段权重敏感性分析 |
| 局部多变量结构 | [GWPCA](models/gwpca.md)、[GWSS](models/gwss.md) | GWDA |
| 属性相似性与功能邻域 | [SGWR](models/sgwr.md) | LG-GWR、GR-GWR |

## 中文文档入口

<div class="pygx-model-groups">
  <article class="pygx-model-group">
    <h3>19 个模型手册</h3>
    <p>模型背景、数学原理、算法流程、参数、结果、诊断、完整代码和能力边界。</p>
    <div class="pygx-model-links"><a href="models/">进入模型手册</a></div>
  </article>
  <article class="pygx-model-group">
    <h3>功能使用指南</h3>
    <p>数据输入、坐标与距离、核函数、带宽、预测、诊断、绘图和 API 使用方式。</p>
    <div class="pygx-model-links"><a href="guides/">进入使用指南</a></div>
  </article>
  <article class="pygx-model-group">
    <h3>理论与原创模型</h3>
    <p>完整算法百科，以及 LG-GWR、GR-GWR 的专题理论资料与验证边界。</p>
    <div class="pygx-model-links"><a href="../theory/model-encyclopedia.zh/">算法百科</a><a href="../theory/lg-gwr-monograph.zh/">LG-GWR</a><a href="../theory/gr-gwr-monograph.zh/">GR-GWR</a></div>
  </article>
</div>

## 使用边界

!!! important "先读能力边界，再调用模型"
    - pyGWRx 采用一致的 `fit → inspect → diagnose → visualize` 使用方式，但当前不宣称完整符合 scikit-learn estimator contract。
    - MGWR 和 MGTWR 当前只提供校准位置结果，不支持尚未验证的独立新位置预测。
    - GWPCA 使用 `transform()`；GWSS 提供局部描述统计；BootstrapGWR 用于非平稳性推断；GWDA 是分类模型。
    - `mgwr` 和 `spglm` 仅用于可选的 GWGLM 外部参考对照测试，不参与正常拟合。
    - LGGWR 和 GRGWR 是原创研究模型，使用时应报告初始化、收敛、敏感性与验证范围。
