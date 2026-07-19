# 中文使用指南

中文指南用于解释模型和公共功能的科学含义、输入契约与使用边界；精确参数和返回值以自动生成的英文 API Reference 为准。

## 安装

pyGWRx 支持 Python 3.11–3.14，普通用户可直接从 PyPI 安装：

```bash
python -m pip install --upgrade pyGWRx
```

需要锁定当前版本时：

```bash
python -m pip install "pyGWRx==0.1.2"
```

可选功能按需安装：

```bash
python -m pip install "pyGWRx[ml]"
python -m pip install "pyGWRx[parquet]"
python -m pip install "pyGWRx[all]"
```

安装后可执行：

```bash
python -c "import pygwrx; print(pygwrx.__version__)"
python -m pip check
```

## 指南目录

- [模型选择](model-selection.md)
- [数据、坐标与输入规范](data-and-inputs.md)
- [核函数、带宽与距离](kernels-and-bandwidths.md)
- [结果对象、预测与能力边界](prediction-and-results.md)
- [诊断与推断](diagnostics.md)
- [API 与完整示例](api-and-examples.md)
- [可视化图谱](visualization.md)
- [19 个模型详细手册](../models/index.md)
- [完整算法百科](../../theory/model-encyclopedia.zh.md)
