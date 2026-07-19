# API 与完整示例使用指南

当前正式公开面包含 **174 个 API 符号**，全部映射到 **45 个可运行示例**。API 页面负责精确契约，模型和功能指南负责解释，示例负责证明调用路径真实可运行。

## 1. 文档的三层结构

1. **模型/功能指南**：为什么使用、数学原理、适用场景、误用风险和结果解释。
2. **API Reference**：导入路径、函数签名、参数、返回值、属性和源码。
3. **Examples**：从项目根目录可直接运行的完整代码。

不要只看 API 签名就进行科学解释，也不要只复制示例而忽略输入和能力边界。

## 2. 示例目录

| 目录 | 数量 | 内容 |
|---|---:|---|
| `examples/models/` | 19 | 每个正式模型一个独立脚本 |
| `examples/core/` | 8 | 核、距离、求解器、指标、优化和带宽 |
| `examples/diagnostics/` | 5 | 诊断、推断、共线性、时间、权重和机制区 |
| `examples/plotting/` | 6 | 全部公开绘图函数 |
| `examples/io/` | 4 | 数据集、表格、GeoDataFrame 与持久化 |
| `examples/workflows/` | 3 | GWR 全流程、模型比较和时空流程 |

## 3. 运行方式

```bash
pip install -e ".[all,test]"
python examples/models/01_gwr.py
python examples/diagnostics/02_inference_and_collinearity.py
python examples/run_all.py
```

绘图输出位于 `examples/output/`。批量运行器为每个脚本启动隔离进程，并限制 BLAS/OpenMP 线程，减少长测试退出挂起。

## 4. API—示例一致性

```bash
python tools/generate_api_docs.py
python tools/generate_example_docs.py
python examples/validate_coverage.py
```

生成器会写出分组 API 页面、详细示例目录和 `API_COVERAGE.json/.csv`。当新增公开 API 却没有示例、旧符号残留或示例不再导入对应接口时，检查会失败。

## 5. 阅读建议

- 初次使用：中文模型页 → 模型示例 → API 页面。
- 开发功能：功能指南 → 对应分类示例 → 精确 API。
- 论文分析：模型原理 → 诊断指南 → 验证与报告清单。
- 排错：先运行最小示例，再逐项替换为自己的数据。

完整英文 API 位于 [API Reference](../../api/index.md)，详细示例目录位于 [Examples](../../examples/index.md)。
