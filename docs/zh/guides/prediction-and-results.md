# 结果对象、预测与能力边界

pyGWRx 统一了 `fit()` 风格，但不同模型不是同一种估计器。读取结果前先确认模型属于回归、分类、变换、描述统计还是推断工具。

## 1. 拟合后常见字段

回归模型通常提供局部系数、截距、拟合值、残差、带宽、诊断和局部推断。字段名以当前 API 页面为准：

```python
model.fit(X, y, coords)
print(model.summary())
frame = model.to_frame()
print(frame.head())
```

`to_frame()` 适合与原观测 ID 或 GeoDataFrame 连接。不要仅凭属性名猜测数组形状。

## 2. 三种“预测”含义

1. **训练位置拟合值**：模型校准时得到的 `fitted_values_`。
2. **新位置局部再标定**：用新位置到训练样本的权重重新估计局部系数。
3. **系数插值**：把训练位置系数做空间插值；这不是 pyGWRx 标准预测语义，也不应与局部再标定混为一谈。

支持 `predict_result()` 的模型会返回包含预测、局部系数、坐标和可用不确定性的结果对象。

## 3. 当前重要边界

- GWR、GTWR、GWGLM 等支持相应的新位置局部计算。
- MGWR 与 MGTWR 当前主要用于校准位置的多尺度结果，不承诺独立新位置预测。
- GWPCA 使用 `transform()`，不是回归 `predict()`。
- GWDA 输出类别与概率。
- GWSS 是局部统计描述。
- BootstrapGWR 用于非平稳性检验。

这些边界在 19 个模型页面和能力矩阵中均有明确标注。

## 4. 预测验证

空间模型的测试集必须与训练位置有真实分离。报告 MAE/RMSE/R² 等指标时，同时说明空间划分、时间划分、缓冲距离和是否在测试位置重新标定。样本内拟合不能替代外部预测验证。

## 5. 导出

```python
result = model.to_frame()
result["obs_id"] = original_ids

# 地理导出时显式恢复 geometry 和 CRS
result_gdf = original_gdf[["obs_id", "geometry"]].merge(result, on="obs_id")
result_gdf.to_file("local_results.gpkg", layer="results", driver="GPKG")
```

导出前检查列名、坐标系、数组长度和结果对应的是校准位置还是查询位置。
