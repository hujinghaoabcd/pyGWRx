# 类型、安全与发布策略

## `py.typed` 与 mypy

pyGWRx 在发行包中保留 `pygwrx/py.typed`，表示安装后的内联类型注解可被类型检查器读取。当前 CI 采用“严格类型岛”策略：对已经完成类型标准化的核心公共模块执行阻断式 mypy 检查，而不是把尚未完成迁移的旧模块用大量忽略规则伪装成全项目严格通过。

```bash
python -m mypy
```

新公共模块原则上必须立即进入严格检查范围；旧模块在完成签名、实现、测试和文档同步后逐步加入。

## 安全审查

根目录 `SECURITY.md` 规定了私密漏洞报告方式。发行环境执行：

```bash
python -m pip check
python -m pip_audit --strict --progress-spinner=off
cyclonedx-py environment --output-format JSON --output-file SBOM.cdx.json
```

其中 `pip-audit` 用于依赖漏洞审查，`SBOM.cdx.json` 为 CycloneDX 软件物料清单。检查通过只代表发行时解析出的环境，不代表未来不会出现新的依赖漏洞。

## 数据证据

`DATA_PROVENANCE.md` 固定数据的包版本、提交、日期、上游路径和处理步骤；`DATA_HASHES.sha256` 标识实际随发行包分发的字节。两者分别回答“来自哪里”和“当前文件是什么”，不能混为一项未经验证的上游一致性声明。

```bash
python tools/update_data_hashes.py
python tools/verify_data_provenance.py
```

## 发布

GitHub Actions 配置 Windows、Linux、macOS 与 Python 3.11–3.14 测试矩阵，并单独执行代码质量、覆盖率、参考测试、最低依赖、wheel/sdist 隔离安装、文档和安全检查。

TestPyPI 与 PyPI 使用 GitHub OIDC Trusted Publishing，不保存长期 API Token。`v0.1.2` 之类的标签必须与 `pygwrx.__version__` 完全一致。PyPI 只上传 wheel 和 sdist；`SHA256SUMS`、SBOM、验证报告和文档站点作为 GitHub Release 资产交付。
