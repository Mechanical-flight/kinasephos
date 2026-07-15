# KinasePhos 4.0

这是一个从零 clean-room 实现的磷酸化位点与激酶家族预测项目。输入为以 S/T/Y 为中心的
61-mer；冻结的 ESM2-650M 教师提供 1280 维表征，四层 Student Transformer 投影至 1280
维，并连接位点二分类头和 `CMGC / AGC / TK / CAMK` 四分类头。部署仅需 Student。

当前代码、测试和合成 CPU smoke 流程已经建立，但没有把历史 PPT 指标当作已复现结果。
已上传数据的来源和许可仍需进一步确认，因此原始数据库、教师权重、embedding cache、
checkpoint 均未进入 Git。

快速验证：

```bash
python -m pip install -e ".[dev]"
make test
make smoke
```

真实数据流程必须先做 accession、位点坐标、残基一致性和许可核验，再按 CD-HIT cluster
分组切分，确保 protein、cluster、site 不跨数据集。未注释候选位点只能称为“未观察到
磷酸化注释的候选负样本”，不能视为绝对阴性。

详见英文 README、`DATA_CARD.md`、`MODEL_CARD.md` 与 `docs/`。

