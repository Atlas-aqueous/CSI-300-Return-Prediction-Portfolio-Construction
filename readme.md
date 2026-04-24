# 代码说明

## 环境配置
- Python 3.10 及以上
- 依赖库：`pandas`、`numpy`、`scikit-learn`、`xgboost`、`joblib`
- 项目目录已按赛题要求组织为 `app/code`、`app/data`、`app/model`、`app/output`、`app/temp`
- 本版本为 `version_zero`，用于 A 赛段第一次提交

## 数据
- 使用公开可获取的沪深 300 历史行情数据，数据字段与赛题基线数据保持一致
- 正式推理仅使用 `app/data/train.csv`
- `app/data/train.csv` 包含提交日前可公开获取的历史行情数据；A 赛段第一次提交时，建议数据截止到 `2026-04-24`
- `app/data/test.csv` 仅作为目录结构占位文件，正式训练和推理均不依赖该文件
- 输入字段包含股票代码、日期、开盘、收盘、最高、最低、成交量、成交额、振幅、涨跌额、换手率、涨跌幅
- 本方案未使用任何收费数据或私有数据

## 预训练模型
- 未使用外部大模型或联网下载的预训练权重
- 模型文件由 `train.sh` 调用 `code/src/train.py` 在本地训练生成
- 深度学习部分采用本地可复现的 `MLPRegressor`，不依赖额外模型文件

## 算法

### 整体思路介绍
- 标签严格对齐赛题收益定义：`(T+5 开盘价 - T+1 开盘价) / T+1 开盘价`
- 训练阶段基于历史数据构造未来 5 日开盘收益标签，用于监督学习
- 预测层采用双模型融合：
  - `XGBoost` 学习树结构非线性关系
  - `MLPRegressor` 学习连续特征之间的组合关系
- 策略层不直接把模型分数当权重，而是额外加入质量因子、波动率惩罚、总仓位控制和单票上限控制
- 正式推理阶段只基于 `train.csv` 中最后一个交易日及其以前的历史数据进行预测，生成 `output/result.csv`

### 方法的针对性设计
- 针对“最多 5 只股票”的约束，组合层只保留评分最高的前 5 只候选
- 针对“不希望简单 all in”的要求，策略层默认控制总仓位，允许保留现金
- 单票权重上限设为 `35%`，尽量避免过度集中
- 当模型置信度不足时主动保留现金，允许部分空仓
- 若所有候选股票预期收益不为正，则允许完全空仓

### 特征工程
- 多窗口动量特征：`1/2/3/5/10/20` 日收益与开盘收益变化
- 量价特征：成交量变化、成交额/成交量、换手率均值
- 波动率特征：多窗口涨跌幅标准差、振幅、日内价差
- 市场相对强弱特征：相对市场平均涨跌幅、相对市场换手率
- 横截面特征：当日成交量、涨跌幅、换手率分位

### 模型结构
- 主模型：`XGBoost Regressor`
- 子模型：`MLPRegressor(hidden_layer_sizes=(128, 64, 32))`
- 融合分数由模型预测值、质量因子和风险惩罚共同组成，主要包括：
  - `XGBoost` 预测收益
  - `MLPRegressor` 预测收益
  - 质量因子
  - 波动率惩罚

### 投资分配策略
- 先按融合分数筛选最多 5 只股票
- 再用风险调整后的 alpha 重新分配权重，降低高波动股票占比
- 根据组合整体置信度决定总仓位，不追求机械满仓
- 单票权重不超过 `35%`
- 多余资金留作现金，对应赛题中的“权重和可小于 1”

### 损失函数
- `XGBoost` 使用 `reg:squarederror`
- `MLPRegressor` 使用 scikit-learn 默认回归损失
- 对绝对收益更大的样本追加更高训练权重，提升对高收益样本的识别能力

### 模型集成
- 训练阶段同时训练 `XGBoost` 和 `MLPRegressor`
- 推理阶段做模型融合，并叠加规则化投资策略

## 训练流程
1. 读取 `app/data/train.csv`
2. 标准化字段名并清洗数据类型
3. 为每只股票按时间顺序构造滚动特征
4. 根据 `T+1` 到 `T+5` 开盘收益构建监督标签
5. 使用训练数据中最近一段历史日期作为内部验证集，打印验证指标
6. 训练 `XGBoost` 与 `MLPRegressor` 两个子模型
7. 使用全量训练样本重新拟合最终模型
8. 保存模型到 `app/model/xgb_return_model.joblib`

## 推理流程
1. 读取 `app/data/train.csv`
2. 不读取、不依赖 `app/data/test.csv`
3. 以 `train.csv` 中最后一个交易日作为预测基准日
4. 基于预测基准日及其以前的历史行情构造最新截面特征
5. 加载 `app/model/xgb_return_model.joblib`；若模型文件不存在，则自动触发训练流程
6. 对候选股票预测未来收益分数，并叠加质量因子、波动率惩罚等组合规则
7. 输出不超过 5 只股票及其权重
8. 保存 `app/output/result.csv`，编码为 UTF-8，字段为 `stock_id,weight`

## 文件说明
- `app/code/src/featurework.py`：数据读取、标准化、特征工程、组合生成、结果保存
- `app/code/src/train.py`：双模型训练、训练过程信息打印与模型保存
- `app/code/src/test.py`：正式推理、策略分配与结果输出；当前版本仅使用 `train.csv`，不依赖 `test.csv`
- `app/init.sh`：初始化目录
- `app/train.sh`：训练入口
- `app/test.sh`：推理入口
- `app/data/test.csv`：占位文件，正式训练和推理不使用

## 运行方式

### 本机直接运行
```bash
cd app
python code/src/train.py
python code/src/test.py
```

### 使用脚本运行
```bash
cd app
sh init.sh
sh train.sh
sh test.sh
```

## 输出结果格式
`app/output/result.csv` 格式如下：

```csv
stock_id,weight
000001,0.25
000002,0.18
000003,0.12
```

要求满足：
- `stock_id` 为 6 位股票代码
- `weight` 为组合权重
- 股票数量不超过 5
- 股票代码不重复
- 权重总和不超过 1
- 未分配部分视为现金

## 复现说明
- 程序固定随机种子，便于从训练过程开始复现
- 正式预测结果由 `train.csv`、代码和训练生成的模型共同决定
- `test.csv` 的内容不会影响正式 `result.csv`
- 为保证复现一致性，建议在 Docker 中使用固定依赖版本运行
- 训练完成后会生成或更新 `app/model/xgb_return_model.joblib`
- 推理完成后会生成 `app/output/result.csv`

## 其他注意事项
- 复现和推理过程不需要联网
- 若模型文件不存在，`test.py` 会自动触发训练流程
- 中间明细会输出到 `app/temp/training_report.json`、`app/temp/prediction_details.csv` 和 `app/temp/prediction_meta.json`
- A 赛段第一次提交`result.csv` 表示基于 2026-04-24 及以前数据对后续持有窗口的预测组合

