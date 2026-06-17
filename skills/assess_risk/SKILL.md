# assess_risk

评估心理风险等级线索，区分低、中、高、紧急风险，并提示现实求助边界。

## 输入

- `user_input` 或 `query`：用户当前表达。

## 输出

- `status`：固定为 `success`。
- `risk_level`：`low`、`medium` 或 `emergency`。
- `urgency`：紧急程度说明。
- `reasons`：触发风险等级的线索。
- `recommendation`：现实求助或自我支持建议。
- `answer`：格式化风险线索评估。

## 失败行为

规则评估不依赖外部服务，通常不返回 unavailable。

## 边界

结果仅用于心理风险线索整理，不替代正式风险评估；出现自伤、伤人或即时危险时应优先寻求现实支持或紧急帮助。
