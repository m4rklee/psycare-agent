# analyze_symptoms

分析情绪、认知、躯体反应和行为变化之间的模式关联。

## 输入

- `user_input` 或 `query`：用户当前表达。

## 输出

- `status`：固定为 `success`。
- `patterns`：识别到的心理困扰模式。
- `categories`：结构化类别列表。
- `follow_up_questions`：建议进一步了解的问题。
- `answer`：格式化模式分析结果。

## 失败行为

规则分析不依赖外部服务，通常不返回 unavailable。

## 边界

只做心理健康线索整理，不做医学诊断、心理障碍诊断或确定性结论。
