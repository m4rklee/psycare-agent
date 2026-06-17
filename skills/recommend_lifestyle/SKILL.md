# recommend_lifestyle

根据心理困扰提供生活方式建议，包括作息、饮食、运动、睡眠和求助安排。

## 输入

- `query`、`user_input` 或 `topic`：用户当前困扰或支持主题。

## 输出

- `status`：固定为 `success`。
- `categories`：建议类别，如 sleep、study_stress、relationship。
- `suggestions`：可执行建议列表。
- `answer`：格式化生活方式与校园支持建议。

## 失败行为

规则建议不依赖外部服务，通常不返回 unavailable。

## 边界

建议用于低风险心理支持；若出现自伤、伤人或即时危险，应优先寻求现实支持或紧急帮助。
