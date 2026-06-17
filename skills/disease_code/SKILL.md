# disease_code

心理健康相关分类或风险类别占位，用于后续扩展结构化分类能力。

## 输入

- `query`、`user_input` 或 `topic`：需要分类的用户表达或主题。

## 输出

- `status`：固定为 `success`。
- `code`：内部心理健康分类码，如 `PSY-SAFETY`、`PSY-STRESS-SLEEP`、`PSY-MOOD`。
- `category`：分类名称。
- `coding_system`：固定为 `campus_mental_health_internal`。
- `diagnostic`：固定为 `false`。
- `answer`：格式化分类说明。

## 失败行为

规则分类不依赖外部服务，通常不返回 unavailable。

## 边界

保留 `disease_code` 名称是为了兼容 agent prompt；本 skill 不查询 ICD，不输出疾病编码，不做诊断。
