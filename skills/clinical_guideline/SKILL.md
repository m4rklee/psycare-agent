# clinical_guideline

检索校园心理支持、危机干预原则和专业共识类资料。

## 输入

- `query` 或 `user_input`：需要检索的主题。
- `max_results`：最多返回多少条资料，默认 3。
- `context.db`：异步数据库会话。
- `context.knowledge_service`：当前项目的 `KnowledgeService`。

## 输出

- `status`：`success` 或 `unavailable`。
- `answer`：格式化后的校园心理支持原则资料。
- `query`：原始查询。
- `total_found`：找到的资料数量。
- `results`：知识片段列表。
- `guideline_type`：固定为 `campus_mental_health_support`。

## 失败行为

缺少 `db` 或 `knowledge_service` 时返回 `status="unavailable"`，不抛异常。

## 边界

这里的 guideline 指校园心理支持原则、危机干预原则和专业共识，不输出临床诊疗方案。
