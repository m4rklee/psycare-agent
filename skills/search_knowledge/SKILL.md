# search_knowledge

搜索校园心理健康知识库，查找情绪困扰、压力、睡眠、人际支持、校园心理资源等信息。

## 输入

- `query` 或 `user_input`：检索问题。
- `max_results`：最多返回多少条资料，默认 5。
- `context.db`：异步数据库会话。
- `context.knowledge_service`：当前项目的 `KnowledgeService`。

## 输出

- `status`：`success` 或 `unavailable`。
- `answer`：格式化后的知识库检索结果。
- `query`：原始查询。
- `total_found`：找到的资料数量。
- `results`：知识片段列表，包含 source/content/score。
- `source`：固定为 `knowledge_service`。

## 失败行为

缺少 `db` 或 `knowledge_service` 时返回 `status="unavailable"`，不抛异常。

## 边界

结果仅用于校园心理健康支持和内部 agent 辅助，不替代专业心理咨询、医学诊断或治疗。
