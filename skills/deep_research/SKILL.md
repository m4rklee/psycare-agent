# deep_research

深度研究心理健康主题，综合知识库、权威资源和证据说明。

## 输入

- `query` 或 `user_input`：研究问题。
- `max_results`：每轮检索和最终汇总的最大资料数，默认 3。
- `context.db`：异步数据库会话。
- `context.knowledge_service`：当前项目的 `KnowledgeService`。

## 输出

- `status`：`success` 或 `unavailable`。
- `answer`：格式化后的深度研究摘要。
- `queries`：实际执行的 2-3 个检索查询。
- `findings`：关键发现摘要。
- `confidence`：`medium` 或 `low`。
- `evidence_strength`：`中` 或 `弱`。
- `results`：去重后的知识片段。

## 失败行为

缺少 `db` 或 `knowledge_service` 时返回 `status="unavailable"`，不抛异常。

## 边界

本 skill 不进行网络搜索，不迁移外部医疗项目的 DeepResearchWorkflow；结果仅用于心理健康支持和内部 agent 辅助。
