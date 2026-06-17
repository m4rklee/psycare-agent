# search_similar_cases

搜索相似历史案例或长期记忆摘要，辅助理解常见支持路径。

## 输入

- `query` 或 `user_input`：检索关键词，可为空。
- `limit` 或 `max_results`：最多返回多少条，默认 3。
- `context.long_memory`：长期记忆摘要列表，支持字符串或 dict。

## 输出

- `status`：固定为 `success`。
- `answer`：格式化相似历史摘要。
- `total_found`：匹配数量。
- `results`：匹配摘要列表。
- `source`：固定为 `context.long_memory`。

## 失败行为

不直接读取数据库或外部向量库；若 `context.long_memory` 为空，返回空结果，不抛异常。

## 边界

仅用于跨会话摘要辅助，不代表真实案例诊断或治疗建议。
