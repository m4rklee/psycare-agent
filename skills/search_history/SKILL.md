# search_history

搜索当前会话的历史对话，理解短期上下文。

## 输入

- `query` 或 `user_input`：检索关键词，可为空。
- `limit` 或 `max_results`：最多返回多少条，默认 10。
- `context.history`：当前会话消息列表，支持 `AiMessage` 或 dict。
- `context.session_id`：可选会话 ID。

## 输出

- `status`：固定为 `success`。
- `answer`：格式化当前会话历史。
- `total_messages`：上下文中消息总数。
- `total_found`：匹配数量。
- `results`：匹配消息列表。
- `session_id`：当前会话 ID。

## 失败行为

不直接读取 Redis；若 `context.history` 为空，返回空结果，不抛异常。

## 边界

只搜索当前传入上下文中的短期历史。跨会话摘要请使用 `search_similar_cases`。
