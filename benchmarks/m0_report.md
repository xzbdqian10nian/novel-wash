# M0 4 模型金标盲测对比报告

> 注：本报告为开发期历史盲测原始记录，表中 `standard` / `dedicated` 为脱敏后的测试通道命名；实际使用只需配置单一 `NOVEL_WASH_BASE_URL` 与 `NOVEL_WASH_API_KEY` 即可运行。

- 纯代码预检：独立群号汤清零 `11/11`，干净难负例零误伤 `16/16`，进入模型评测共 `28` 行。

## 单模型表现（带 ⟦编号+字⟧ 受限二分类协议）

| 模型 | 测试通道 | 精确率 (P) | 召回率 (R) | F1 | 正常字误删数 (FP / Tier-C FP) | 漏删 (FN) | JSON 首轮全过 | 耗时 | 实测费用 (CNY) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `deepseek-v4.1-flash-1017` | `standard` | 0.8182 | 0.9942 | 0.8976 | 38 (38) | 1 | 3/3 | 31.03s | ¥0.000505 |
| `glm-5.3-flash` | `standard` | 0.9657 | 0.9826 | 0.9741 | 6 (6) | 3 | 3/3 | 72.71s | ¥0.000443 |
| `minimax-m3.1-flash` | `dedicated` | 0.9583 | 0.9360 | 0.9471 | 7 (7) | 11 | 3/3 | 54.17s | ¥0.002619 |
| `hy4-preview` | `standard` | 0.9162 | 0.9535 | 0.9345 | 15 (15) | 8 | 3/3 | 30.54s | ¥0.001255 |

## 跨家族 Tier-C 双审组合效果（经 `verify()` 逐点门禁）

| 主力 + 辅助模型 | 精确率 (P) | 召回率 (R) | F1 | 最终误删 (FP) | 拦截进 review 队列 |
|---|---:|---:|---:|---:|---:|
| `deepseek-v4.1-flash-1017+glm-5.3-flash` | 0.9657 | 0.9826 | 0.9741 | 6 | 34 |
| `deepseek-v4.1-flash-1017+minimax-m3.1-flash` | 0.9704 | 0.9535 | 0.9619 | 5 | 40 |
| `deepseek-v4.1-flash-1017+hy4-preview` | 0.9330 | 0.9709 | 0.9516 | 12 | 30 |
| `hy4-preview+glm-5.3-flash` | 0.9643 | 0.9419 | 0.9529 | 6 | 11 |
| `hy4-preview+minimax-m3.1-flash` | 0.9634 | 0.9186 | 0.9405 | 6 | 15 |
| `hy4-preview+deepseek-v4.1-flash-1017` | 0.9318 | 0.9535 | 0.9425 | 12 | 3 |
| `minimax-m3.1-flash+glm-5.3-flash` | 0.9817 | 0.9360 | 0.9583 | 3 | 4 |
| `minimax-m3.1-flash+deepseek-v4.1-flash-1017` | 0.9699 | 0.9360 | 0.9527 | 5 | 2 |

## 详细费用账本
```json
{
  "calls": 12,
  "fallbacks": 0,
  "total_cost_cny": 0.004822,
  "by_route": {
    "standard/deepseek-v4.1-flash-1017": {
      "calls": 3,
      "prompt_tokens": 4039,
      "cached_tokens": 0,
      "completion_tokens": 1097,
      "reasoning_tokens": 0,
      "cost_cny": 0.000505,
      "latency_s": 15.532
    },
    "standard/glm-5.3-flash": {
      "calls": 3,
      "prompt_tokens": 4110,
      "cached_tokens": 0,
      "completion_tokens": 4091,
      "reasoning_tokens": 2663,
      "cost_cny": 0.000443,
      "latency_s": 63.731
    },
    "dedicated/minimax-m3.1-flash": {
      "calls": 3,
      "prompt_tokens": 4440,
      "cached_tokens": 384,
      "completion_tokens": 7162,
      "reasoning_tokens": 6410,
      "cost_cny": 0.002619,
      "latency_s": 54.169
    },
    "standard/hy4-preview": {
      "calls": 3,
      "prompt_tokens": 4168,
      "cached_tokens": 0,
      "completion_tokens": 936,
      "reasoning_tokens": 0,
      "cost_cny": 0.001255,
      "latency_s": 16.689
    }
  }
}
```
