# Latency Benchmark Report

**Queries**: 100 successful out of 100 total
**Guardrails**: 37 fully generated, 63 refused (off-topic/unsafe/ungrounded)
**Target**: Deployed API

## Stage-by-Stage Latency

| Stage      |   P50 (ms) |   P70 (ms) |   P100 (ms) |   Mean (ms) |
|------------|------------|------------|-------------|-------------|
| Retrieval  |       43.9 |       49.3 |       166.3 |        47.8 |
| Generation |     1512.8 |     1517   |      1527.9 |      1372.7 |
| Guardrails |        0.9 |        1.3 |        15.8 |         1.3 |
| End-to-End |     1557.8 |     1566.4 |      1695.6 |      1422.3 |

## Key Findings

- **Retrieval P50: 43.9 ms** ✓ Under 200ms target
- **End-to-End P50: 1557.8 ms**
- End-to-end latency includes STT, retrieval, LLM generation, and guardrail checks.
- The retrieval component (chunk lookup + vector search) is the part under direct control
  and is the component the <200ms requirement targets.

## Slowest Retrieval Queries (P100 Investigation)

The top 5 slowest queries for the retrieval stage were:
- Query #50 | Retrieval: 166.3ms | Total: 1695.6ms | Language: hi
- Query #83 | Retrieval: 115.4ms | Total: 1635.9ms | Language: hi
- Query #19 | Retrieval: 106.4ms | Total: 1626.3ms | Language: hi
- Query #96 | Retrieval: 85.2ms | Total: 1598.8ms | Language: hi
- Query #95 | Retrieval: 81.1ms | Total: 1608.0ms | Language: hi
