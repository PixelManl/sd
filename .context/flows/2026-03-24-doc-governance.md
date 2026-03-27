# 2026-03-24 Doc Governance

## Goal

把文档入口、上下文轨道和测试注册表收敛成固定治理结构，减少后续继续开发时的“找文档”和“找入口”成本。

## Decisions

- 增加公共文档门户 `docs/README.md`。
- 增加两份治理注册表：
  - `docs/project_test_registry.md`
  - `docs/env_regression_registry.md`
- `AGENTS.md` 明确切到“检测优先、生成并行支撑”。
- `.context/*/README.md` 增加内容边界说明，避免 flow / scratch / archive 混写。

## Implemented

- 文档门户已补齐。
- 测试注册表和环境回归注册表已补齐。
- `bolt/docs/README.md` 已改成主线文档索引入口。

## Next Step

- 后续每次主线变化时，先更新 `.context/flows/`，再决定是否需要同步到 `docs/` 或 `bolt/docs/`。
