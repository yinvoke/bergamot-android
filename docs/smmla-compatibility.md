# SMMLA(i8mm)路径兼容性清单

以下芯片/设备已在真机上完整跑过 SMMLA 路径的测试集、门控检查与引擎哈希
回归,**全部通过,无异常行为**(崩溃、SIGILL、译文差异、回退失效均未出现)。
无 i8mm 的机型记录的是自动回退到 ruy 路径的行为。

| 设备 | SoC | 核心 | ISA | i8mm | 实际路径 | 状态 | 验证日期 |
|---|---|---|---|---|---|---|---|
| 小米 14(23127PN0CC,Android 16) | 骁龙 8 Gen 3(SM8650) | Cortex-X4 / A720 / A520 | Armv9.2 | 有 | SMMLA | 通过 | 2026-09-02 |
| 小米 12(2201123C,Android 13) | 骁龙 8 Gen 1(SM8450) | Cortex-X2 / A710 / A510 | Armv9.0 | 有 | SMMLA | 通过 | 2026-09-02 |
| 小米 10(umi,Android 13) | 骁龙 865(SM8250) | Cortex-A77 / A55 | Armv8.2 | 无 | ruy(自动回退) | 通过 | 2026-09-02 |
| Apple M3 Pro(macOS,CI 主机口径) | Apple M3 | — | Armv8.6+ | 有 | SMMLA | 通过 | 2026-09-02 |

验证内容与新设备验证流程见 `docs/perf-notes/i8mm-smmla.md`。
