# SMMLA GEMM 测试集

- `smmla_test.cpp`:正式测试集(ctest 目标 `smmla_test`,`-DBUILD_SMMLA_TEST=ON`):
  ruy 式形状族 + 页边界守卫 + int8 极值 + 打包器独立对拍 + one-hot +
  缓存/generation 语义 + 4 线程 + 与 ruy 逐字节对拍 + tile 覆盖断言;
  `SMMLA_TEST_REQUIRE_I8MM=1` 让无 i8mm 的静默跳过变成失败;
  `--killswitch-probe` / `--force-sigill` 见文件头
- `check-opcodes.sh`:i8mm 指令限定扫描(所有 smmla/ummla/usmmla 必须在
  smmla 命名空间的符号内);Apple objdump 需 `--mattr=+v9.2a`(脚本默认),
  NDK objdump 用 `MATTR=+i8mm`
- `ab-bench.cpp`:形状级性能 A/B(目标 `ab_bench`,同一开关,主机和 Android
  都编),ruy 对 SMMLA,引擎的 decode / encoder / shortlist 形状,结果逐字节
  对拍并给几何均值;真机钉簇跑并记 scaling_max_freq(B 缓冲全程保活,否则
  ruy 的 prepack 缓存会按复用地址命中陈旧 pack——这不是引擎 bug,引擎只
  缓存真常量权重)
- 引擎级回归:`tools/regress-hash.sh`(双路径 + 正典哈希表)
- 已验证的芯片/设备清单:`docs/smmla-compatibility.md`

主机(在仓库根目录):
```
cmake -B build-host -DCMAKE_BUILD_TYPE=Release -DSSPLIT_USE_INTERNAL_PCRE2=ON \
  -DCOMPILE_TESTS=OFF -DBUILD_SMMLA_TEST=ON
cmake --build build-host --target smoke smmla_test ab_bench
SMMLA_TEST_REQUIRE_I8MM=1 ctest --test-dir build-host --output-on-failure
tools/smmla-test/check-opcodes.sh build-host/tools/smoke/smoke "$(xcrun -f llvm-objdump)"
./build-host/tools/smmla-test/ab_bench
```
引擎内 A/B:同一 smoke 二进制,`BERGAMOT_NO_I8MM=1` 强制回退 ruy。
