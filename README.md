<div align="center">

# bergamot-android

**Firefox 同款翻译引擎的 Android 移植,推理完全离线**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Android-3DDC84?logo=android&logoColor=white)](#-构建)
[![minSdk](https://img.shields.io/badge/minSdk-28-blue)](#-范围与限制)
[![ABI](https://img.shields.io/badge/ABI-arm64--v8a-orange)](#-范围与限制)

[特性](#-特性) · [基准测试](#-基准测试) · [支持的语言模型](#-支持的语言模型) · [快速开始](#-快速开始) · [构建](#-构建) · [Roadmap](#-roadmap)

</div>

---

基于 [mozilla/translations](https://github.com/mozilla/translations) 中
Firefox 内置整页翻译所使用的 [Bergamot](https://browser.mt/) 引擎,
完成 Android 平台的 NDK 移植与 Kotlin 封装,并提供为 AAR。

## ✨ 特性

- **离线推理**:全程无网络请求,模型来自 Mozilla 官方(MPL-2.0)
- **质量**:COMET 领先 Google ML Kit 端侧翻译 13.5–17.5 分,真机实测见下
- **Kotlin suspend API**:批量翻译、pivot 中转、HTML 感知翻译
- **内存管理**:模型按需加载、空闲自动卸载,可挂 `onTrimMemory`

## 📊 基准测试

两台真机上的对比实测(COMET × 100,越高越好):

![小米 14 基准](docs/benchmark-mi14.png)

![小米 10 基准](docs/benchmark-mi10.png)

| 方向 | ML Kit | Bergamot |
|---|---|---|
| 英 → 中 COMET | 73.7 | **87.3** |
| 日 → 中 COMET | 69.8 | **87.3** |

更多语向的官方评测数据见 Mozilla 的
[评测面板](https://mozilla.github.io/translations/final-evals/?langpair=en-zh)。

Bergamot 的翻译质量明显优于 Google ML Kit,但时间和内存开销都显著
更高:同为单线程,耗时是 ML Kit 的 2.4–3.2 倍(老机型差距更大),
峰值内存约 2.4 倍(日→中双模型常驻时约 3.4 倍)。多线程能明显提速,
4 线程已接近 ML Kit 的速度,但内存随线程数增长,4 线程峰值约为单线程
的 2.5 倍。

ML Kit 的成绩几乎不随机型变化,新旧设备差距很小;Bergamot 是计算与
内存带宽密集型负载,性能更依赖 SoC:单线程下 8 Gen 3 比骁龙 865
英→中快 44%(24.8s vs 35.7s)、日→中快 53%(51.5s vs 78.8s)。

### 评测方法

| 维度 | 规则 |
|---|---|
| 测试集 | [FLORES-200](https://github.com/facebookresearch/flores) devtest 前 150 句。公开、英/日/中三语内容对齐、自带人工中文参考译文;两个引擎翻同一批句子,对同一份参考打分 |
| 质量 | [COMET](https://github.com/Unbabel/COMET)(`wmt22-comet-da` × 100):机器翻译评测的标准神经评分模型,输入源文、机翻与参考译文,与人工评价的相关性远高于 BLEU。另用 BLEU(中文分词)交叉验证,结论一致;评分在主机侧统一计算 |
| 速度 | 基准 app 内计时,150 句总耗时。ML Kit 逐句调用(它只有此模式),Bergamot 整批喂入,各按原生调用方式;均不含下载时间,Bergamot 计时含模型加载 |
| 内存 | 每 250ms 采样一次进程 PSS,取阶段峰值;每阶段独立记录 |
| 公平性 | 同批句子、同参考译文;日→中双方都经英语中转(ML Kit 内部中转,Bergamot 引擎内建 pivot);每种配置独立进程运行,互不干扰 |

复现:装上 `sample/` 基准 app,一条 adb 命令跑完一种线程配置(见下文;
marian 持有进程级全局状态,不同线程数须分进程各跑一次),产出 JSON,
含每阶段内存与 CPU 曲线。

## 🌍 支持的语言模型

`registry.json` 当前索引 **104 个模型、53 种语言**,全部以英语为轴:
51 种语言与英语互译,阿塞拜疆语仅英→阿、阿尔巴尼亚语仅阿→英。
任意两种非英语语言之间用 `translatePivot` 经英语中转(如日→中)。
模型由 Mozilla 随 Firefox 持续更新。`from` / `to` 为 `registry.json`
中的语向代码,可直接用于下载脚本(见[快速开始](#-快速开始))。

<details>
<summary>展开完整模型列表(104 个)</summary>

| 源语言 | 目标语言 | from | to | 版本 | 大小 |
|---|---|---|---|---|---|
| 英语 | 阿拉伯语 | `en` | `ar` | 2.2 | 36 MB |
| 阿拉伯语 | 英语 | `ar` | `en` | 2.2 | 37 MB |
| 英语 | 阿塞拜疆语 | `en` | `az` | 1.0 | 21 MB |
| 英语 | 保加利亚语 | `en` | `bg` | 2.0 | 36 MB |
| 保加利亚语 | 英语 | `bg` | `en` | 2.0 | 37 MB |
| 英语 | 孟加拉语 | `en` | `bn` | 1.0 | 21 MB |
| 孟加拉语 | 英语 | `bn` | `en` | 1.0 | 23 MB |
| 英语 | 波斯尼亚语 | `en` | `bs` | 2.0 | 36 MB |
| 波斯尼亚语 | 英语 | `bs` | `en` | 2.0 | 38 MB |
| 英语 | 加泰罗尼亚语 | `en` | `ca` | 2.0 | 37 MB |
| 加泰罗尼亚语 | 英语 | `ca` | `en` | 2.0 | 37 MB |
| 英语 | 捷克语 | `en` | `cs` | 2.0 | 36 MB |
| 捷克语 | 英语 | `cs` | `en` | 2.0 | 37 MB |
| 英语 | 丹麦语 | `en` | `da` | 1.0 | 22 MB |
| 丹麦语 | 英语 | `da` | `en` | 1.0 | 22 MB |
| 英语 | 德语 | `en` | `de` | 2.1 | 37 MB |
| 德语 | 英语 | `de` | `en` | 2.0 | 37 MB |
| 英语 | 希腊语 | `en` | `el` | 1.0 | 21 MB |
| 希腊语 | 英语 | `el` | `en` | 1.1 | 22 MB |
| 英语 | 西班牙语 | `en` | `es` | 2.1 | 37 MB |
| 西班牙语 | 英语 | `es` | `en` | 2.0 | 37 MB |
| 英语 | 爱沙尼亚语 | `en` | `et` | 2.0 | 36 MB |
| 爱沙尼亚语 | 英语 | `et` | `en` | 2.0 | 37 MB |
| 英语 | 巴斯克语 | `en` | `eu` | 2.0 | 36 MB |
| 巴斯克语 | 英语 | `eu` | `en` | 2.1 | 36 MB |
| 英语 | 波斯语 | `en` | `fa` | 1.1 | 22 MB |
| 波斯语 | 英语 | `fa` | `en` | 1.0 | 22 MB |
| 英语 | 芬兰语 | `en` | `fi` | 2.0 | 36 MB |
| 芬兰语 | 英语 | `fi` | `en` | 2.0 | 38 MB |
| 英语 | 法语 | `en` | `fr` | 2.0 | 37 MB |
| 法语 | 英语 | `fr` | `en` | 2.0 | 37 MB |
| 英语 | 加利西亚语 | `en` | `gl` | 2.1 | 35 MB |
| 加利西亚语 | 英语 | `gl` | `en` | 2.1 | 37 MB |
| 英语 | 古吉拉特语 | `en` | `gu` | 1.0 | 21 MB |
| 古吉拉特语 | 英语 | `gu` | `en` | 1.0 | 22 MB |
| 英语 | 希伯来语 | `en` | `he` | 1.0 | 21 MB |
| 希伯来语 | 英语 | `he` | `en` | 1.0 | 23 MB |
| 英语 | 印地语 | `en` | `hi` | 1.0 | 22 MB |
| 印地语 | 英语 | `hi` | `en` | 1.0 | 23 MB |
| 英语 | 克罗地亚语 | `en` | `hr` | 1.0 | 21 MB |
| 克罗地亚语 | 英语 | `hr` | `en` | 1.0 | 22 MB |
| 英语 | 匈牙利语 | `en` | `hu` | 2.0 | 36 MB |
| 匈牙利语 | 英语 | `hu` | `en` | 1.0 | 23 MB |
| 英语 | 印尼语 | `en` | `id` | 1.0 | 21 MB |
| 印尼语 | 英语 | `id` | `en` | 1.0 | 22 MB |
| 英语 | 冰岛语 | `en` | `is` | 2.0 | 36 MB |
| 冰岛语 | 英语 | `is` | `en` | 2.0 | 36 MB |
| 英语 | 意大利语 | `en` | `it` | 2.1 | 37 MB |
| 意大利语 | 英语 | `it` | `en` | 2.0 | 37 MB |
| 英语 | 日语 | `en` | `ja` | 2.3 | 50 MB |
| 日语 | 英语 | `ja` | `en` | 2.1 | 55 MB |
| 英语 | 卡纳达语 | `en` | `kn` | 1.0 | 21 MB |
| 卡纳达语 | 英语 | `kn` | `en` | 1.0 | 23 MB |
| 英语 | 韩语 | `en` | `ko` | 2.1 | 52 MB |
| 韩语 | 英语 | `ko` | `en` | 2.1 | 54 MB |
| 英语 | 立陶宛语 | `en` | `lt` | 2.1 | 36 MB |
| 立陶宛语 | 英语 | `lt` | `en` | 1.0 | 23 MB |
| 英语 | 拉脱维亚语 | `en` | `lv` | 2.1 | 36 MB |
| 拉脱维亚语 | 英语 | `lv` | `en` | 1.0 | 22 MB |
| 英语 | 马拉雅拉姆语 | `en` | `ml` | 1.0 | 21 MB |
| 马拉雅拉姆语 | 英语 | `ml` | `en` | 1.0 | 23 MB |
| 英语 | 马拉地语 | `en` | `mr` | 2.0 | 35 MB |
| 马拉地语 | 英语 | `mr` | `en` | 2.0 | 37 MB |
| 英语 | 马来语 | `en` | `ms` | 1.0 | 22 MB |
| 马来语 | 英语 | `ms` | `en` | 1.0 | 22 MB |
| 英语 | 挪威语(书面) | `en` | `nb` | 2.0 | 22 MB |
| 挪威语(书面) | 英语 | `nb` | `en` | 2.0 | 22 MB |
| 英语 | 荷兰语 | `en` | `nl` | 2.1 | 36 MB |
| 荷兰语 | 英语 | `nl` | `en` | 2.0 | 37 MB |
| 英语 | 波兰语 | `en` | `pl` | 2.1 | 36 MB |
| 波兰语 | 英语 | `pl` | `en` | 2.0 | 37 MB |
| 英语 | 葡萄牙语 | `en` | `pt` | 2.1 | 36 MB |
| 葡萄牙语 | 英语 | `pt` | `en` | 2.0 | 37 MB |
| 英语 | 罗马尼亚语 | `en` | `ro` | 1.0 | 22 MB |
| 罗马尼亚语 | 英语 | `ro` | `en` | 1.0 | 23 MB |
| 英语 | 俄语 | `en` | `ru` | 2.1 | 35 MB |
| 俄语 | 英语 | `ru` | `en` | 1.1 | 23 MB |
| 英语 | 斯洛伐克语 | `en` | `sk` | 2.1 | 36 MB |
| 斯洛伐克语 | 英语 | `sk` | `en` | 1.0 | 23 MB |
| 英语 | 斯洛文尼亚语 | `en` | `sl` | 2.1 | 36 MB |
| 斯洛文尼亚语 | 英语 | `sl` | `en` | 2.1 | 37 MB |
| 阿尔巴尼亚语 | 英语 | `sq` | `en` | 1.0 | 22 MB |
| 英语 | 塞尔维亚语 | `en` | `sr` | 2.0 | 35 MB |
| 塞尔维亚语 | 英语 | `sr` | `en` | 1.0 | 23 MB |
| 英语 | 瑞典语 | `en` | `sv` | 1.0 | 22 MB |
| 瑞典语 | 英语 | `sv` | `en` | 1.0 | 23 MB |
| 英语 | 泰米尔语 | `en` | `ta` | 2.0 | 35 MB |
| 泰米尔语 | 英语 | `ta` | `en` | 2.0 | 38 MB |
| 英语 | 泰卢固语 | `en` | `te` | 1.0 | 21 MB |
| 泰卢固语 | 英语 | `te` | `en` | 1.0 | 23 MB |
| 英语 | 泰语 | `en` | `th` | 2.0 | 36 MB |
| 泰语 | 英语 | `th` | `en` | 2.0 | 37 MB |
| 英语 | 土耳其语 | `en` | `tr` | 1.0 | 21 MB |
| 土耳其语 | 英语 | `tr` | `en` | 1.0 | 23 MB |
| 英语 | 乌克兰语 | `en` | `uk` | 2.2 | 36 MB |
| 乌克兰语 | 英语 | `uk` | `en` | 1.1 | 22 MB |
| 英语 | 乌尔都语 | `en` | `ur` | 2.0 | 35 MB |
| 乌尔都语 | 英语 | `ur` | `en` | 2.0 | 36 MB |
| 英语 | 越南语 | `en` | `vi` | 2.0 | 37 MB |
| 越南语 | 英语 | `vi` | `en` | 1.0 | 22 MB |
| 英语 | 中文(简体) | `en` | `zh-Hans` | 2.2 | 52 MB |
| 中文(简体) | 英语 | `zh-Hans` | `en` | 2.1 | 55 MB |
| 英语 | 中文(繁体) | `en` | `zh-Hant` | 2.0 | 49 MB |
| 中文(繁体) | 英语 | `zh-Hant` | `en` | 2.0 | 52 MB |

</details>

## 📁 仓库结构

```
engine/        Bergamot 引擎,vendor 自 mozilla/translations(来源与升级手顺见 engine/UPSTREAM.md)
patches/       对上游的全部本地改动,git 补丁形式存档
jni/           C++ 胶水层:批量进出,走 AsyncService
bergamot/      Android 库(Kotlin suspend API)→ AAR
tools/smoke    主机端 CLI,供 CI / 开发做正确性与性能冒烟(不随库发布)
sample/        基准测试 app:ML Kit vs Bergamot,内存/CPU 曲线,JSON 导出
registry.json  Mozilla 模型下载索引(每个方向的 URL / sha256 / 大小)
```

## 🚀 快速开始

### 引入 AAR

从 [Releases](https://github.com/yinvoke/bergamot-android/releases)
下载 AAR,放进 app 模块的 `libs/` 目录:

```kotlin
dependencies {
    implementation(files("libs/bergamot-v0.1.0.aar"))
    // 本地 AAR 不携带传递依赖,须自行声明:
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
}
```

### 获取模型

模型是 Mozilla 为 Firefox 发布的官方翻译模型(MPL-2.0),经
Firefox Remote Settings 分发;`registry.json` 是其索引快照,记录每个
语向的文件 URL、sha256 与大小。每个方向 3–4 个文件(模型、
SentencePiece 词表、lexical shortlist),合计 21–55 MB,下载到同一个
目录即可——`ModelFiles.fromDirectory` 按文件名识别,目录名随意:

```bash
python3 - <<'EOF'
import json, hashlib, urllib.request, pathlib
src, dst = 'en', 'zh-Hans'                    # 语向,取值见 registry.json
m = next(x for x in json.load(open('registry.json'))['models']
         if x['from'] == src and x['to'] == dst)
out = pathlib.Path('models/enzh'); out.mkdir(parents=True, exist_ok=True)
for f in m['files']:
    p = out / f['name']
    urllib.request.urlretrieve(f['url'], p)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == f['sha256'], f['name']
    print('ok', f['name'])
EOF
```

集成到 app 时同理:运行时下载到应用私有目录,按 sha256 校验后交给
`ModelFiles.fromDirectory`。

### 调用

```kotlin
val engine = BergamotEngine(EngineConfig(threads = 2))
val model = ModelFiles.fromDirectory(File(modelsDir, "enzh"))
val translated: List<String> = engine.translate(texts, model)          // suspend
// 日→中经英语中转,两个模型常驻:
engine.translatePivot(texts, jaEn, enZh)
engine.releaseAllModels()   // 例如挂在 onTrimMemory
```

**每个进程只能创建一个 `BergamotEngine`**(底层 marian 运行时持有
进程级全局状态)。

## 🔨 构建

仅支持 ARM 主机(Apple Silicon / ARM Linux)。依赖:JDK 17、
Android SDK、NDK r29、CMake 3.31.6。

```bash
./gradlew :bergamot:assembleRelease          # AAR
./gradlew :sample:assembleDebug              # 基准 app
cmake -B build-host -DCMAKE_BUILD_TYPE=Release -DSSPLIT_USE_INTERNAL_PCRE2=ON \
  -DCOMPILE_TESTS=OFF && cmake --build build-host --target smoke   # 主机 CLI
```

真机一键基准(结果写入 app files 目录的 JSON,含每阶段内存/CPU 曲线):

```bash
adb shell am start -n io.github.yinvoker.bergamot.bench/.MainActivity \
  --ez autorun true --ei threads 2
```

## ⚠️ 范围与限制

- 仅 arm64-v8a,minSdk 28,支持 16 KB page size。
- int8 矩阵乘在支持 dotprod 的 CPU 上走 ruy 的 SDOT 内核(运行时分发;
  cpuinfo 检测失败时回落到操作系统层检测)。

## 🗺️ Roadmap

- [x] 基础引擎 + NDK 构建,并完成真机测试
- [x] CI(构建与测试)
- [x] 发布 AAR 构件
- [ ] HTML 模式验证(`html = true` 已接通,待基准)
- [ ] 性能优化

## 📄 许可

本仓库自有代码(jni/、bergamot/、sample/、tools/、构建脚本)为 **MIT**。
`engine/` 内捆绑的第三方组件各按其自身许可分发(含 MPL-2.0 的
Bergamot 翻译层文件),见 [NOTICE](NOTICE) 与各 vendor 目录内的
LICENSE 文件。模型为 Mozilla 官方发布,MPL-2.0。

---

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=yinvoke/bergamot-android&type=Date)](https://www.star-history.com/#yinvoke/bergamot-android&Date)

</div>
