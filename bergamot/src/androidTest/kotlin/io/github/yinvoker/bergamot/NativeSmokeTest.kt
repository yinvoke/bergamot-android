package io.github.yinvoker.bergamot

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Assume
import org.junit.Test
import org.junit.runner.RunWith

/**
 * 设备端最小冒烟:验证 .so 装载、JNI 绑定与引擎生命周期,不依赖模型文件。
 * 注意 marian 的进程级全局态:全程只创建一个 service。
 */
@RunWith(AndroidJUnit4::class)
class NativeSmokeTest {

    @Test
    fun serviceLifecycleAndBadModelRejection() {
        val service = NativeBridge.createService(1)
        assertNotEquals(0L, service)

        assertThrows(RuntimeException::class.java) {
            NativeBridge.loadModel(service, "models:\n  - /nonexistent/model.bin\n")
        }

        NativeBridge.destroyService(service)
    }

    /**
     * 真机端到端正典断言:固定句子必须译出固定文本(设备正典,SMMLA 与
     * ruy 两路径逐字节相同),并且两次调用逐字节一致(确定性)。需要模型:
     *   adb push models/enzh/ /data/local/tmp/models-enzh
     *   adb shell run-as io.github.yinvoker.bergamot.test sh -c \
     *     'mkdir -p files/models/enzh && cp /data/local/tmp/models-enzh/[the four model files] files/models/enzh/'
     * 没有模型时跳过(Assume),不算失败。
     */
    @Test
    fun canonicalSentenceWhenModelsPresent() = runBlocking {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = File(context.filesDir, "models/enzh")
        Assume.assumeTrue("no enzh model under ${dir.absolutePath}", dir.listFiles()?.any { it.name.endsWith(".bin") } == true)
        BergamotEngine(EngineConfig(threads = 1)).use { engine ->
            val source = "\"We now have 4-month-old mice that are non-diabetic that used to be diabetic,\" he added."
            val first = engine.translate(listOf(source), ModelFiles.fromDirectory(dir))
            val second = engine.translate(listOf(source), ModelFiles.fromDirectory(dir))
            assertEquals(CANONICAL_ZH, first.single())
            assertEquals(first, second)
        }
    }

    private companion object {
        // Mi 14 (8 Gen 3) and Mi 10 (865), SMMLA and ruy paths: identical bytes.
        const val CANONICAL_ZH = "他补充道:“我们现在有4个月大的小鼠,它们是非糖尿病,曾经患有糖尿病。”"
    }
}
