package io.github.yinvoker.bergamot

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
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
}
