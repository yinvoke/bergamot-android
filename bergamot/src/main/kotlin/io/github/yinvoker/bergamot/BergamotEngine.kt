package io.github.yinvoker.bergamot

import java.io.Closeable
import java.io.File
import java.util.concurrent.Executors
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext

/** Local files of one translation direction (Mozilla student model layout). */
data class ModelFiles(
    val model: File,
    val srcVocab: File,
    val trgVocab: File,
    val shortlist: File,
) {
    companion object {
        /** Directory holding model.*.bin, *vocab*.spm, lex.*.bin for one direction. */
        fun fromDirectory(dir: File): ModelFiles {
            fun pick(what: String, predicate: (String) -> Boolean): File =
                dir.listFiles()?.firstOrNull { predicate(it.name) }
                    ?: throw IllegalArgumentException("no $what file in $dir")
            val model = pick("model") { it.startsWith("model.") && it.endsWith(".bin") }
            val shortlist = pick("shortlist") { it.startsWith("lex.") && it.endsWith(".bin") }
            val srcVocab = pick("vocab") { it.contains("vocab") && it.endsWith(".spm") && !it.startsWith("trg") }
            val trgVocab = dir.listFiles()?.firstOrNull { it.name.startsWith("trgvocab") && it.name.endsWith(".spm") }
                ?: srcVocab // single shared vocab
            return ModelFiles(model, srcVocab, trgVocab, shortlist)
        }
    }

    internal fun toConfigYaml(workspaceMb: Int): String = """
        models:
          - ${model.absolutePath}
        vocabs:
          - ${srcVocab.absolutePath}
          - ${trgVocab.absolutePath}
        shortlist:
          - ${shortlist.absolutePath}
          - false
        beam-size: 1
        normalize: 1.0
        word-penalty: 0
        max-length-break: 128
        mini-batch-words: 1024
        workspace: $workspaceMb
        max-length-factor: 2.0
        skip-cost: true
        cpu-threads: 0
        quiet: true
        quiet-translation: true
        gemm-precision: int8shiftAlphaAll
        alignment: soft
    """.trimIndent()
}

class EngineConfig(
    /** Worker translation threads inside the engine (>=1). */
    val threads: Int = 1,
    /** Marian workspace per worker, MB. Smaller = less RAM, may cost speed. */
    val workspaceMb: Int = 128,
    
    
    /** Unload a model after this long without use. */
    val idleUnloadMillis: Long = 60_000,
)

/**
 * Bergamot engine with lazy model loading and idle-based unloading.
 *
 * All native work runs on one dedicated thread; [translate] and
 * [translatePivot] suspend until their batch completes. Cancellation is
 * cooperative at batch granularity: a single native batch cannot be
 * interrupted (mirror of the engine's own contract).
 */
class BergamotEngine(private val config: EngineConfig = EngineConfig()) : Closeable {

    private val executor = Executors.newSingleThreadExecutor { r -> Thread(r, "bergamot") }
    private val dispatcher = executor.asCoroutineDispatcher()

    private var service: Long = 0
    private val models = HashMap<String, LoadedModel>()

    private class LoadedModel(val handle: Long, var lastUsedAt: Long)

    /** Translate [texts] with the direction in [model]. */
    suspend fun translate(texts: List<String>, model: ModelFiles, html: Boolean = false): List<String> =
        withContext(dispatcher) {
            val handle = acquire(model)
            try {
                NativeBridge.translate(serviceHandle(), handle, texts.toTypedArray(), html).toList()
            } finally {
                touch(model)
                unloadIdle()
            }
        }

    /**
     * Pivot translation (e.g. ja->en->zh) with both models resident — fastest,
     * but peak memory is the sum of both. For a RAM-capped sequential pivot,
     * call [translate] twice and let idle-unload reclaim the first model.
     */
    suspend fun translatePivot(
        texts: List<String>,
        first: ModelFiles,
        second: ModelFiles,
        html: Boolean = false,
    ): List<String> = withContext(dispatcher) {
        val firstHandle = acquire(first)
        val secondHandle = acquire(second)
        try {
            NativeBridge.translatePivot(serviceHandle(), firstHandle, secondHandle, texts.toTypedArray(), html)
                .toList()
        } finally {
            touch(first)
            touch(second)
            unloadIdle()
        }
    }

    /** Release models regardless of idle deadline (hook for onTrimMemory). */
    fun releaseAllModels() {
        executor.execute {
            models.values.forEach { NativeBridge.destroyModel(it.handle) }
            models.clear()
        }
    }

    override fun close() {
        executor.execute {
            models.values.forEach { NativeBridge.destroyModel(it.handle) }
            models.clear()
            if (service != 0L) NativeBridge.destroyService(service)
            service = 0
        }
        executor.shutdown()
    }

    // ---- All below runs on the engine thread. ----

    private fun serviceHandle(): Long {
        if (service == 0L) service = NativeBridge.createService(config.threads)
        return service
    }

    private fun keyOf(model: ModelFiles) = model.model.absolutePath

    private fun acquire(model: ModelFiles): Long {
        serviceHandle()
        val loaded = models.getOrPut(keyOf(model)) {
            LoadedModel(NativeBridge.loadModel(serviceHandle(), model.toConfigYaml(config.workspaceMb)), System.nanoTime())
        }
        loaded.lastUsedAt = System.nanoTime()
        return loaded.handle
    }

    private fun touch(model: ModelFiles) {
        models[keyOf(model)]?.lastUsedAt = System.nanoTime()
    }

    private fun unloadIdle() {
        val deadline = System.nanoTime() - config.idleUnloadMillis * 1_000_000
        val iterator = models.entries.iterator()
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (entry.value.lastUsedAt < deadline) {
                NativeBridge.destroyModel(entry.value.handle)
                iterator.remove()
            }
        }
    }
}
