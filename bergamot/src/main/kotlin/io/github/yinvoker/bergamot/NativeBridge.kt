package io.github.yinvoker.bergamot

/**
 * Raw JNI surface. Blocking, batch-in/batch-out; no threading or lifecycle
 * here — [BergamotEngine] owns both. Handles are opaque native pointers.
 */
internal object NativeBridge {
    init {
        System.loadLibrary("bergamot")
    }

    external fun createService(workers: Int): Long
    external fun destroyService(service: Long)
    external fun loadModel(service: Long, configYaml: String): Long
    external fun destroyModel(model: Long)
    external fun translate(service: Long, model: Long, texts: Array<String>, html: Boolean): Array<String>
    external fun translatePivot(
        service: Long,
        first: Long,
        second: Long,
        texts: Array<String>,
        html: Boolean,
    ): Array<String>
}
