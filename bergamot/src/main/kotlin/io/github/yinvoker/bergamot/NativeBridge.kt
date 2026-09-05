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

    /**
     * Hand a model back to [service] and reclaim what it holds on the model's
     * behalf: each worker's cached reference and last batch, the aggregate
     * queue's reference, and every worker's GEMM weight-packing cache. Just
     * dropping the handle frees none of those until the service itself dies.
     *
     * Blocks until every worker acknowledges (bounded by one in-flight batch).
     * Returns true when the model was actually destroyed; false means a
     * request was still in flight and it dies when that finishes.
     */
    external fun releaseModel(service: Long, model: Long): Boolean
    external fun translate(service: Long, model: Long, texts: Array<String>, html: Boolean): Array<String>
    external fun translatePivot(
        service: Long,
        first: Long,
        second: Long,
        texts: Array<String>,
        html: Boolean,
    ): Array<String>
}
