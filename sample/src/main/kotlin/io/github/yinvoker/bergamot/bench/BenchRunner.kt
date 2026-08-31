package io.github.yinvoker.bergamot.bench

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.common.model.DownloadConditions
import com.google.mlkit.nl.translate.TranslateLanguage
import com.google.mlkit.nl.translate.Translation
import com.google.mlkit.nl.translate.TranslatorOptions
import io.github.yinvoker.bergamot.BergamotEngine
import io.github.yinvoker.bergamot.EngineConfig
import io.github.yinvoker.bergamot.ModelFiles
import java.io.File
import java.util.concurrent.TimeUnit
import org.json.JSONArray
import org.json.JSONObject

/**
 * One-tap benchmark: ML Kit vs Bergamot on the same fixed test set
 * (FLORES-200 devtest, first 150 sentences), en->zh and ja->zh.
 * Emits bench_result.json into the app's external files dir for host-side
 * scoring (COMET) and curve plotting.
 */
class BenchRunner(
    private val context: Context,
    private val bergamotThreads: Int,
    private val workspaceMb: Int,
    private val log: (String) -> Unit,
) {

    private val filesDir = context.getExternalFilesDir(null)!!

    private fun asset(name: String): List<String> =
        context.assets.open("bench/$name").bufferedReader().readLines().filter { it.isNotBlank() }

    private fun deviceInfo(): JSONObject {
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mem = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val soc = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else Build.HARDWARE
        return JSONObject()
            .put("model", Build.MODEL)
            .put("device", Build.DEVICE)
            .put("soc", soc)
            .put("cores", Runtime.getRuntime().availableProcessors())
            .put("totalRamMb", mem.totalMem / 1024 / 1024)
            .put("androidVersion", Build.VERSION.RELEASE)
            .put("sdk", Build.VERSION.SDK_INT)
    }

    /** Idle baseline (app + UI, no engine): what phase metrics subtract. */
    private fun sampleBaseline(): JSONObject {
        val pss = ArrayList<Long>()
        repeat(6) { pss.add(android.os.Debug.getPss()); Thread.sleep(250) }
        return JSONObject()
            .put("pssMb", Math.round(pss.average() / 1024.0 * 10) / 10.0)
            .put("pssMinMb", Math.round(pss.min() / 1024.0 * 10) / 10.0)
    }

    suspend fun run(): File {
        val baseline = sampleBaseline()
        val eng = asset("eng.txt")
        val jpn = asset("jpn.txt")
        val phases = JSONArray()

        // --- ML Kit (once; identical across bergamot thread configs) ---
        if (bergamotThreads == 1) {
            phases.put(mlkitPhase("mlkit-enzh", TranslateLanguage.ENGLISH, eng))
            phases.put(mlkitPhase("mlkit-jazh", TranslateLanguage.JAPANESE, jpn))
        }

        // --- Bergamot ---
        // External first (adb-push friendly on older Androids), internal as
        // fallback (run-as friendly where shell can't touch Android/data).
        val modelsDir = listOf(File(filesDir, "models"), File(context.filesDir, "models"))
            .firstOrNull { File(it, "enzh").listFiles()?.any { f -> f.name.endsWith(".bin") } == true }
            ?: File(filesDir, "models")
        val enzh = File(modelsDir, "enzh")
        val jaen = File(modelsDir, "jaen")
        // App-owned dirs: on Android 14+ adb-created dirs under Android/data are
        // unreadable by the app; the app creates them, adb pushes files into them.
        enzh.mkdirs(); jaen.mkdirs()
        val ready = enzh.listFiles()?.any { it.name.endsWith(".bin") } == true &&
            jaen.listFiles()?.any { it.name.endsWith(".bin") } == true
        if (!ready) {
            log("!! bergamot models missing under ${modelsDir.absolutePath} (adb push models/ there); skipping bergamot phases")
        } else {
            // One engine per process run: marian keeps process-global state and a
            // second AsyncService in the same process fails (known engine limit).
            val t = bergamotThreads
            val engine = BergamotEngine(EngineConfig(threads = t, workspaceMb = workspaceMb, idleUnloadMillis = Long.MAX_VALUE / 2))
            try {
                phases.put(bergamotPhase("bergamot-enzh-${t}t", engine) {
                    it.translate(eng, ModelFiles.fromDirectory(enzh))
                })
                engine.releaseAllModels()
                phases.put(bergamotPhase("bergamot-jazh-pivot-${t}t", engine) {
                    it.translatePivot(jpn, ModelFiles.fromDirectory(jaen), ModelFiles.fromDirectory(enzh))
                })
                engine.releaseAllModels()
                phases.put(bergamotPhase("bergamot-jazh-seq-${t}t", engine) {
                    // RAM-capped variant: one model resident at a time.
                    val english = it.translate(jpn, ModelFiles.fromDirectory(jaen))
                    it.releaseAllModels()
                    it.translate(english, ModelFiles.fromDirectory(enzh))
                })
            } finally {
                engine.close()
            }
        }

        val result = JSONObject()
            .put("baseline", baseline)
            .put("timestamp", System.currentTimeMillis())
            .put("device", deviceInfo())
            .put("n", eng.size)
            .put("phases", phases)
        val suffix = if (workspaceMb == 128) "" else "-w$workspaceMb"
        val out = File(filesDir, "bench_result_${bergamotThreads}t$suffix.json")
        out.writeText(result.toString())
        // Mirror into internal storage: on Android 14+ neither shell nor run-as
        // can read the external app dir, so this copy is what adb collects.
        File(context.filesDir, out.name).writeText(result.toString())
        log("done -> ${out.absolutePath}")
        return out
    }

    private fun mlkitPhase(name: String, sourceLang: String, texts: List<String>): JSONObject {
        log("[$name] starting")
        val sampler = MetricsSampler().also { it.start() }
        val phase = JSONObject().put("name", name).put("engine", "mlkit")
        try {
            val translator = Translation.getClient(
                TranslatorOptions.Builder()
                    .setSourceLanguage(sourceLang)
                    .setTargetLanguage(TranslateLanguage.CHINESE)
                    .build()
            )
            try {
                val dl0 = System.nanoTime()
                Tasks.await(translator.downloadModelIfNeeded(DownloadConditions.Builder().build()), 600, TimeUnit.SECONDS)
                phase.put("modelPrepMs", (System.nanoTime() - dl0) / 1_000_000)
                repeat(3) { Tasks.await(translator.translate(texts[it]), 120, TimeUnit.SECONDS) }

                val outputs = JSONArray()
                val perSentenceMs = JSONArray()
                val t0 = System.nanoTime()
                for (text in texts) {
                    val s0 = System.nanoTime()
                    val translated = Tasks.await(translator.translate(text), 120, TimeUnit.SECONDS)
                    perSentenceMs.put((System.nanoTime() - s0) / 1_000_000.0)
                    outputs.put(translated)
                }
                phase.put("totalMs", (System.nanoTime() - t0) / 1_000_000)
                phase.put("perSentenceMs", perSentenceMs)
                phase.put("outputs", outputs)
            } finally {
                translator.close()
            }
        } catch (e: Exception) {
            log("[$name] FAILED: $e")
            phase.put("error", e.toString())
        }
        phase.put("metrics", sampler.stopAndReport())
        log("[$name] finished")
        return phase
    }

    private suspend fun bergamotPhase(
        name: String,
        engine: BergamotEngine,
        block: suspend (BergamotEngine) -> List<String>,
    ): JSONObject {
        log("[$name] starting")
        val sampler = MetricsSampler().also { it.start() }
        val phase = JSONObject().put("name", name).put("engine", "bergamot").put("threads", bergamotThreads)
        try {
            val t0 = System.nanoTime()
            val outputs = block(engine)
            phase.put("totalMs", (System.nanoTime() - t0) / 1_000_000)
            phase.put("outputs", JSONArray().also { arr -> outputs.forEach { arr.put(it) } })
        } catch (e: Exception) {
            log("[$name] FAILED: $e")
            phase.put("error", e.toString())
        }
        phase.put("metrics", sampler.stopAndReport())
        log("[$name] finished")
        return phase
    }
}
