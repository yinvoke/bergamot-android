package io.github.yinvoker.bergamot.bench

import android.os.Debug
import android.system.Os
import android.system.OsConstants
import java.io.File
import org.json.JSONArray
import org.json.JSONObject

/**
 * Samples this process's memory (PSS) and CPU usage on a background thread
 * while a benchmark phase runs. cpuPct is percent of ONE core (can exceed 100
 * when multiple threads are busy).
 */
class MetricsSampler(private val intervalMs: Long = 250) {
    private data class Sample(val tMs: Long, val pssKb: Long, val cpuPct: Double)

    private val samples = ArrayList<Sample>()
    private var thread: Thread? = null
    @Volatile private var running = false

    private val clockTicksPerSec = Os.sysconf(OsConstants._SC_CLK_TCK).toDouble()

    private fun procJiffies(): Long {
        // /proc/self/stat fields 14+15 = utime+stime; comm can contain spaces,
        // so parse after the closing paren.
        val stat = File("/proc/self/stat").readText()
        val fields = stat.substringAfterLast(')').trim().split(' ')
        return fields[11].toLong() + fields[12].toLong()
    }

    fun start() {
        running = true
        thread = Thread {
            val t0 = System.nanoTime()
            var lastJiffies = procJiffies()
            var lastNanos = t0
            while (running) {
                try {
                    Thread.sleep(intervalMs)
                } catch (_: InterruptedException) {
                    break
                }
                val now = System.nanoTime()
                val jiffies = procJiffies()
                val dtSec = (now - lastNanos) / 1e9
                val cpuPct = if (dtSec > 0) (jiffies - lastJiffies) / clockTicksPerSec / dtSec * 100.0 else 0.0
                lastJiffies = jiffies
                lastNanos = now
                synchronized(samples) {
                    samples.add(Sample((now - t0) / 1_000_000, Debug.getPss(), cpuPct))
                }
            }
        }.apply { name = "metrics-sampler"; start() }
    }

    fun stopAndReport(): JSONObject {
        running = false
        thread?.interrupt()
        thread?.join(2000)
        val curve = JSONArray()
        var peakPssKb = 0L
        var cpuSum = 0.0
        synchronized(samples) {
            for (s in samples) {
                curve.put(
                    JSONObject()
                        .put("tMs", s.tMs)
                        .put("pssKb", s.pssKb)
                        .put("cpuPct", Math.round(s.cpuPct * 10) / 10.0)
                )
                if (s.pssKb > peakPssKb) peakPssKb = s.pssKb
                cpuSum += s.cpuPct
            }
            return JSONObject()
                .put("peakPssMb", Math.round(peakPssKb / 1024.0 * 10) / 10.0)
                .put("avgCpuPct", if (samples.isEmpty()) 0 else Math.round(cpuSum / samples.size * 10) / 10.0)
                .put("curve", curve)
        }
    }
}
