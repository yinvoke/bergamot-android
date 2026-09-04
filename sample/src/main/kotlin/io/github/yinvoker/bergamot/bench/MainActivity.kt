package io.github.yinvoker.bergamot.bench

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject

class MainActivity : Activity() {
    private lateinit var root: FrameLayout
    private val scope = CoroutineScope(Dispatchers.Main)
    private var backAction: (() -> Unit)? = null

    companion object {
        /** One engine per process (marian global state); set once a bench ran here. */
        private var benchRanInProcess = false
        /** Guards against activity recreation re-firing an autorun intent. */
        private var autorunConsumed = false
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        root = FrameLayout(this).apply { fitsSystemWindows = true }
        setContentView(root)
        // Screen-off scheduling throttles everything 4-5x (little-core bias,
        // measured on Mi 10); a benchmark app must keep the display alive.
        // KEEP_SCREEN_ON = the video-player mechanism (no permission needed);
        // show-when-locked + turn-screen-on make adb-triggered runs light the
        // display even when launched against a lockscreen.
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setShowWhenLocked(true)
        setTurnScreenOn(true)

        if (intent.hasExtra("export")) {
            // Salvage previously produced results into run-as-readable storage.
            getExternalFilesDir(null)?.listFiles()?.filter { it.name.startsWith("bench_result") }?.forEach {
                File(filesDir, it.name).writeText(it.readText())
            }
        }

        showMain()

        if (intent.hasExtra("autorun") && !autorunConsumed) {
            autorunConsumed = true
            startBench(intent.getIntExtra("threads", 1), intent.getIntExtra("workspace", 128))
        }
    }

    override fun onBackPressed() {
        backAction?.invoke() ?: super.onBackPressed()
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun swap(view: View, onBack: (() -> Unit)?) {
        root.removeAllViews()
        root.addView(view, ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT)
        backAction = onBack
    }

    // ---- main screen: threads picker + run + history ----

    private fun showMain() {
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(12), dp(16), dp(12))
        }
        val radios = RadioGroup(this).apply { orientation = RadioGroup.HORIZONTAL }
        val ids = HashMap<Int, Int>()
        for (t in listOf(1, 2, 4)) {
            val rb = RadioButton(this@MainActivity).apply { text = "${t}线程"; id = View.generateViewId() }
            ids[rb.id] = t
            radios.addView(rb)
            if (t == 1) radios.check(rb.id)
        }
        val runBtn = Button(this).apply { text = "RUN BENCHMARK" }
        val historyBtn = Button(this).apply { text = "历史记录" }
        val buttons = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            addView(runBtn, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(historyBtn, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        }
        val hint = TextView(this).apply {
            textSize = 12f
            text = "FLORES-150 · en→zh + ja→zh(pivot/seq)· 每次 RUN 使用全新进程(引擎全局态限制)\n结果自动存档,跑完立刻显示耗时/速度与 CPU、内存曲线"
        }
        col.addView(radios)
        col.addView(buttons)
        col.addView(hint)

        runBtn.setOnClickListener {
            val threads = ids[radios.checkedRadioButtonId] ?: 1
            if (benchRanInProcess) {
                // Fresh process required for a second engine: bounce.
                startActivity(
                    Intent(this, RestartActivity::class.java)
                        .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                        .putExtra("threads", threads)
                )
                Runtime.getRuntime().exit(0)
            } else {
                startBench(threads, 128)
            }
        }
        historyBtn.setOnClickListener { showHistory() }
        swap(ScrollView(this).apply { addView(col) }, null)
    }

    // ---- run screen: live log, then straight into the result view ----

    private fun startBench(threads: Int, workspaceMb: Int) {
        benchRanInProcess = true
        val logView = TextView(this).apply { setPadding(dp(12), dp(12), dp(12), dp(12)); textSize = 12f }
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(TextView(this@MainActivity).apply {
                text = "跑分中 · ${threads}线程 · 请勿切后台(worker 会落小核)"
                textSize = 14f
                setPadding(dp(12), dp(10), dp(12), 0)
            })
            addView(ScrollView(this@MainActivity).apply { addView(logView) },
                LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        }
        swap(col, null)
        scope.launch {
            try {
                val result = withContext(Dispatchers.Default) {
                    BenchRunner(this@MainActivity, threads, workspaceMb) { line ->
                        scope.launch { logView.append(line + "\n") }
                    }.run()
                }
                showResult(result, "本次结果", ::showMain)
            } catch (e: Exception) {
                logView.append("FATAL: $e\n")
            }
        }
    }

    // ---- result screen: header + per-phase stats and curves ----

    private fun showResult(json: JSONObject, title: String, back: () -> Unit) {
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(10), dp(14), dp(24))
        }
        col.addView(Button(this).apply {
            text = "← 返回"
            setOnClickListener { back() }
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        val dev = json.optJSONObject("device") ?: JSONObject()
        val n = json.optInt("n", 150)
        val when_ = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).format(Date(json.optLong("timestamp")))
        col.addView(TextView(this).apply {
            textSize = 15f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
            text = title
        })
        col.addView(TextView(this).apply {
            textSize = 12f
            text = "${dev.optString("model")} · ${dev.optString("soc")} · ${dev.optLong("totalRamMb")}MB RAM · $when_\n" +
                "$n 句 · 空载 PSS ${json.optJSONObject("baseline")?.optDouble("pssMb") ?: 0} MB"
        })

        val phases = json.optJSONArray("phases") ?: return
        for (i in 0 until phases.length()) {
            val p = phases.getJSONObject(i)
            val card = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setBackgroundColor(0xFFF3F3F3.toInt())
                setPadding(dp(12), dp(10), dp(12), dp(10))
            }
            val lp = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dp(10) }

            card.addView(TextView(this).apply {
                textSize = 13f
                setTypeface(typeface, android.graphics.Typeface.BOLD)
                text = p.optString("name")
            })

            if (p.has("error")) {
                card.addView(TextView(this).apply {
                    textSize = 12f
                    setTextColor(0xFFB00020.toInt())
                    text = p.optString("error")
                })
            } else {
                val ms = p.optLong("totalMs")
                val speed = if (ms > 0) n * 1000.0 / ms else 0.0
                val m = p.optJSONObject("metrics") ?: JSONObject()
                card.addView(TextView(this).apply {
                    textSize = 12f
                    text = String.format(
                        Locale.US, "耗时 %.1f s · %.1f 句/s · 峰值 PSS %.0f MB · 平均 CPU %.0f%%",
                        ms / 1000.0, speed, m.optDouble("peakPssMb"), m.optDouble("avgCpuPct")
                    )
                })
                val curve = m.optJSONArray("curve")
                if (curve != null && curve.length() >= 2) {
                    val t = ArrayList<Float>(curve.length())
                    val cpu = ArrayList<Float>(curve.length())
                    val pss = ArrayList<Float>(curve.length())
                    for (j in 0 until curve.length()) {
                        val s = curve.getJSONObject(j)
                        t.add(s.optLong("tMs").toFloat())
                        cpu.add(s.optDouble("cpuPct").toFloat())
                        pss.add((s.optLong("pssKb") / 1024f))
                    }
                    val chartLp = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(96))
                        .apply { topMargin = dp(6) }
                    card.addView(ChartView(this, "CPU", "%", 0xFF2F7DE1.toInt())
                        .apply { setData(t, cpu) }, chartLp)
                    card.addView(ChartView(this, "内存 PSS", "MB", 0xFF1E9E62.toInt())
                        .apply { setData(t, pss) },
                        LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(96))
                            .apply { topMargin = dp(6) })
                }
            }
            col.addView(card, lp)
        }
        swap(ScrollView(this).apply { addView(col) }, back)
    }

    // ---- history: records saved by every completed run ----

    private fun showHistory() {
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(10), dp(14), dp(24))
        }
        col.addView(Button(this).apply {
            text = "← 返回"
            setOnClickListener { showMain() }
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT))
        col.addView(TextView(this).apply {
            textSize = 15f; setTypeface(typeface, android.graphics.Typeface.BOLD); text = "历史记录"
        })

        val files = File(filesDir, "records").listFiles()?.sortedByDescending { it.name } ?: emptyList()
        if (files.isEmpty()) {
            col.addView(TextView(this).apply { textSize = 12f; text = "暂无记录——跑一次基准后自动存档于此。" })
        }
        for (f in files) {
            val row = TextView(this).apply {
                textSize = 13f
                setPadding(dp(12), dp(12), dp(12), dp(12))
                setBackgroundColor(0xFFF3F3F3.toInt())
                text = f.name.removePrefix("bench_").removeSuffix(".json")
                setOnClickListener {
                    try {
                        showResult(JSONObject(f.readText()), f.name, ::showHistory)
                    } catch (e: Exception) {
                        text = "${f.name}(损坏: $e)"
                    }
                }
                setOnLongClickListener { f.delete(); showHistory(); true }
            }
            col.addView(row, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { topMargin = dp(8) })
        }
        col.addView(TextView(this).apply {
            textSize = 11f; text = "点按查看 · 长按删除"; gravity = Gravity.CENTER
            setPadding(0, dp(8), 0, 0)
        })
        swap(ScrollView(this).apply { addView(col) }, ::showMain)
    }
}
