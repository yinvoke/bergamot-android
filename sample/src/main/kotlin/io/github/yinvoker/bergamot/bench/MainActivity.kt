package io.github.yinvoker.bergamot.bench

import android.app.Activity
import android.os.Bundle
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : Activity() {
    private lateinit var logView: TextView
    private lateinit var runButton: Button
    private val scope = CoroutineScope(Dispatchers.Main)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            // Android 15+ enforces edge-to-edge for targetSdk 35+; without this
            // the button hides under the status bar.
            fitsSystemWindows = true
        }
        runButton = Button(this).apply { text = "RUN BENCHMARK" }
        logView = TextView(this).apply { setPadding(24, 24, 24, 24); textSize = 12f }
        root.addView(runButton)
        root.addView(ScrollView(this).apply {
            addView(logView)
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        setContentView(root)

        if (intent.hasExtra("export")) {
            // Salvage previously produced results into run-as-readable storage.
            getExternalFilesDir(null)?.listFiles()?.filter { it.name.startsWith("bench_result") }?.forEach {
                java.io.File(filesDir, it.name).writeText(it.readText())
                log("exported ${it.name}")
            }
        }
        if (intent.hasExtra("autorun")) runButton.post { runButton.performClick() }

        runButton.setOnClickListener {
            runButton.isEnabled = false
            log("benchmark started")
            scope.launch {
                try {
                    withContext(Dispatchers.Default) {
                        BenchRunner(this@MainActivity, intent.getIntExtra("threads", 1), intent.getIntExtra("workspace", 128)) { line -> scope.launch { log(line) } }.run()
                    }
                    log("ALL DONE")
                } catch (e: Exception) {
                    log("FATAL: $e")
                } finally {
                    runButton.isEnabled = true
                }
            }
        }
    }

    private fun log(line: String) {
        logView.append(line + "\n")
    }
}
