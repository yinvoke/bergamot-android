package io.github.yinvoker.bergamot.bench

import android.app.Activity
import android.content.Intent
import android.os.Bundle

/**
 * Process bounce for "run again": marian keeps process-global state (spdlog
 * registry, backend globals), so a second engine in the same process aborts.
 * This activity lives in its own :bounce process; the main process exits,
 * then this relaunches MainActivity fresh and exits itself.
 */
class RestartActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val target = Intent(this, MainActivity::class.java)
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
            .putExtra("autorun", true)
            .putExtra("threads", intent.getIntExtra("threads", 1))
            .putExtra("workspace", intent.getIntExtra("workspace", 128))
        window.decorView.postDelayed({
            startActivity(target)
            finishAndRemoveTask()
            Runtime.getRuntime().exit(0)
        }, 400)
    }
}
