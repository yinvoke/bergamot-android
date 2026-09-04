package io.github.yinvoker.bergamot.bench

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.view.View

/**
 * Minimal time-series chart: polyline over a light grid, peak value and
 * duration labelled. No dependencies, draws whatever [setData] provides.
 */
class ChartView(
    context: Context,
    private val label: String,
    private val unit: String,
    lineColor: Int,
) : View(context) {

    private var ts = FloatArray(0)
    private var vs = FloatArray(0)
    private var maxV = 1f
    private var durMs = 1f

    private val line = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = lineColor; style = Paint.Style.STROKE; strokeWidth = 3f
    }
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = (lineColor and 0x00FFFFFF) or 0x22000000; style = Paint.Style.FILL
    }
    private val grid = Paint().apply { color = 0x1F000000; strokeWidth = 1f }
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF444444.toInt(); textSize = 11f * resources.displayMetrics.density
    }

    fun setData(timesMs: List<Float>, values: List<Float>) {
        ts = timesMs.toFloatArray()
        vs = values.toFloatArray()
        maxV = (vs.maxOrNull() ?: 1f).coerceAtLeast(1f)
        durMs = (ts.lastOrNull() ?: 1f).coerceAtLeast(1f)
        invalidate()
    }

    override fun onDraw(c: Canvas) {
        val d = resources.displayMetrics.density
        val padL = 8f * d
        val padR = 8f * d
        val padT = 22f * d
        val padB = 18f * d
        val w = width - padL - padR
        val h = height - padT - padB
        if (w <= 0 || h <= 0) return

        for (i in 0..3) {
            val y = padT + h * i / 3f
            c.drawLine(padL, y, padL + w, y, grid)
        }
        if (ts.size >= 2) {
            val path = Path()
            val area = Path()
            for (i in ts.indices) {
                val x = padL + w * ts[i] / durMs
                val y = padT + h * (1f - vs[i] / maxV)
                if (i == 0) { path.moveTo(x, y); area.moveTo(x, padT + h); area.lineTo(x, y) }
                else { path.lineTo(x, y); area.lineTo(x, y) }
            }
            area.lineTo(padL + w * ts.last() / durMs, padT + h)
            area.close()
            c.drawPath(area, fill)
            c.drawPath(path, line)
        }
        c.drawText("$label · 峰值 ${fmt(maxV)}$unit", padL, 15f * d, text)
        val durLabel = String.format("%.1fs", durMs / 1000f)
        c.drawText(durLabel, padL + w - text.measureText(durLabel), height - 5f * d, text)
    }

    private fun fmt(v: Float) = if (v >= 100f) String.format("%.0f", v) else String.format("%.1f", v)
}
