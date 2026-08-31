package io.github.yinvoker.bergamot

import java.io.File
import kotlin.io.path.createTempDirectory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ModelFilesTest {

    private fun dir(vararg names: String): File {
        val d = createTempDirectory("mf").toFile()
        names.forEach { File(d, it).writeText("x") }
        return d
    }

    @Test
    fun `mozilla dual-vocab layout resolves all four files`() {
        val d = dir(
            "model.enzh.intgemm.alphas.bin",
            "srcvocab.enzh.spm",
            "trgvocab.enzh.spm",
            "lex.50.50.enzh.s2t.bin",
        )
        val m = ModelFiles.fromDirectory(d)
        assertEquals("model.enzh.intgemm.alphas.bin", m.model.name)
        assertEquals("srcvocab.enzh.spm", m.srcVocab.name)
        assertEquals("trgvocab.enzh.spm", m.trgVocab.name)
        assertEquals("lex.50.50.enzh.s2t.bin", m.shortlist.name)
    }

    @Test
    fun `single shared vocab is used for both sides`() {
        val d = dir(
            "model.jaen.intgemm.alphas.bin",
            "vocab.jaen.spm",
            "lex.50.50.jaen.s2t.bin",
        )
        val m = ModelFiles.fromDirectory(d)
        assertEquals("vocab.jaen.spm", m.srcVocab.name)
        assertEquals(m.srcVocab, m.trgVocab)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `missing model file fails loudly`() {
        ModelFiles.fromDirectory(dir("vocab.x.spm", "lex.x.s2t.bin"))
    }

    @Test
    fun `config yaml carries absolute paths and workspace`() {
        val d = dir(
            "model.enzh.intgemm.alphas.bin",
            "srcvocab.enzh.spm",
            "trgvocab.enzh.spm",
            "lex.50.50.enzh.s2t.bin",
        )
        val yaml = ModelFiles.fromDirectory(d).toConfigYaml(workspaceMb = 96)
        assertTrue(yaml.contains(File(d, "model.enzh.intgemm.alphas.bin").absolutePath))
        assertTrue(yaml.contains("workspace: 96"))
        assertTrue(yaml.contains("gemm-precision: int8shiftAlphaAll"))
        assertTrue(yaml.lines().none { it.startsWith(" ") && it.contains("\t") })
    }
}
