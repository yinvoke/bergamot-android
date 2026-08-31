import org.jetbrains.kotlin.gradle.dsl.JvmTarget
plugins {
    id("com.android.application")

}

android {
    namespace = "io.github.yinvoker.bergamot.bench"
    compileSdk = 36

    defaultConfig {
        applicationId = "io.github.yinvoker.bergamot.bench"
        minSdk = 28
        targetSdk = 36
        versionCode = 1
        versionName = "0.1"
        ndk { abiFilters += "arm64-v8a" }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

dependencies {
    implementation(project(":bergamot"))
    implementation("com.google.mlkit:translate:17.0.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
}
