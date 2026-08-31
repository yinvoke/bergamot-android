import org.jetbrains.kotlin.gradle.dsl.JvmTarget
plugins {
    id("com.android.library")

}

android {
    namespace = "io.github.yinvoker.bergamot"
    compileSdk = 36
    ndkVersion = "29.0.13113456"

    defaultConfig {
        minSdk = 28
        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DSSPLIT_USE_INTERNAL_PCRE2=ON",
                    "-DCOMPILE_TESTS=OFF",
                    "-DANDROID_STL=c++_shared",
                    "-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON",
                    // BUILD_ARCH=native runs *host* SSE probing — poison when
                    // cross-compiling; the explicit arch skips it entirely.
                    "-DBUILD_ARCH=armv8-a",
                )
                targets += "bergamot"
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("../CMakeLists.txt")
            version = "3.31.6"
        }
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
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
}
