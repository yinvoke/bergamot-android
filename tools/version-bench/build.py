#!/usr/bin/env python3
"""Export both released engine trees and build the same measurement harness."""
import argparse
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--ndk', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--jobs', type=int, default=4)
a = p.parse_args()
repo = Path(__file__).resolve().parents[2]
a.output.mkdir(parents=True, exist_ok=True)
for tag in ['v0.1.0', 'v0.2.0']:
    dest = a.output.resolve() / tag
    dest.mkdir()  # Refuse to overwrite an existing source/build tree.
    archive = a.output / (tag + '.tar')
    with archive.open('wb') as f:
        subprocess.run(['git', 'archive', tag], cwd=repo, stdout=f, check=True)
    subprocess.run(['tar', '-xf', str(archive.resolve()), '-C', str(dest)], check=True)
    shutil.copyfile(repo / 'tools/version-bench/main.cpp', dest / 'tools/smoke/smoke.cpp')
    if tag == 'v0.1.0':
        cmake = dest / 'CMakeLists.txt'
        cmake.write_text(cmake.read_text().replace('  add_subdirectory(jni)', '  add_subdirectory(jni)\n  add_subdirectory(tools/smoke)'))
    # Vendored CMake generates git_revision.h from a HEAD log. This empty commit
    # is build metadata only; all engine inputs above are exported from the tag.
    subprocess.run(['git', 'init', str(dest)], check=True)
    subprocess.run(['git', '-C', str(dest), '-c', 'user.name=Benchmark', '-c', 'user.email=benchmark@localhost', 'commit', '--allow-empty', '-m', f'Build metadata for {tag} source export'], check=True)
    flags = ['-DANDROID_ABI=arm64-v8a', '-DANDROID_PLATFORM=android-28', '-DCMAKE_BUILD_TYPE=Release', '-DANDROID_STL=c++_static', '-DSSPLIT_USE_INTERNAL_PCRE2=ON', '-DCOMPILE_TESTS=OFF', '-DBUILD_ARCH=armv8-a', '-DBUILD_SMOKE=ON']
    subprocess.run(['cmake', '-S', str(dest), '-B', str(dest / 'build'), '-DCMAKE_TOOLCHAIN_FILE=' + str(a.ndk.resolve() / 'build/cmake/android.toolchain.cmake'), *flags], check=True)
    subprocess.run(['cmake', '--build', str(dest / 'build'), '--target', 'smoke', '-j', str(a.jobs)], check=True)
