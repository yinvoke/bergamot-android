#!/usr/bin/env python3
"""Run paired Android engine benchmarks; see docs/benchmarks/v0.2.0/README.md."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--adb', default='adb')
p.add_argument('--serial', required=True)
p.add_argument('--build-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
repo = Path(__file__).resolve().parents[2]
remote = '/data/local/tmp/bergamot-v010-v020'
versions = ['v0.1.0', 'v0.2.0']

def adb(*args):
    return subprocess.check_output([a.adb, '-s', a.serial, *args], text=True).strip()

def shell(command):
    return adb('shell', command)

def conditions():
    caps = {str(i): shell(f'cat /sys/devices/system/cpu/cpu{i}/cpufreq/scaling_max_freq') for i in [4, 7]}
    battery = shell('dumpsys battery')
    temp = int(next(x.split(':')[1] for x in battery.splitlines() if x.strip().startswith('temperature:'))) / 10
    return {'max_freq_khz': caps, 'battery_temp_c': temp}

shell('mkdir -p ' + remote)
binaries = {}
for version in versions:
    binary = a.build_root / version / 'build/tools/smoke/smoke'
    binaries[version] = hashlib.sha256(binary.read_bytes()).hexdigest()
    adb('push', str(binary), remote + '/' + version)
shell('chmod 755 ' + remote + '/v0.*')
# The device already holds the official models. Keep both configs at AAR defaults.
configs = {}
for lang, source in [('enzh', '/data/local/tmp/bg/config-mbw1024.yml'), ('jaen', '/data/local/tmp/bg/config-jaen.yml')]:
    content = shell('cat ' + source).replace('mini-batch-words: 512', 'mini-batch-words: 1024') + '\n'
    configs[lang] = content
    local = a.output.parent / (lang + '.yml')
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(content)
    adb('push', str(local), remote + '/' + lang + '.yml')
for lang, source in [('eng', 'eng200.txt'), ('jpn', 'jpn200.txt')]:
    shell(f'cp /data/local/tmp/bg/{source} {remote}/{lang}.txt')
    expected = (repo / f'sample/src/main/assets/bench/{lang}.txt').read_text().splitlines()
    actual = shell(f'cat {remote}/{lang}.txt').splitlines()
    assert expected == actual and len(actual) == 200, lang

report = {
    'measured_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'device': {'model': shell('getprop ro.product.model'), 'android': shell('getprop ro.build.version.release'), 'soc': 'Snapdragon 865', 'i8mm': False},
    'versions': {v: subprocess.check_output(['git', 'rev-parse', v + '^{commit}'], cwd=repo, text=True).strip() for v in versions},
    'binary_sha256': binaries,
    'engine_tree': {v: subprocess.check_output(['git', 'rev-parse', v + ':engine'], cwd=repo, text=True).strip() for v in versions},
    'model_files_sha256': {path: shell('sha256sum ' + path).split()[0] for path in sorted({line.strip()[2:] for content in configs.values() for line in content.splitlines() if line.strip().startswith('- /')})},
    'harness_sha256': hashlib.sha256((repo / 'tools/version-bench/main.cpp').read_bytes()).hexdigest(),
    'config_yaml': configs,
    'corpus_sha256': {lang: hashlib.sha256((repo / f'sample/src/main/assets/bench/{lang}.txt').read_bytes()).hexdigest() for lang in ['eng', 'jpn']},
    'protocol': {'affinity': 'f0', 'rounds': 3, 'version_order': ['AB', 'BA', 'AB'], 'passes_per_process': 3, 'cache_size': 0, 'cooldown_seconds': 4, 'cold_ms': 'service + model creation + first translation; OS page cache not flushed', 'peak_rss_mib': 'VmHWM immediately after first translation, KiB / 1024', 'warm_ms': 'passes 1 and 2; reported separately'},
    'runs': [],
}
scenarios = [('enzh_w1', 1, 'eng', ['enzh']), ('enzh_w2', 2, 'eng', ['enzh']), ('enzh_w4', 4, 'eng', ['enzh']), ('pivot_w1', 1, 'jpn', ['jaen', 'enzh'])]
for round_id in range(3):
    order = versions if round_id % 2 == 0 else versions[::-1]
    for name, workers, corpus, models in scenarios:
        for version in order:
            before = conditions()
            cmd = f'taskset f0 {remote}/{version} {workers} ' + ' '.join(f'{remote}/{m}.yml' for m in models) + f' < {remote}/{corpus}.txt'
            result = subprocess.run([a.adb, '-s', a.serial, 'shell', cmd], capture_output=True, text=True, timeout=180)
            samples = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
            run = {'round': round_id + 1, 'scenario': name, 'version': version, 'command': cmd, 'before': before, 'after': conditions(), 'returncode': result.returncode, 'samples': samples}
            if result.returncode or len(samples) != 3:
                run['error'] = (result.stdout + result.stderr)[-3000:]
            report['runs'].append(run)
            a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            print(round_id + 1, name, version, result.returncode, samples[0] if samples else run.get('error'), flush=True)
            time.sleep(4)
print('Saved', a.output, flush=True)
