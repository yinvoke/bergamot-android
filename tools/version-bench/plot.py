#!/usr/bin/env python3
"""Render version-comparison bars from the committed raw measurements."""
import json
from pathlib import Path
from statistics import median
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'docs/benchmarks/v0.2.0'
raw = json.loads((DATA / 'results.json').read_text())
font = Path('/Library/Fonts/Arial Unicode.ttf')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 11, 'savefig.facecolor': 'white'})
versions = ['v0.1.0', 'v0.2.0']
cases = ['enzh_w1', 'enzh_w2', 'enzh_w4', 'pivot_w1']
label_map = dict(zip(cases, ['英 → 中 · 1 worker', '英 → 中 · 2 workers', '英 → 中 · 4 workers', '日 → 中 · 1 worker']))
assert len(raw['runs']) == 24, 'benchmark incomplete'
def complete(case, version):
    rows = [r for r in raw['runs'] if r['scenario']==case and r['version']==version]
    return len(rows)==3 and all(r['returncode']==0 and len(r['samples'])==3 for r in rows)
cases = [c for c in cases if all(complete(c,v) for v in versions)]
assert cases, 'no complete paired scenarios'
labels = [label_map[c] for c in cases]
# Contract: README static image; two grouped horizontal bar panels, all complete scenarios,
# three fresh processes/version/scenario; median + observed min/max. Zero axes.
# One blue root for the new version, open gray bars for the old version; direct
# labels retain version identity in grayscale. Units and cohort are always visible.
colors = ['#E6E9EF', '#356BC4']
edges = ['#8893A3', '#285391']
ink = '#202936'
muted = '#606C7D'
fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.7 + .4*len(cases)), dpi=180)
fig.subplots_adjust(left=.155, right=.93, bottom=.27, top=.74, wspace=.42)
fig.text(.04, .935, 'v0.1.0 → v0.2.0 速度与内存对比', fontsize=23, color=ink, weight='bold')
fig.text(.04, .883, '小米 10 / 骁龙 865 · FLORES-200 前 200 句 · AsyncService · 三轮独立进程中位数', fontsize=12, color=muted)
fig.legend(handles=[Patch(facecolor=colors[i], edgecolor=edges[i], label=v) for i,v in enumerate(versions)], loc='upper right', bbox_to_anchor=(.96,.965), ncol=2, frameon=False, fontsize=11)
summary = {}
for ax, metric, scale, title, unit in zip(axes, ['elapsed_ms', 'peak_rss_mib'], [1000, 1], ['首次翻译耗时', '首次翻译峰值内存'], ['秒 · 越低越好', 'RSS / MiB · 越低越好']):
    values = {}
    for case in cases:
        for version in versions:
            runs = [r for r in raw['runs'] if r['scenario']==case and r['version']==version]
            assert len(runs)==3 and all(r['returncode']==0 and len(r['samples'])==3 for r in runs)
            values[case,version] = [r['samples'][0][metric]/scale for r in runs]
    high = max(max(v) for v in values.values())
    ax.set_xlim(0, high*1.22)
    for idx,case in enumerate(cases):
        before, after = [median(values[case,v]) for v in versions]
        reduction = 100*(1-after/before)
        summary.setdefault(case,{})[metric] = {'before':before*scale,'after':after*scale,'reduction_pct':reduction}
        for i,version in enumerate(versions):
            nums = values[case,version]
            value = median(nums)
            y = idx + (-.19 if i==0 else .19)
            ax.barh(y,value,height=.30,color=colors[i],edgecolor=edges[i],linewidth=.8,zorder=3)
            ax.errorbar(value,y,xerr=[[value-min(nums)],[max(nums)-value]],fmt='none',ecolor=ink,elinewidth=.8,capsize=2,zorder=4)
            label = f'{value:.2f}' if metric=='elapsed_ms' else f'{value:.0f}'
            ax.text(max(nums)+high*.022,y,label,va='center',fontsize=11,color=ink)
        ax.text(1.03,idx+.19,f'−{reduction:.1f}%',transform=ax.get_yaxis_transform(),va='center',fontsize=11,weight='bold',color=edges[1])
    ax.set_yticks(range(len(cases)),labels if ax is axes[0] else ['']*len(cases))
    ax.set_ylim(len(cases)-.45,-.6)
    ax.set_title(title,pad=19,loc='left',fontsize=16,weight='bold',color=ink)
    ax.set_xlabel(unit,labelpad=12,color=muted)
    ax.xaxis.grid(True,color='#E8ECF1',linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both',length=0,pad=8,labelcolor=muted)
    for spine in ['top','right','left']: ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#D1D8E2')
fig.text(.04,.115,'首次翻译含服务 / 模型创建与加载；短线为三轮最小–最大值。相同模型、1024 mini-batch-words、128 MiB workspace，绑大核 f0。',fontsize=10,color=muted)
fig.text(.04,.065,'每次新进程，未清空系统文件缓存；图为原生引擎 RSS，非 Android app PSS。骁龙 865 不支持 i8mm，两版均走 ruy。',fontsize=10,color=muted)
fig.savefig(DATA / 'comparison.png')
(DATA / 'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
