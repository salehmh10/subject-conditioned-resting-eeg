"""Export a self-contained presentation snapshot from prepared evidence only."""
from __future__ import annotations

import html
import json
from pathlib import Path
import sys

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
sys.dont_write_bytecode = True

import numpy as np
from plotly.offline import get_plotlyjs
import plotly.io as pio
from lab.charts import (workflow_chart, split_chart, fidelity_bars, psd_chart,
                        connectivity_grid, fingerprint_chart, utility_comparison,
                        contrast_forest, waveform_chart)
from lab.config import PLOT_CONFIG


def create_snapshot(bundle: dict) -> str:
    sections = []

    def add(title, message, visual, notes):
        if not isinstance(visual, str):
            visual = pio.to_html(visual, include_plotlyjs=False, full_html=False, config=PLOT_CONFIG,
                                 div_id=f'figure-{len(sections)}')
        sections.append((title, message, visual, notes))

    add('Question', 'How closely do synthetic EEG signals reproduce real EEG, and does that similarity translate into useful eye-state classification?',
        workflow_chart(), 'Fidelity and utility answer different questions. A better signal metric does not ensure a better classifier.')
    add('Data', 'The evaluation concerns different blocks from the same 198 participants.',
        split_chart(bundle['metadata']), 'MPI-LEMON resting-state EEG; 19 channels, 128 Hz, four-second windows. There is no demonstrated unseen-participant or new-session generalization.')
    architecture = APP / 'assets/architecture.svg'
    model_visual = architecture.read_text(encoding='utf-8') if architecture.exists() else workflow_chart()
    add('Models', 'B0 receives eye state; B1 and B2 receive participant identity and eye state.', model_visual,
        'All use a temporal 1D U-Net. B2 is a continued-training, selected-checkpoint variant of the same participant-conditioned family. Different training histories prevent a pure causal conditioning comparison. Classifier training seeds are not new generators.')
    preview = APP / 'data/presentation_bundle/previews.npz'
    preview_meta = APP / 'data/presentation_bundle/previews.json'
    meta = json.loads(preview_meta.read_text(encoding='utf-8')) if preview_meta.exists() else {}
    if preview.exists() and meta.get('available'):
        with np.load(preview, allow_pickle=False) as arrays:
            index = next((i for i, r in enumerate(meta['records']) if r['state'] == 'EC'), 0)
            channels = meta['channels']
            channel = 'O1' if 'O1' in channels else channels[0]
            # Share only one channel from four displayed windows, never the full preview pool.
            ci = channels.index(channel)
            waves = {m: arrays['waveforms_uv'][index, j, ci:ci+1].copy() for j, m in enumerate(meta['sources'])}
            visual = waveform_chart(arrays['time_s'].copy(), waves, [channel], channel)
        add('Waveforms', f'One bounded research example: {channel}, {meta["records"][index]["state"]}, four separately displayed windows.', visual,
            'Precomputed synthetic samples are not time-aligned reconstructions of the real window. This snapshot includes one channel from one deterministic Train-only preview record (four sources, 512 samples each); it is illustrative, not evidence of population-level performance.')
    else:
        add('Signal representation', 'Train previews are unavailable in this bundle; group spectral evidence remains available.',
            fidelity_bars(bundle['fidelity_metrics'], 'log_psd_mae_db'),
            'No substitute waveform was generated. Prepare verified Train-only previews locally to enable the Signal explorer.')
    psd = bundle['psd']
    spectra = {m: (10*np.log10(np.maximum(np.asarray(v, dtype=float), 1e-30))).tolist()
               for m, v in psd['group']['EC'].items()}
    add('Spectrum and alpha', 'Compare spectrum shape and level against the Real Train reference.',
        psd_chart(psd['frequencies_hz'], spectra, state='EC', aggregation='198 participants · linear-power averages before dB conversion'),
        'Official precomputed spectra are averaged in linear power before dB conversion. This is descriptive full-pool Fidelity relative to Real Train. Explore alpha EC-minus-EO effects separately in the application.')
    conn = bundle['connectivity']
    add('Connectivity', 'The same color scale makes sensor-space phase-lag patterns comparable.',
        connectivity_grid(conn['group']['EC']['alpha'], conn['channels'], state='EC', band='alpha'),
        'dwPLI² is a debiased sensor-space phase-lag estimate, not anatomical or causal connectivity. The diagonal is masked; negative estimates and missing edges are preserved. Do not square it again.')
    add('Participant identity', 'Participant-conditioned signals retain strong in-sample log-bandpower fingerprints.',
        fingerprint_chart(bundle['fingerprint'], metric='top1'),
        '76 log-bandpower features; 198 candidates. B0 group labels are evaluation strata, not model inputs. In-sample identification and zero exact-copy detections do not establish privacy protection.')
    add('Classification utility', 'All eleven conditions and all three classifier families remain visible.',
        utility_comparison(bundle['utility_summary'], bundle['utility_runs']),
        'Final Test evaluates held-out blocks. The EEGNet mean uses three classifier seeds; whiskers represent population SD, not a confidence interval. The two classical families each have one fit per condition. Real Duplicate controls repeated data presentations.')
    family = next(r['family'] for r in bundle['utility_summary'] if 'EEGNET' in r['family'].upper())
    rows = {r['condition']: r for r in bundle['utility_summary'] if r['family'] == family and r['split'] == 'final_test'}
    delta = 100*(rows['REAL_PLUS_B2_1X']['mean']-rows['REAL_ONLY']['mean'])
    add('What the results establish', f'B2 1× augmentation changes this EEGNet comparison by {delta:+.3f} percentage points.',
        contrast_forest(bundle['contrasts'], family=family),
        'The existing paired subject-bootstrap intervals are descriptive and are not multiplicity-adjusted. B2 does not show universal classifier benefit or conclusive superiority over B1.')
    limits = '''<div class="limit-grid"><article><h3>Known participants</h3><p>Different held-out blocks; no established transfer to new people or new sessions.</p></article><article><h3>Descriptive Fidelity</h3><p>Real Train is the reference. Matched-count Train calibration is a separate comparison.</p></article><article><h3>Unequal training histories</h3><p>B0 versus B1/B2 does not isolate a pure causal effect of participant conditioning.</p></article><article><h3>Real Validation remains involved</h3><p>Even TSTR EEGNet checkpoint selection uses real Validation. Equal epoch ceilings do not imply equal optimizer updates.</p></article><article><h3>Different uncertainty summaries</h3><p>Between-seed population SD and paired subject-bootstrap confidence intervals quantify different variation.</p></article><article><h3>No privacy guarantee</h3><p>In-sample fingerprints and no exact-copy detections do not establish anonymity or absence of memorization.</p></article></div>'''
    add('Limitations and conclusion', 'Improved Fidelity can coexist with small, classifier-dependent utility gains.', limits,
        'This is an offline presentation snapshot of saved evidence. Use the local app for full participant, metric, seed, and condition exploration. No training, inference, or new signal generation occurs.')

    body = []
    for i, (title, message, visual, notes) in enumerate(sections):
        body.append(f'<section class="slide{" active" if i == 0 else ""}" data-step="{i}"><p class="eyebrow">EEG SYNTHETIC LAB / {i+1:02d}</p><h1>{html.escape(title)}</h1><p class="message">{html.escape(message)}</p><div class="visual">{visual}</div><details><summary>Speaker notes and context</summary><p>{html.escape(notes)}</p></details></section>')
    return '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none';"><title>EEG Synthetic Lab — presentation snapshot</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f5f8fa;color:#173449;font:17px Arial,sans-serif}nav{position:sticky;top:0;z-index:9;background:#fff;border-bottom:1px solid #dbe6ec;display:flex;align-items:center;justify-content:space-between;padding:14px 4vw;gap:16px}button{padding:10px 22px;background:#087f8c;color:white;border:0;border-radius:7px;font-size:16px;cursor:pointer}button:disabled{opacity:.4;cursor:default}main{max-width:1500px;margin:auto;padding:28px 4vw 45px}.slide{display:none}.slide.active{display:block}.eyebrow{font-size:12px;letter-spacing:2px;color:#087f8c;font-weight:bold}h1{font-size:38px;margin:12px 0}.message{font-size:23px;line-height:1.5;max-width:1200px}.visual{background:white;padding:18px;border:1px solid #dce6ec;border-radius:12px;overflow:hidden}.visual>svg{width:100%;height:auto;max-height:560px}.limit-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}.limit-grid article{border-top:3px solid #087f8c;padding:10px 15px;line-height:1.6}details{margin-top:22px;line-height:1.6}summary{cursor:pointer;color:#087f8c;font-weight:bold}footer{padding:20px 4vw;font-size:12px;color:#536b7e}#counter{min-width:100px;text-align:center}@media(max-width:800px){h1{font-size:28px}.message{font-size:19px}.limit-grid{grid-template-columns:1fr 1fr}nav strong{display:none}}@media print{nav{display:none}.slide{display:block;break-after:page}.visual{break-inside:avoid}details{display:block}}
    </style><script>''' + get_plotlyjs() + '''</script></head><body><nav><strong>EEG Synthetic Lab <small>· offline snapshot</small></strong><div><button id="previous" onclick="move(-1)">Previous</button> <span id="counter"></span> <button id="next" onclick="move(1)">Next</button></div></nav><main>''' + ''.join(body) + '''</main><footer>Aggregate saved results, with one small Train-only research example when available. Embedded Plotly assets and data; no internet or Python required. No external service, telemetry, or model execution.</footer><script>
    let step=0; const slides=[...document.querySelectorAll('.slide')];
    function show(){slides.forEach((s,i)=>s.classList.toggle('active',i===step));document.getElementById('counter').textContent=(step+1)+' / '+slides.length;document.getElementById('previous').disabled=step===0;document.getElementById('next').disabled=step===slides.length-1;requestAnimationFrame(()=>slides[step].querySelectorAll('.js-plotly-plot').forEach(p=>Plotly.Plots.resize(p)));}
    function move(d){step=Math.max(0,Math.min(slides.length-1,step+d));show();window.scrollTo(0,0)}document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')move(1);if(e.key==='ArrowLeft')move(-1)});window.addEventListener('resize',show);show();
    </script></body></html>'''


def main():
    bundle_path = APP / 'data/presentation_bundle/bundle.json'
    if not bundle_path.exists():
        raise SystemExit('Presentation bundle missing. Run prepare_data.ps1 first.')
    bundle = json.loads(bundle_path.read_text(encoding='utf-8'))
    output = APP / 'exports/EEG_Synthetic_Lab_Presentation.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(create_snapshot(bundle), encoding='utf-8')
    print(f'Created {output.name} ({output.stat().st_size:,} bytes). Embedded assets; no internet required to view.')


if __name__ == '__main__':
    main()
