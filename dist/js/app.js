/* scCopilot 静态站逻辑
 *
 * 数据全部来自 web/data/*.json（scripts/export_site_data.py 本机真实跑批产出）。
 * 这里只做展示与交互，不重算任何统计量——页面上出现的每个数字都能在 JSON 里找到。
 */
(function () {
  'use strict';

  const PALETTE = ['#2563eb', '#06b6d4', '#7c3aed', '#f59e0b', '#10b981',
                   '#ef4444', '#ec4899', '#84cc16', '#6366f1', '#14b8a6',
                   '#f97316', '#a855f7', '#0ea5e9', '#22c55e', '#e11d48'];

  const state = { manifest: null, data: null, cluster: null, colorMode: 'cluster' };
  const $ = (id) => document.getElementById(id);

  // Plotly 在深色玻璃卡片上：透明底 + 浅色字，网格线压暗。
  const DARK_FONT = { color: '#c6d5f2', size: 11 };
  const DARK_GRID = '#223252';

  // 基因名、模型文本都按不可信输入处理：进 DOM 前一律转义。
  const esc = (v) => String(v == null ? '' : v)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

  const fmt = (v, d = 2) => (v == null || Number.isNaN(v)) ? '—' : Number(v).toFixed(d);
  const fmtInt = (v) => (v == null || Number.isNaN(v)) ? '—' : Number(v).toLocaleString();
  const pct = (v) => (v == null || Number.isNaN(v)) ? '—' : `${(Number(v) * 100).toFixed(0)}%`;

  function clusterColor(c) {
    const order = state.data.clusters.map((r) => String(r.cluster));
    const i = order.indexOf(String(c));
    return PALETTE[(i < 0 ? 0 : i) % PALETTE.length];
  }

  // 簇的数字 ID 在图上一眼看不出是什么细胞。名字只从 ai 记录里取：
  // 被拦下或证据不足的簇，predicted_type 本身就是"证据不足，建议人工复核"，不替它编类型。
  function clusterName(c) {
    const ai = state.data.ai ? state.data.ai[String(c)] : null;
    return (ai && ai.predicted_type) || '未判定';
  }

  // 三种着色共用同一条 hover：单细胞编号 + 所在簇 + 该簇判定 + 两个 QC 数值，
  // 免得切到"线粒体占比"就只剩一个百分比。数值直接取 JSON 原值，前端不重算。
  function hoverText(i) {
    const u = state.data.umap, c = u.cluster[i];
    return `细胞 #${i} ｜ 簇 ${c} · ${clusterName(c)}`
      + ` ｜ 线粒体 ${Number(u.pct_mito[i]).toFixed(2)}%`
      + ` ｜ UMI ${Number(u.n_counts[i]).toFixed(0)}`;
  }

  const UMAP_CONFIG = {
    responsive: true,
    // 圈框放大之后要能退回去：悬停时浮出工具条，里面有放大/平移/还原
    displayModeBar: 'hover',
    modeBarButtonsToRemove: ['toImage', 'autoScale2d', 'sendDataToCloud', 'lasso2d', 'select2d'],
  };

  function resetView() {
    const el = $('umap');
    if (el && el.data) Plotly.relayout(el, { 'xaxis.autorange': true, 'yaxis.autorange': true });
  }

  // ----------------------------------------------------------------- 载入
  // 首屏只渲染入口按钮，不取 data/*.json：点了「内置演示数据」才拉数、才展开结果区。
  // 显隐挂在 body 的 class 上（CSS 里 .sc-gated 隐藏 #demo-zone / #scenario-picker），
  // 不写 HTML 的 hidden 属性：静态站没有构建号时浏览器可能拿旧缓存的 app.js，
  // 旧脚本不认识 hidden，结果就是"点了没反应"。放在 JS 里最坏只退回"直接显示"。
  let revealed = false;
  document.body.classList.add('sc-gated');

  async function reveal(id, btn) {
    if (!revealed) {
      revealed = true;
      document.body.classList.remove('sc-gated');
    }
    await switchScenario(id, btn);
  }

  async function boot() {
    const m = await (await fetch('data/manifest.json')).json();
    state.manifest = m;
    const box = $('scenario-buttons');
    box.innerHTML = m.scenarios.map((s, i) =>
      `<button class="sc-btn" data-scenario="${esc(s.id)}" aria-pressed="${i === 0}">${esc(s.title)}</button>`
    ).join(' ');
    box.querySelectorAll('button').forEach((b) => {
      b.addEventListener('click', () => reveal(b.dataset.scenario, b));
    });
  }

  // 首屏两个功能入口：演示数据直接进场；上传入口如实指向本地工作台，不放假日上传框。
  $('btn-demo').addEventListener('click', async () => {
    const first = $('scenario-buttons').firstElementChild;
    if (first) await reveal(first.dataset.scenario, first);
    $('metrics').scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
  // 上传入口：本机 Streamlit 工作台在跑就直接跳过去上传；探测不到才展开启动说明。
  // 静态站自身不在浏览器里跑分析，所以这里不做假的上传框。
  $('btn-upload').addEventListener('click', async () => {
    const note = $('upload-note');
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 1500);
      await fetch('http://localhost:8501/', { mode: 'no-cors', signal: ctrl.signal });
      clearTimeout(timer);
      note.hidden = true;
      $('btn-upload').setAttribute('aria-expanded', 'false');
      window.open('http://localhost:8501/', '_blank', 'noopener');
    } catch (err) {
      note.hidden = false;
      $('btn-upload').setAttribute('aria-expanded', 'true');
    }
  });

  async function switchScenario(id, btn) {
    if (btn) {
      $('scenario-buttons').querySelectorAll('button')
        .forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
    }
    const entry = state.manifest.scenarios.find((s) => s.id === id) || { blurb: '' };
    $('scenario-blurb').textContent = entry.blurb;
    let data;
    try {
      data = await (await fetch(`data/${id}.json`)).json();
    } catch (err) {
      // "点了没反应"最坏的一档是取数失败。取不到就写脸上，不能只留在控制台。
      $('scenario-blurb').textContent =
        `数据 data/${id}.json 读取失败（${err.message}）。` +
        `请用 python -m http.server 打开本站，file:// 直开用 fetch 取不到数据。`;
      return;
    }
    state.data = data;
    state.cluster = null;
    renderMetrics();
    renderUmap();
    renderClusterTable();
    renderResolution();
    renderRunInfo();
    renderAi();
  }

  // ----------------------------------------------------------------- 指标
  function renderMetrics() {
    const d = state.data, qc = d.qc;
    const removed = Number(qc.cells_removed_pct) || 0;
    const okCells = d.clusters.filter((c) => c.evidence_ok).length;
    const items = [
      ['细胞（QC 后）', fmtInt(qc.n_cells_final), removed > 0 ? `-${removed.toFixed(1)}%` : null],
      ['基因', fmtInt(qc.n_genes_final), null],
      ['簇数', String(d.clusters.length), null],
      ['分辨率', fmt(d.config.resolution), null],
      ['证据充足的簇', `${okCells}/${d.clusters.length}`, null],
    ];
    $('metrics').innerHTML = items.map(([label, value, delta]) => `
      <div class="metric">
        <div class="metric-label">${esc(label)}</div>
        <div class="metric-value">${esc(value)}${delta ? `<span class="metric-delta">${esc(delta)}</span>` : ''}</div>
      </div>`).join('');
  }

  // ----------------------------------------------------------------- UMAP
  function renderUmap() {
    const u = state.data.umap;
    const mode = state.colorMode;
    let trace, layout;

    if (mode === 'cluster') {
      trace = {
        type: 'scattergl', mode: 'markers',
        x: u.x, y: u.y,
        text: u.x.map((_, i) => hoverText(i)),
        hoverinfo: 'text', hoverdistance: 30,
        marker: { size: 5, color: u.cluster.map(clusterColor), opacity: 0.85, line: { width: 0 } },
        customdata: u.cluster,
      };
      layout = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: DARK_FONT,
                 margin: { l: 8, r: 8, t: 8, b: 8 }, xaxis: { visible: false }, yaxis: { visible: false },
                 showlegend: false, dragmode: 'zoom' };
    } else {
      const vals = u[mode];
      trace = {
        type: 'scattergl', mode: 'markers', x: u.x, y: u.y,
        text: u.x.map((_, i) => hoverText(i)),
        hoverinfo: 'text', hoverdistance: 30,
        marker: { size: 5, color: vals, colorscale: 'Viridis', opacity: 0.9,
                  colorbar: { thickness: 10, tickfont: DARK_FONT,
                              title: { text: mode === 'pct_mito' ? '% 线粒体' : 'UMI', font: { size: 10, color: '#c6d5f2' } } } },
        customdata: u.cluster,
      };
      layout = { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: DARK_FONT,
                 margin: { l: 8, r: 8, t: 8, b: 8 }, xaxis: { visible: false }, yaxis: { visible: false },
                 showlegend: false, dragmode: 'zoom' };
    }

    const el = $('umap');
    if (el.dataset.bound !== '1') {
      // Plotly 不会替我们清掉占位节点，留着会一直压在散点上面。
      el.innerHTML = '';
      Plotly.newPlot(el, [trace], layout, UMAP_CONFIG);
      el.on('plotly_click', (ev) => {
        const p = ev && ev.points && ev.points[0];
        if (!p) return;
        const label = p.customdata != null ? p.customdata : state.data.umap.cluster[p.pointNumber];
        selectCluster(String(label));
      });
      // Plotly 自带的双击复位在这张图上实测不生效，改挂它自己的 doubleclick 事件
      el.on('plotly_doubleclick', resetView);
      el.addEventListener('dblclick', resetView);
      el.dataset.bound = '1';
    } else {
      Plotly.react(el, [trace], layout, UMAP_CONFIG);
    }

    $('umap-caption').textContent =
      mode === 'cluster'
        ? `共 ${u.x.length.toLocaleString()} 个细胞，颜色 = 簇。点击任意细胞即可解读它所在的簇；悬停看单个细胞。`
        : mode === 'pct_mito'
          ? '颜色 = 线粒体基因占比（%）。高占比簇通常意味着应激或濒死细胞，AI 会据此降低可信度。'
          : '颜色 = 每个细胞的 UMI 总量，反映测序深度。';
  }

  document.querySelectorAll('[data-color]').forEach((b) => {
    b.addEventListener('click', () => {
      state.colorMode = b.dataset.color;
      document.querySelectorAll('[data-color]').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
      renderUmap();
    });
  });
  $('umap-reset').addEventListener('click', resetView);

  // ----------------------------------------------------------------- 选中
  function selectCluster(c) {
    state.cluster = String(c);
    renderAi();
    renderBarplot();
    document.querySelectorAll('#cluster-table tbody tr')
      .forEach((tr) => { tr.style.outline = tr.dataset.cluster === state.cluster ? '2px solid #2563eb' : 'none'; });
  }

  // ----------------------------------------------------------------- AI 卡
  function renderAi() {
    const slot = $('escalate-slot');
    slot.innerHTML = '';
    $('trace-card').hidden = true;
    $('kg-card').hidden = true;
    $('evidence-card').hidden = true;
    $('gate-card').hidden = true;
    $('het-card').hidden = true;

    if (!state.cluster) {
      $('ai-placeholder').hidden = false;
      $('ai-body').innerHTML = '';
      $('ai-meta').textContent = '等待选择簇…';
      $('bar-caption').textContent = '';
      Plotly.purge('barplot');
      return;
    }

    const d = state.data;
    const ai = d.ai[state.cluster];
    const rows = d.clusters.filter((r) => String(r.cluster) === state.cluster);
    if (!ai || !rows.length) {
      $('ai-placeholder').hidden = false;
      $('ai-body').innerHTML = '<p class="sc-hint">该簇没有可用的 marker（例如只有 1 个细胞，统计检验被跳过）。</p>';
      return;
    }
    $('ai-placeholder').hidden = true;

    const nTop = Math.min(d.n_markers, d.markers[state.cluster].length);
    $('ai-meta').textContent = `簇 ${state.cluster} ｜ Top ${nTop} 个差异基因 ｜ 本地规则 + 知识图谱`;

    const refused = ai.refused || ai.critique.status === 'escalate';
    if (ai.critique.status === 'escalate') {
      slot.innerHTML = `
        <div class="sc-alert" role="alert">
          <div class="sc-alert-icon">🚫</div>
          <div>
            <div class="sc-alert-title">AI 主动拒绝回答</div>
            <div class="sc-alert-body">
              原因：${esc(ai.critique.reason || '未知')}<br>
              这体现了系统的防幻觉能力：当证据不足或与知识图谱冲突时，AI 不会强行给出不可靠的结论。
            </div>
          </div>
        </div>`;
    }

    const confCls = { '高': 'high', '中': 'mid', '低': 'low' }[ai.confidence_label] || 'none';
    const cited = ai.cited_genes && ai.cited_genes.length
      ? `<p class="sc-eyebrow">理由中引用到的本次 marker</p>` +
        ai.cited_genes.slice(0, 12).map((g) => `<span class="sc-chip sc-chip--gene">${esc(g)}</span>`).join('')
      : '';
    const viol = (ai.violations || []).map((v) => `<div class="sc-note sc-note--bad">防幻觉告警：${esc(v)}</div>`).join('');
    const notes = (ai.notes || []).map((n) => `<div class="sc-note">说明：${esc(n)}</div>`).join('');

    $('ai-body').innerHTML = `
      <p class="sc-eyebrow">【预测细胞类型】</p>
      <p class="sc-type${refused ? ' sc-type--refuse' : ''}">${esc(ai.predicted_type)}</p>
      <div>
        <span class="sc-badge sc-badge--${confCls}">置信度 ${esc(ai.confidence_label)}${ai.confidence_score != null ? '（' + fmt(ai.confidence_score) + '）' : ''}</span>
        <span class="sc-chip sc-chip--blue">${esc(ai.source)}</span>
        <span class="sc-chip sc-chip--ghost">本地推断 · 未调用大模型</span>
      </div>
      ${cited}${viol}${notes}
      <p class="sc-eyebrow" style="margin-top:.7rem">【推断理由】</p>
      <div class="sc-reason">${esc(ai.reasoning || '（本次没有可展示的推理文本）')}</div>`;

    // 时间线
    if (ai.trace && ai.trace.length) {
      const icon = { Planner: '🧠', Annotator: '🔍', Critic: '⚖️' };
      $('trace-body').innerHTML = ai.trace.map((t) => {
        // approve 时编排层只给 status、reason 为 null，不补一句就成了空白。
        const why = t.output_summary && (t.output_summary.reason ||
          (t.output_summary.status === 'approve' ? '预测谱系与知识图谱最优候选一致，通过' : '') ||
          t.output_summary.predicted_type ||
          t.output_summary.top_candidate);
        return `
          <div class="sc-timeline-step">
            <div class="sc-timeline-dot">${icon[t.agent_name] || '•'}</div>
            <div class="sc-timeline-content">
              <div class="sc-timeline-title"><strong>${esc(t.agent_name)}</strong> · ${esc(t.action)}</div>
              <div class="sc-timeline-detail">${esc(why || '')}</div>
            </div>
          </div>`;
      }).join('');
      $('trace-card').hidden = false;
    }

    // KG 候选
    if (ai.kg_candidates && ai.kg_candidates.length) {
      $('kg-body').innerHTML = ai.kg_candidates.slice(0, 5).map((c) => `
        <p style="margin:.3rem 0">
          <span class="sc-chip sc-chip--gene">${esc(c.cl_id)}</span>
          <strong>${esc(c.name)}</strong>
          <span class="sc-chip">score ${fmt(c.score, 3)}</span>
          <span class="sc-sub">命中：${esc((c.matched_markers || []).join(', '))}</span>
        </p>`).join('');
      $('kg-card').hidden = false;
    }

    // 簇内异质性判定（逐细胞判型 → 这一簇该不该整体命名）
    const het = ai.heterogeneity;
    if (het && het.verdict) {
      $('het-body').innerHTML = renderHet(het);
      $('het-card').hidden = false;
    }

    // 证据表
    const ev = d.markers[state.cluster] || [];    $('evidence-body').innerHTML =
      '<thead><tr><th>基因</th><th>平均表达</th><th>簇内检出率</th><th>log2FC</th><th>差异打分</th><th>校正 p</th><th>类别</th></tr></thead><tbody>' +
      ev.map((r) => {
        const tag = r.is_mito ? '线粒体' : r.is_ribo ? '核糖体' : r.is_novel ? '未注释' : '符号';
        // 指数补齐两位，跟 Streamlit 端 format(v, '.2e') 的输出对齐（JS 会给 0.00e+0）
        const padj = r.pvals_adj == null ? '—'
          : r.pvals_adj < 0.01
            ? r.pvals_adj.toExponential(2).replace(/e[+-]\d$/, (m) => m + '0')
            : r.pvals_adj.toFixed(3);
        return `<tr>
          <td><code>${esc(r.gene)}</code></td>
          <td>${fmt(r.mean_expr, 3)}</td>
          <td>${r.pct_expressed == null ? '—' : (r.pct_expressed * 100).toFixed(1) + '%'}</td>
          <td>${fmt(r.logfoldchanges)}</td>
          <td>${fmt(r.scores, 1)}</td>
          <td>${padj}</td>
          <td><span class="sc-chip">${esc(tag)}</span></td>
        </tr>`;
      }).join('') + '</tbody>';
    $('evidence-card').hidden = false;

    // 闸门
    if (ai.gate && Object.keys(ai.gate).length) {
      $('gate-body').innerHTML = renderGate(ai.gate);
      $('gate-card').hidden = false;
    }
  }

  // 簇内异质性判定：逐细胞判型后这一簇该不该整体命名。同样只挑判断依据配人话标签。
  const HET_VERDICTS = {
    homogeneous: '单一类型，整体命名成立',
    heterogeneous: '混了多种细胞类型，不建议整体命名',
    ambiguous: '本次面板判读不出归属，不建议整体命名',
    not_assessed: '未做判定（没有可比参照或凑不齐候选类型）',
  };
  const HET_LABELS = [
    ['判定', (h) => HET_VERDICTS[h.verdict] || h.verdict],
    ['主标签', (h) => (h.dominant ? `${h.dominant} ${pct(h.dominant_prop)}` : '—')],
    ['第二种真实类型', (h) => (h.runner_up_type ? `${h.runner_up_type} ${pct(h.runner_up_type_share)}` : '无')],
    ['判读得出的细胞', (h) => pct(h.assigned_prop)],
    ['判读不出的细胞', (h) => pct(h.unassigned_prop)],
    ['标签分布熵', (h) => `${fmt(h.entropy_bits, 2)} / ${fmt(h.max_entropy_bits, 2)} bit`],
    ['本簇细胞', (h) => fmtInt(h.n_cells)],
    ['簇外参照细胞', (h) => fmtInt(h.n_reference_cells)],
  ];

  // not_assessed 时这些字段全是占位 0，摆出来只会让人以为"判读得出 0% = 全读不出"。
  const HET_HIDDEN_IF_UNASSESSED = new Set(
    ['主标签', '第二种真实类型', '判读得出的细胞', '判读不出的细胞', '标签分布熵']);

  function renderHet(h) {
    const kv = (k, v) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`;
    const chips = (items) => items.join('');
    const skip = h.verdict === 'not_assessed' ? HET_HIDDEN_IF_UNASSESSED : new Set();
    const rows = HET_LABELS.filter(([k]) => !skip.has(k)).map(([k, get]) => kv(k, get(h))).join('');
    const dist = chips((h.distribution || []).map((r) =>
      `<span class="sc-chip">${esc(r.label)} ${fmtInt(r.n_cells)} ｜ ${pct(r.share)}</span>`));
    const zs = chips(Object.entries(h.z_median || {}).sort((a, b) => b[1] - a[1])
      .map(([t, v]) => `<span class="sc-chip">${esc(t)} z ${fmt(v, 2)}</span>`));
    const used = Object.entries(h.types_used || {}).map(([t, gs]) =>
      `<p class="sc-eyebrow" style="margin-top:.6rem">${esc(t)} · ${gs.length} 个判别基因</p>` +
      chips(gs.map((g) => `<span class="sc-chip sc-chip--gene">${esc(g)}</span>`))).join('');
    const blocked = Object.entries(h.types_unresolvable || {}).map(([t, v]) =>
      `<p class="sc-sub">${esc(t)}：只有 ${v.n_discriminative}/${v.n_panel_genes} 个判别基因，` +
      `与 ${esc((v.shared_with || []).join('、'))} 分不开 → 不参与判型。</p>`).join('');
    const sub = (h.subclusters || []).length ? `
        <p class="sc-eyebrow" style="margin-top:.6rem">亚簇拆分（${h.subclusters.length} 个亚簇，
          加权纯度 ${pct(h.subcluster_mean_purity)}，比整簇主标签高 ${pct(h.subcluster_purity_gain)}）
          → ${esc(h.coherent_substructure ? '分歧成片存在，建议提高分辨率拆开' : '分歧零散，更像噪声或过渡态')}</p>
        ${chips(h.subclusters.map((r) =>
          `<span class="sc-chip">亚簇 ${esc(r.subcluster)} ｜ ${fmtInt(r.n_cells)} 细胞 ｜ ` +
          `主 ${esc(r.dominant)} ${pct(r.purity)}</span>`))}` : '';
    return `<dl class="sc-kv">${rows}</dl>
      <p class="sc-sub" style="margin-top:.5rem">${esc(h.reason)}</p>
      ${dist ? `<p class="sc-eyebrow" style="margin-top:.6rem">逐细胞判型分布</p>${dist}` : ''}
      ${zs ? `<p class="sc-eyebrow" style="margin-top:.6rem">各类型判型分中位数（判别基因平均表达，相对簇外参照细胞标准化）</p>${zs}` : ''}
      ${used}
      ${blocked ? `<p class="sc-eyebrow" style="margin-top:.6rem">与别的类型分不开</p>${blocked}` : ''}
      ${sub}`;
  }

  // 闸门结构体有 17 个字段，直接平铺出来的话字段名会连成一片（实测很难读）。
  // 这里只挑判断依据，配上人话标签。
  const GATE_LABELS = [
    ['判定', (g) => (g.ok ? '证据充足，可以解读' : '证据不足，拒绝下结论')],
    ['输入 marker 数', (g) => String(g.n_input)],
    ['可用证据数（名字读得懂 + 表达达标 + 有分类价值）', (g) => String(g.n_usable)],
    ['表达达标数（检出率>0.2 且 mean_expr>0.3，不看基因名）', (g) => String(g.n_strong)],
    ['未注释基因占比', (g) => fmt(g.novel_ratio, 2)],
    ['线粒体基因占比', (g) => fmt(g.mito_ratio, 2)],
    ['簇内细胞数', (g) => fmtInt(g.n_cells)],
    ['拒绝理由', (g) => (g.reasons || []).join('；') || '无'],
    ['数据质量提示', (g) => (g.warnings || []).join('；') || '无'],
    ['补救建议', (g) => (g.suggestions || []).join('；') || '无'],
  ];
  const GATE_LISTS = [
    ['被判为未注释的基因', 'unannotated'],
    ['名字合法但表达不达标的基因', 'rejected_by_expression'],
    ['无分类价值的基因（持家/核糖体等）', 'uninformative'],
  ];

  function renderGate(gate) {    const kv = (k, v) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`;
    const rows = GATE_LABELS.map(([k, get]) => kv(k, get(gate))).join('');
    const lists = GATE_LISTS.filter(([, key]) => (gate[key] || []).length).map(([k, key]) => {
      const items = gate[key];
      const shown = items.slice(0, 8).map((x) => `<code>${esc(x)}</code>`).join('、');
      return `<p class="sc-eyebrow" style="margin-top:.6rem">${esc(k)} · ${items.length} 个</p>
              <p class="sc-sub">${shown}${items.length > 8 ? ' …' : ''}</p>`;
    }).join('');
    return `<dl class="sc-kv">${rows}</dl>${lists}`;
  }

  // ----------------------------------------------------------------- 条形图
  function renderBarplot() {
    const ev = (state.data.markers[state.cluster] || []).slice().reverse();
    const trace = {
      type: 'bar', orientation: 'h',
      y: ev.map((r) => r.gene), x: ev.map((r) => r.scores || 0),
      marker: { color: ev.map((r) => (r.is_mito || r.is_novel) ? '#94a3b8' : clusterColor(state.cluster)) },
      text: ev.map((r) => `检出率 ${r.pct_expressed == null ? '—' : (r.pct_expressed * 100).toFixed(1) + '%'} ｜ 平均表达 ${fmt(r.mean_expr, 3)}`),
      hoverinfo: 'text+y',
    };
    Plotly.react('barplot', [trace], {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: DARK_FONT,
      // 左边距交给 automargin 算：写死 90 时，窄屏上基因名会被裁掉半截。
      margin: { l: 10, r: 16, t: 8, b: 34 },
      xaxis: { title: { text: '差异打分', font: { size: 11, color: '#c6d5f2' } }, gridcolor: DARK_GRID },
      yaxis: { automargin: true, tickfont: { family: 'ui-monospace, Consolas, monospace', size: 11, color: '#c6d5f2' } },
    }, { displayModeBar: false, responsive: true });

    const nMito = ev.filter((r) => r.is_mito).length;
    const nNovel = ev.filter((r) => r.is_novel).length;
    $('bar-caption').textContent =
      `灰色 = 线粒体或读不懂的基因名，不计入可用证据。本簇 ${ev.length} 个 Top marker 中，线粒体 ${nMito} 个、未注释 ${nNovel} 个。`;
  }

  // ----------------------------------------------------------------- 簇表
  function renderClusterTable() {
    const head = ['簇', '细胞数', '占比', '表达达标 marker', '线粒体marker', '中位UMI', '中位MT%', '证据', 'AI 判定'];
    const body = state.data.clusters.map((c) => {
      const ai = state.data.ai[String(c.cluster)] || {};
      const ok = c.evidence_ok
        ? '<span class="sc-chip sc-chip--gene" style="background:rgba(74,222,128,.14);color:#4ade80">充足</span>'
        : '<span class="sc-chip sc-chip--gene" style="background:rgba(251,113,133,.14);color:#fb7185">不足</span>';
      const verdict = ai.critique
        ? (ai.critique.status === 'approve'
            ? '<span class="sc-chip" style="background:rgba(74,222,128,.14);color:#4ade80">通过</span>'
            : '<span class="sc-chip" style="background:rgba(251,113,133,.14);color:#fb7185">拒绝</span>')
        : '—';
      return `<tr data-cluster="${esc(c.cluster)}" style="cursor:pointer">
        <td><span class="sc-chip sc-chip--gene" style="border-color:${clusterColor(c.cluster)};color:${clusterColor(c.cluster)}">${esc(c.cluster)}</span></td>
        <td>${fmtInt(c.n_cells)}</td>
        <td>${fmt(c.pct_cells, 1)}%</td>
        <td>${fmtInt(c.n_strong_markers)}</td>
        <td>${fmt(c.mito_marker_ratio, 2)}</td>
        <td>${fmtInt(c.median_counts)}</td>
        <td>${fmt(c.median_pct_mito, 2)}</td>
        <td>${ok}</td>
        <td>${verdict}</td>
      </tr>`;
    }).join('');
    const t = $('cluster-table');
    t.innerHTML = `<thead><tr>${head.map((h) => `<th>${h}</th>`).join('')}</tr></thead><tbody>${body}</tbody>`;
    t.querySelectorAll('tbody tr').forEach((tr) => {
      tr.addEventListener('click', () => selectCluster(tr.dataset.cluster));
    });
  }

  // ----------------------------------------------------------------- 分辨率
  function renderResolution() {
    const card = $('resolution-card');
    const diff = state.data.resolution_diff;
    if (!diff) { card.hidden = true; return; }
    card.hidden = false;
    $('resolution-body').innerHTML = `
      <dl class="sc-kv">
        <div><dt>簇数</dt><dd>${diff.n_before} → ${diff.n_after}</dd></div>
        <div><dt>中位簇规模</dt><dd>${fmt(diff.median_before, 0)} → ${fmt(diff.median_after, 0)}</dd></div>
        <div><dt>最大簇</dt><dd>${fmtInt(diff.max_before)} → ${fmtInt(diff.max_after)}</dd></div>
        <div><dt>拆分 / 合并</dt><dd>${diff.n_splits} / ${diff.n_merges}</dd></div>
      </dl>
      ${(diff.splits || []).slice(0, 4).map((s) =>
        `<div class="sc-note sc-note--info">簇 ${esc(s.from)} 被拆成 ${s.into.length} 份 [${s.sizes.join(', ')}]，纯度 ${fmt(s.purity * 100, 0)}%</div>`).join('')}
      ${(diff.merges || []).slice(0, 4).map((m) =>
        `<div class="sc-note">簇 ${esc(m.from.join(', '))} 合并进 ${esc(m.into)}</div>`).join('')}`;
  }

  // ----------------------------------------------------------------- 运行信息
  function renderRunInfo() {
    const d = state.data, qc = d.qc;
    const funnel = [
      ['输入细胞', fmtInt(qc.n_cells_start)], ['硬过滤后', fmtInt(qc.n_cells_after_hard_filter)],
      ['MT 过滤后', fmtInt(qc.n_cells_after_mito_filter)], ['最终细胞', fmtInt(qc.n_cells_final)],
      ['保留基因', fmtInt(qc.n_genes_final)], ['滤除比例', fmt(qc.cells_removed_pct, 1) + '%'],
    ];
    const rows = Object.entries(d.timings || {}).map(([k, v]) =>
      `<tr><td>${esc(k)}</td><td>${fmt(v, 2)}</td></tr>`).join('');
    const funnelHtml = funnel.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${v}</dd></div>`).join('');
    $('run-body').innerHTML = `
      <dl class="sc-kv">${funnelHtml}</dl>
      <p class="sc-eyebrow" style="margin-top:.7rem">各步骤耗时（秒）</p>
      <div class="sc-table-scroll" style="max-height:220px"><table class="sc-table">
        <thead><tr><th>步骤</th><th>秒</th></tr></thead><tbody>${rows}</tbody></table></div>
      ${(d.warnings || []).map((w) => `<div class="sc-note">${esc(w)}</div>`).join('')}
      <p class="sc-sub" style="margin-top:.7rem">复现：<code>python scripts/export_site_data.py</code>（seed 固定，同一台机器结果一致）</p>`;
  }

  // ----------------------------------------------------------------- 启动
  boot().catch((err) => {
    document.body.innerHTML =
      `<div class="wrap"><div class="sc-card"><p class="sc-title" style="color:#be123c">加载失败</p>
       <p class="sc-sub">${esc(err && err.message || err)}<br>
       静态站需要通过 HTTP 打开（<code>fetch</code> 不允许 <code>file://</code>）。
       本地预览：<code>python -m http.server --directory web 8080</code></p></div></div>`;
  });
})();
