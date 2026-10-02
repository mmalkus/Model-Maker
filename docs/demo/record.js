// Records the "Build with AI" demo: demo credit data -> tag roles -> AI builds a
// logistic-regression PD model. Writes raw.webm + segments.json (the waiting
// stretches that post-processing speeds up).
const { chromium } = require('playwright');
const fs = require('fs');

const BASE = 'http://127.0.0.1:8001';
const W = 1600, H = 900;
const GOAL =
  'Build a PD model for default_flag: logistic regression on weight-of-evidence ' +
  'binned features, a 70/30 train/test split, and validate it with Gini, KS and PSI.';

(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({
    viewport: { width: W, height: H },
    recordVideo: { dir: 'video', size: { width: W, height: H } },
  });
  const t0 = Date.now();
  const now = () => (Date.now() - t0) / 1000;
  const segments = [];
  const log = (...a) => console.log(`[${now().toFixed(1)}s]`, ...a);

  const p = await ctx.newPage();
  p.on('dialog', (d) => d.accept());
  await p.goto(BASE);
  await p.waitForTimeout(1500);

  // ---- overlay: caption, fast-forward badge, fake cursor ----------------------
  const installOverlay = () =>
    p.evaluate(() => {
      if (document.getElementById('demo-cap')) return;
      const css = document.createElement('style');
      css.textContent = `
        #demo-cap{position:fixed;left:50%;bottom:28px;transform:translateX(-50%);z-index:99999;
          background:rgba(17,24,39,.88);color:#fff;font:600 22px/1.35 system-ui,sans-serif;
          padding:12px 22px;border-radius:12px;max-width:1100px;text-align:center;
          box-shadow:0 8px 30px rgba(0,0,0,.25);transition:opacity .3s;pointer-events:none}
        #demo-cap small{display:block;font-weight:400;font-size:16px;opacity:.85;margin-top:2px}
        #demo-ff{position:fixed;left:50%;top:76px;transform:translateX(-50%);z-index:99999;
          background:#7c3aed;color:#fff;font:700 18px system-ui,sans-serif;padding:8px 18px;
          border-radius:999px;display:none;pointer-events:none;box-shadow:0 6px 20px rgba(124,58,237,.4)}
        #demo-cur{position:fixed;width:22px;height:22px;margin:-11px 0 0 -11px;border-radius:50%;
          background:rgba(239,68,68,.35);border:2px solid rgba(239,68,68,.9);z-index:100000;
          pointer-events:none;transition:transform .12s;left:-50px;top:-50px}
        #demo-cur.down{transform:scale(.6);background:rgba(239,68,68,.7)}`;
      document.head.appendChild(css);
      for (const id of ['demo-cap', 'demo-ff', 'demo-cur']) {
        const d = document.createElement('div');
        d.id = id;
        document.body.appendChild(d);
      }
      document.getElementById('demo-cap').style.opacity = '0';
      const cur = document.getElementById('demo-cur');
      addEventListener('mousemove', (e) => { cur.style.left = e.clientX + 'px'; cur.style.top = e.clientY + 'px'; }, true);
      addEventListener('mousedown', () => cur.classList.add('down'), true);
      addEventListener('mouseup', () => cur.classList.remove('down'), true);
    });
  await installOverlay();
  const dismiss = p.getByRole('button', { name: 'Dismiss', exact: true });
  if (await dismiss.count()) await dismiss.click();
  // Clear any build left over from an earlier attempt.
  await p.evaluate(() => fetch('/api/agent/builds/current/discard', { method: 'POST' })).catch(() => {});
  await p.waitForTimeout(1500);

  const caption = async (title, sub = '') => {
    await p.evaluate(([t, s]) => {
      const c = document.getElementById('demo-cap');
      if (!t) { c.style.opacity = '0'; return; }
      c.innerHTML = t + (s ? `<small>${s}</small>` : '');
      c.style.opacity = '1';
    }, [title, sub]);
  };
  const ffBadge = (text) =>
    p.evaluate((t) => {
      const b = document.getElementById('demo-ff');
      b.style.display = t ? 'block' : 'none';
      b.textContent = t || '';
    }, text);

  const moveTo = async (loc) => {
    await loc.scrollIntoViewIfNeeded();
    const box = await loc.boundingBox();
    const x = box.x + box.width / 2, y = box.y + box.height / 2;
    await p.mouse.move(x, y, { steps: 18 });
    await p.waitForTimeout(250);
    return { x, y };
  };
  const click = async (loc, pause = 500) => {
    const { x, y } = await moveTo(loc);
    await p.mouse.down();
    await p.waitForTimeout(80);
    await p.mouse.up();
    await p.waitForTimeout(pause);
  };
  const api = (path, method = 'GET') =>
    p.evaluate(async ([u, m]) => (await fetch(u, { method: m })).json(), [BASE + '/api' + path, method]);
  const fitView = async () => {
    const b = p.locator('.react-flow__controls-fitview');
    if (await b.count()) await b.click().catch(() => {});
  };

  // ---- 0. intro --------------------------------------------------------------
  await click(p.getByRole('button', { name: 'New', exact: true }), 800);
  await p.mouse.move(700, 450, { steps: 10 });
  await caption('Model-Maker · Build with AI', 'From synthetic credit data to a logistic-regression PD model');
  await p.waitForTimeout(3500);

  // ---- 1. demo data block -------------------------------------------------------
  await caption('Step 1 — Add the demo credit data block', 'A synthetic retail book of term loans and credit cards, generated in memory');
  await p.waitForTimeout(1500);
  await click(p.getByText('Demo credit data (PD/LGD/CCF)'), 1200);
  const node = p.locator('.react-flow__node').first();
  await click(node.locator('div').first(), 1000);
  await caption('Step 1 — Run it', '5,000 facilities, one row each, with a 12-month default flag');
  await click(p.getByRole('button', { name: 'Run', exact: true }), 2500);

  // ---- 2. tag roles -------------------------------------------------------------
  await caption('Step 2 — Tag column roles', 'The AI only ever sees column names, types, roles and summary stats — never rows');
  await click(node.getByText('out').first(), 1800);

  const tag = async (col, role, label) => {
    const sel = p.locator(`span[title="${col}"]`).locator('xpath=../..').locator('select');
    await moveTo(sel);
    await sel.selectOption(role);
    log('tagged', col, role);
    await p.waitForTimeout(label ? 900 : 350);
  };
  await caption('Step 2 — Tag the target and the keys', 'default_flag = Target · application_id = ID · reference_date = Date');
  await tag('application_id', 'id', true);
  await tag('reference_date', 'date', true);
  await tag('default_flag', 'target', true);
  await caption('Step 2 — Exclude post-default columns', 'Leaky outcome fields can never be used as features — enforced by the tools, not just the prompt');
  for (const c of ['default_date', 'balance_at_default', 'recovery_amount', 'workout_cost', 'workout_months', 'cure_flag'])
    await tag(c, 'excluded', false);
  await p.waitForTimeout(1200);

  // Re-run so the block is current with its new roles.
  await click(p.getByRole('button', { name: 'Close', exact: true }), 600);
  await click(node.locator('div').first(), 600);
  await caption('Step 2 — Re-run so the roles are applied');
  await click(p.getByRole('button', { name: 'Run', exact: true }), 2500);

  // ---- 3. build with AI ----------------------------------------------------------
  await caption('Step 3 — Select the block and click “Build with AI”');
  await click(node.locator('div').first(), 700);
  await click(p.getByRole('button', { name: 'Build with AI' }), 1200);
  await caption('Step 3 — Describe the model you want');
  const ta = p.locator('textarea').first();
  await click(ta, 300);
  await p.keyboard.type(GOAL, { delay: 22 });
  await p.waitForTimeout(1200);
  await caption('Step 3 — Plan first: nothing changes until you approve');
  const prev = await api('/agent/builds/current?cursor=0');
  const prevId = prev.build ? prev.build.id : null;
  await click(p.getByRole('button', { name: 'Plan the build' }), 1000);

  // ---- 4/5. drive the build ------------------------------------------------------
  let seg = null;
  const startFF = async (label) => {
    if (seg) return;
    seg = { start: now(), label };
    await ffBadge('⏩  Fast-forward — ' + label);
    log('FF start', label);
  };
  const endFF = async () => {
    if (!seg) return;
    seg.end = now();
    segments.push(seg);
    log('FF end', seg.label, (seg.end - seg.start).toFixed(0) + 's');
    seg = null;
    await ffBadge('');
    await p.waitForTimeout(300);
  };

  const deadline = Date.now() + 60 * 60 * 1000;
  let lastPhase = '';
  let approved = false;
  let answered = 0;
  while (Date.now() < deadline) {
    const st = await api('/agent/builds/current?cursor=0');
    const b = st.build && st.build.id !== prevId ? st.build : null;
    const phase = b ? b.phase : 'none';
    if (phase !== lastPhase) { log('phase', phase, 'busy', st.busy); lastPhase = phase; }

    if (phase === 'preflight') {
      await endFF();
      await caption('Preflight checks the data before any AI call');
      await p.waitForTimeout(2500);
      const proceed = p.getByRole('button', { name: 'Proceed anyway' });
      if (await proceed.isEnabled()) await click(proceed, 1500);
      else { log('preflight blocked', JSON.stringify(b.preflight)); break; }
    } else if (phase === 'planning') {
      await caption('Step 4 — The AI plans the model', 'Reading the column roles and summary statistics…');
      await startFF('AI is planning');
      await fitView();
      await p.waitForTimeout(1500);
    } else if (phase === 'awaiting_approval') {
      if (st.busy) { await p.waitForTimeout(1000); continue; }
      await endFF();
      await fitView();
      await caption('Step 4 — Review the plan', 'Ghost blocks on the canvas, step-by-step details in the panel');
      await p.waitForTimeout(1500);
      // Scroll the plan into view so the viewer can read it.
      const panel = p.locator('strong', { hasText: 'Build with AI' }).locator('xpath=../..');
      for (let i = 0; i < 6; i++) {
        await panel.evaluate((el) => el.scrollBy({ top: 160, behavior: 'smooth' })).catch(() => {});
        await p.waitForTimeout(900);
      }
      const approve = p.getByRole('button', { name: 'Approve and build' });
      if (await approve.isEnabled()) {
        await caption('Step 4 — Approve and build');
        await click(approve, 1200);
        approved = true;
      } else {
        await caption('The AI asked a question — answer it to continue');
        const fb = p.locator('textarea').first();
        await click(fb, 200);
        await p.keyboard.type('Use your recommended defaults.', { delay: 25 });
        await click(p.getByRole('button', { name: 'Re-plan with feedback' }), 1000);
      }
    } else if (phase === 'building' || phase === 'final_run') {
      await caption(
        phase === 'building' ? 'Step 5 — The AI builds, runs and checks each block' : 'Step 5 — Final run on the full data',
        'It reads the summary stats, fixes failures, and the canvas stays read-only meanwhile'
      );
      await startFF('AI is building');
      await fitView();
      await p.waitForTimeout(2000);
    } else if (phase === 'awaiting_input') {
      if (st.busy) { await p.waitForTimeout(1000); continue; }
      await endFF();
      await caption('The AI asks when it is stuck');
      await p.waitForTimeout(2000);
      const q = p.locator('textarea').first();
      await click(q, 200);
      await p.keyboard.type('Use your best judgement and keep going.', { delay: 25 });
      await click(p.getByRole('button', { name: 'Send', exact: true }), 1000);
      if (++answered > 5) break;
    } else if (['done', 'done_with_errors', 'stopped', 'failed', 'discarded'].includes(phase)) {
      await endFF();
      log('final', phase, b.error || '');
      break;
    } else {
      await p.waitForTimeout(1000);
    }
  }
  await endFF();

  // ---- 6. result ---------------------------------------------------------------
  await p.waitForTimeout(1500);
  await fitView();
  await caption('Step 6 — Done: a full PD pipeline, built and run', 'Every AI-built block carries an AI badge and provenance; one Undo reverts the whole build');
  await p.waitForTimeout(4000);
  // Bring the report's headline table into view.
  await p.evaluate(() => {
    const el = [...document.querySelectorAll('div')].find(
      (d) => d.style.whiteSpace === 'pre-wrap' && d.style.overflowY === 'auto' && d.textContent.includes('Headline results'));
    if (!el) return;
    el.scrollIntoView({ block: 'center', behavior: 'smooth' });
    el.style.maxHeight = '420px';
    const i = el.textContent.indexOf('Headline results');
    setTimeout(() => el.scrollTo({ top: (el.scrollHeight * i) / el.textContent.length - 10, behavior: 'smooth' }), 800);
  });
  await caption('Step 6 — The AI writes a build report', 'Test Gini, KS and PSI — plus what it did and why');
  await p.waitForTimeout(6500);

  // Open the Gini/KS comparison table the AI built.
  const st = await api('/agent/builds/current?cursor=0');
  const tableRes = (st.build.results || []).find((r) => r.type === 'dataframe');
  if (tableRes) {
    const closePanel = p.locator('button[title="Close (the build keeps going)"]');
    if (await closePanel.count()) await click(closePanel, 800);
    await fitView();
    await p.waitForTimeout(800);
    const tnode = p.locator(`.react-flow__node[data-id="${tableRes.block}"]`);
    await caption('Step 6 — Inspect any block the AI built', 'Here: Gini, KS and AUC on train vs test');
    await click(tnode.getByText(tableRes.port).first(), 1500);
    const full = p.getByRole('button', { name: 'View full table' });
    if (await full.count()) {
      await click(full, 1500);
      await p.waitForTimeout(5000);
      await p.screenshot({ path: 'table.png' });
      await click(p.getByRole('button', { name: 'Close', exact: true }).first(), 800);
      const c2 = p.getByRole('button', { name: 'Close', exact: true }).first();
      if (await c2.isVisible().catch(() => false)) await click(c2, 800);
      await p.mouse.click(700, 200);
    }
  }
  await fitView();
  await caption('Model-Maker · Build with AI', 'Compile the graph to a single, human-readable Python script whenever you like');
  await p.mouse.move(1500, 860, { steps: 10 });
  await p.waitForTimeout(4500);
  await p.screenshot({ path: 'final.png' });

  fs.writeFileSync('segments.json', JSON.stringify(segments, null, 2));
  const vpath = await p.video().path();
  await ctx.close();
  await browser.close();
  fs.renameSync(vpath, 'raw.webm');
  log('saved raw.webm', JSON.stringify(segments));
})();
