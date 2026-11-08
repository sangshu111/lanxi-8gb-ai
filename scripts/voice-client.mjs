#!/usr/bin/env node
/**
 * 语音服务客户端：服务没起来就自动拉起，用完靠服务端空闲退出回收显存
 *
 * 环境变量：
 *   GSV_PYTHON   python 解释器路径（venv 里的那个）
 *   GSV_SERVICE  voice-service.py 路径
 *   GSV_PORT     服务端口（默认 9880）
 *
 * 用法：
 *   node voice-client.mjs --text "主人，你回来啦。" [--out out.wav] [--idle 300]
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const args = process.argv.slice(2);
const opt = { text: '', out: '', idle: '300' };
for (let i = 0; i < args.length; i += 2) {
  const k = args[i].replace(/^--/, '');
  if (k in opt) opt[k] = args[i + 1];
}
if (!opt.text) {
  console.log(JSON.stringify({ ok: false, error: 'need --text' }));
  process.exit(1);
}

const PYTHON = process.env.GSV_PYTHON || 'python';
const SERVICE = process.env.GSV_SERVICE || './voice-service.py';
const PORT = process.env.GSV_PORT || '9880';
const URL = `http://127.0.0.1:${PORT}`;

async function health() {
  try {
    const r = await fetch(URL + '/health', { signal: AbortSignal.timeout(1500) });
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}

async function ensureServer() {
  const h = await health();
  if (h) return { started: false };
  const fd = fs.openSync(path.join(path.dirname(SERVICE), 'voice-service.log'), 'a');
  const child = spawn(PYTHON, [SERVICE, '--idle', String(opt.idle)], {
    detached: true,
    stdio: ['ignore', fd, fd],
    windowsHide: true,
  });
  child.unref();
  const t0 = Date.now();
  while (Date.now() - t0 < 120000) {
    await new Promise((r) => setTimeout(r, 1000));
    if (await health()) return { started: true, bootMs: Date.now() - t0 };
  }
  return { started: false, error: 'boot timeout' };
}

const boot = await ensureServer();
if (boot.error) {
  console.log(JSON.stringify({ ok: false, ...boot }));
  process.exit(1);
}

const t1 = Date.now();
const r = await fetch(URL + '/say', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: opt.text, out: opt.out || undefined }),
  signal: AbortSignal.timeout(180000),
});
const j = await r.json();
console.log(JSON.stringify({ ...j, clientMs: Date.now() - t1, serverBooted: boot.started, bootMs: boot.bootMs || 0 }));
