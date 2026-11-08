#!/usr/bin/env node
/**
 * 通过 QQ 官方 Bot 发送语音/图片（C2C 私聊）
 *
 * 环境变量（必填，不要写死在代码里）：
 *   QQ_APP_ID       官方 Bot AppID
 *   QQ_APP_SECRET   官方 Bot AppSecret
 *   QQ_OWNER_OPENID 接收者的 openid
 *
 * 用法：
 *   node qq-send.mjs --type voice|image --file <路径> [--file <路径> ...]
 */
import fs from 'node:fs';
import path from 'node:path';

const APP_ID = process.env.QQ_APP_ID || '';
const APP_SECRET = process.env.QQ_APP_SECRET || '';
const OWNER = process.env.QQ_OWNER_OPENID || '';

const argv = process.argv.slice(2);
let type = 'voice';
const files = [];
for (let i = 0; i < argv.length; i++) {
  if (argv[i] === '--type') type = argv[++i];
  else if (argv[i] === '--file') files.push(argv[++i]);
}
if (!APP_ID || !APP_SECRET || !OWNER) {
  console.log(JSON.stringify({ ok: false, error: 'missing QQ_APP_ID / QQ_APP_SECRET / QQ_OWNER_OPENID' }));
  process.exit(1);
}
if (!files.length) {
  console.log(JSON.stringify({ ok: false, error: 'need --file' }));
  process.exit(1);
}

const FILE_TYPE = type === 'image' ? 1 : 3;   // 1=图片 3=语音

const token = await (async () => {
  const r = await fetch('https://bots.qq.com/app/getAppAccessToken', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ appId: APP_ID, clientSecret: APP_SECRET }),
    signal: AbortSignal.timeout(20000),
  });
  return (await r.json()).access_token || '';
})();
if (!token) {
  console.log(JSON.stringify({ ok: false, error: 'no access_token' }));
  process.exit(1);
}

for (const file of files) {
  if (!fs.existsSync(file)) {
    console.log(JSON.stringify({ ok: false, file, error: 'not found' }));
    continue;
  }
  const b64 = fs.readFileSync(file).toString('base64');
  const up = await fetch(`https://api.sgroup.qq.com/v2/users/${OWNER}/files`, {
    method: 'POST',
    headers: { Authorization: 'QQBot ' + token, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      file_type: FILE_TYPE,
      file_data: b64,
      srv_send_msg: false,
      file_name: path.basename(file),
    }),
    signal: AbortSignal.timeout(120000),
  });
  const uj = await up.json().catch(() => null);
  const info = uj && uj.file_info;
  if (!info) {
    console.log(JSON.stringify({ ok: false, file, stage: 'upload', status: up.status }));
    continue;
  }
  const s = await fetch(`https://api.sgroup.qq.com/v2/users/${OWNER}/messages`, {
    method: 'POST',
    headers: { Authorization: 'QQBot ' + token, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      content: '',
      msg_type: 7,
      media: { file_info: info },
      msg_seq: Math.floor(Math.random() * 10000),
    }),
    signal: AbortSignal.timeout(30000),
  });
  console.log(JSON.stringify({ ok: s.ok, status: s.status, file: path.basename(file) }));
  await new Promise((r) => setTimeout(r, 1500));
}
