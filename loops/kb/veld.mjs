// A small veld client: upsert a memory by external id, recall by query or by tags. The URL and key
// come from a project's connection.env, as veld-desktop writes it.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';
import { request } from 'node:http';

function connection() {
  const env = readFileSync(join(homedir(), '.config', 'veld', 'commitwork', 'connection.env'), 'utf8');
  const get = (k) => (new RegExp(`^${k}=(.*)$`, 'm').exec(env) || [])[1]?.trim().replace(/^["']|["']$/g, '');
  return { url: get('VELD_API_URL') || 'http://127.0.0.1:3030', key: get('VELD_API_KEY') };
}

export class VeldUnavailable extends Error {}

function post(path, body, timeoutMs = 120000) {
  const { url, key } = connection();
  const data = JSON.stringify(body);
  return new Promise((resolve, reject) => {
    const req = request(`${url}${path}`, { method: 'POST', agent: false, headers: { 'content-type': 'application/json', 'content-length': Buffer.byteLength(data), ...(key ? { 'X-API-Key': key } : {}) } }, (res) => {
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => {
        const text = Buffer.concat(chunks).toString('utf8');
        if (res.statusCode >= 200 && res.statusCode < 300) { try { resolve(JSON.parse(text)); } catch { resolve(text); } }
        else reject(new VeldUnavailable(`veld ${path} answered ${res.statusCode}: ${text.slice(0, 200)}`));
      });
    });
    req.setTimeout(timeoutMs, () => req.destroy(new VeldUnavailable(`veld ${path} timed out after ${timeoutMs} ms`)));
    req.on('error', (e) => reject(e instanceof VeldUnavailable ? e : new VeldUnavailable(`veld ${path}: ${e.code || e.message}`)));
    req.end(data);
  });
}

export const health = () => new Promise((resolve) => {
  const { url } = connection();
  const req = request(`${url}/health`, { agent: false }, (res) => { res.resume(); resolve(res.statusCode === 200); });
  req.setTimeout(5000, () => { req.destroy(); resolve(false); });
  req.on('error', () => resolve(false));
  req.end();
});

export const upsert = ({ userId, externalId, content, tags = [], memoryType = 'Learning' }) =>
  post('/api/upsert', { user_id: userId, external_id: externalId, content, tags, memory_type: memoryType, changed_by: 'veld-kb' });

// [{ id, content, tags, score }] for a query, optionally limited to memories carrying any of tags.
export async function recall({ userId, query, limit = 6, tags = null }) {
  const r = await post('/api/recall', { user_id: userId, query, limit, mode: 'hybrid', ...(tags ? { tags } : {}) });
  return (r.memories || []).map((m) => ({ id: m.id, content: m.experience?.content ?? m.content, tags: m.experience?.tags ?? m.tags ?? [], score: m.score }));
}

export async function byTags({ userId, tags, limit = 50 }) {
  const r = await post('/api/recall/tags', { user_id: userId, tags, limit });
  return r.memories || [];
}
