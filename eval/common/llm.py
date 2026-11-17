"""Minimal OpenAI-compatible chat client (stdlib only). Key: $LLM_API_KEY, else ~/.omlx/settings.json auth.api_key."""
import json, os, time, urllib.request
def api_key():
    k = os.environ.get('LLM_API_KEY')
    if k: return k
    try: return json.load(open(os.path.expanduser('~/.omlx/settings.json')))['auth']['api_key']
    except Exception: return ''
def chat(messages, model, base_url=None, max_tokens=2048, temperature=0.0, extra=None, timeout=600):
    base_url = base_url or os.environ.get('LLM_BASE_URL', 'http://localhost:8000/v1')
    body = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
    if extra: body.update(extra)
    req = urllib.request.Request(base_url.rstrip('/') + '/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + api_key()})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r: d = json.load(r)
    ch = d['choices'][0]['message']; usage = d.get('usage', {})
    return dict(text=ch.get('content') or '', reasoning=ch.get('reasoning_content') or ch.get('reasoning') or '',
                usage=usage, seconds=round(time.time() - t0, 2))
def models(base_url=None):
    base_url = base_url or os.environ.get('LLM_BASE_URL', 'http://localhost:8000/v1')
    req = urllib.request.Request(base_url.rstrip('/') + '/models', headers={'Authorization': 'Bearer ' + api_key()})
    with urllib.request.urlopen(req, timeout=30) as r: return [m['id'] for m in json.load(r)['data']]
