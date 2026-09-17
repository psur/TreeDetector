"""Local-only benchmark review. Run with .venv-prep/Scripts/python.exe."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.dataset.final_decisions import ALLOWED, CANONICAL_FIELDS, read_table, validate_decisions

VALIDATOR_COMMAND = r'.venv-prep\Scripts\python.exe main.py --config config/config.yaml prepare --dataset benchmark_v5 --validate-decisions'


def digest(data):
    return hashlib.sha256(data).hexdigest()


class ReviewStore:
    def __init__(self, project=PROJECT, validator=validate_decisions):
        self.project = Path(project).resolve()
        self.root = self.project/'results/benchmark_v5_review'
        self.canonical = self.project/'config/benchmark_v5_decisions.csv'
        if self.canonical.resolve() != self.canonical:
            raise ValueError('Canonical file must not be a symlink/junction')
        rows, _ = read_table(self.root/'human_review_required.csv')
        self.items = {r['review_id']:r for r in rows}
        if len(rows) != len(self.items):
            raise ValueError('Duplicate review IDs in review data')
        catalog = {r['review_id']:r for r in json.loads((self.project/'config/benchmark_v5_review_catalog.json').read_text(encoding='utf-8'))}
        required = {k for k,r in catalog.items() if r.get('review_status') != 'AUTO_REPAIRED_DUPLICATE_VERTEX'}
        if set(self.items) != required:
            raise ValueError('Review IDs do not match canonical catalog')
        self.assets = {}
        for identity, item in self.items.items():
            for key in ('issue_type','content_sha256','annotation_a','annotation_b'):
                if item[key] != catalog[identity][key]:
                    raise ValueError('Review metadata disagrees with catalog: '+identity)
            item['assets'] = []
            directory = self.root/'Unresolved'/identity
            if directory.resolve() != directory:
                raise ValueError('Unsafe asset directory')
            for asset in sorted(directory.glob('*.jpg')):
                if asset.resolve() != asset:
                    raise ValueError('Unsafe asset path')
                key = digest(str(asset).encode())
                self.assets[key] = asset
                item['assets'].append(dict(name=asset.stem, url='/asset/'+key))
        self.validator = validator
        self.lock = threading.Lock()
        self.cached = None
        self.snapshot()  # Validate saved decisions before serving; never rewrite them.

    def _rows(self):
        before = self.canonical.read_bytes()
        rows, fields = read_table(self.canonical)
        if self.canonical.read_bytes() != before:
            raise ValueError('CSV changed while reading; refresh and try again')
        ids = [r['review_id'] for r in rows]
        if len(set(ids)) != len(ids) or set(ids) != set(self.items):
            raise ValueError('Canonical CSV IDs must exactly match review items')
        for row in rows:
            item = self.items[row['review_id']]
            if any(row[k] != item[k] for k in ('issue_type','content_sha256')):
                raise ValueError('Canonical identity mismatch: '+row['review_id'])
        return before, rows, fields

    def snapshot(self):
        with self.lock:
            return self._snapshot()

    def _snapshot(self):
        data, rows, _ = self._rows()
        revision = digest(data)
        if self.cached and self.cached['revision'] == revision:
            return self.cached
        validation = self.validator(self.project)
        if self.canonical.read_bytes() != data:
            raise ValueError('CSV changed during validation; refresh and try again')
        statuses = {r['review_id']:r for r in validation['items']}
        items = []
        for row in rows:
            status = statuses[row['review_id']]
            item = dict(self.items[row['review_id']])
            item.update({k:row.get(k,'') for k in ('human_decision','selected_annotation','notes')})
            item['decision_status'] = status['decision_status']
            item['errors'] = status['errors']
            items.append(item)
        order = {identity:i for i,identity in enumerate(self.items)}
        items.sort(key=lambda r:order[r['review_id']])
        self.cached = dict(revision=revision,items=items,summary={k:validation[k] for k in ('total_review_items','complete','pending','invalid','conflicts_complete','geometry_complete')},validator_command=VALIDATOR_COMMAND)
        return self.cached

    def save(self, identity, decision, revision, notes=None, selected_annotation=None):
        with self.lock:
            if identity not in self.items:
                raise ValueError('Unknown review_id')
            item = self.items[identity]
            if decision not in ALLOWED[item['issue_type']]:
                raise ValueError('Decision not allowed for this issue type')
            for value in (notes, selected_annotation):
                if value is not None and (not isinstance(value,str) or len(value)>32768):
                    raise ValueError('Invalid text field')
            # Coordinate app writers; revision checking also catches external Excel/editor saves.
            lock_path = self.canonical.with_suffix('.csv.lock')
            try:
                fd = os.open(lock_path, os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            except FileExistsError:
                raise ValueError('Another save is active; retry after it finishes')
            os.close(fd)
            temporary = None
            try:
                before, rows, fields = self._rows()
                if revision != digest(before):
                    raise ValueError('Decisions changed in another window/editor. Reload before replacing a decision.')
                row = next(r for r in rows if r['review_id']==identity)
                row['human_decision'] = decision
                if decision in ('KEEP_A','KEEP_B'):
                    row['selected_annotation'] = item['annotation_'+decision[-1].lower()]
                elif decision == 'MANUAL_FIX':
                    if selected_annotation is not None: row['selected_annotation'] = selected_annotation
                else:
                    row['selected_annotation'] = ''
                if notes is not None: row['notes'] = notes
                # Full status is recalculated from disk immediately after saving.
                row['decision_status'] = 'PENDING'
                if 'decision_status' not in fields: fields.append('decision_status')
                fd, temporary = tempfile.mkstemp(prefix='.review-',suffix='.csv',dir=self.canonical.parent)
                with os.fdopen(fd,'w',encoding='utf-8-sig',newline='') as stream:
                    writer = csv.DictWriter(stream,fieldnames=fields,delimiter=';')
                    writer.writeheader(); writer.writerows(rows)
                    stream.flush(); os.fsync(stream.fileno())
                if self.canonical.read_bytes() != before:
                    raise ValueError('CSV changed during save; no decision was overwritten')
                history = self.canonical.parent/'benchmark_v5_decision_history'
                if history.resolve() != history:
                    raise ValueError('Unsafe decision history directory')
                history.mkdir(exist_ok=True)
                backup = history/(digest(before)+'.csv')
                if not backup.exists():
                    with backup.open('xb') as stream:
                        stream.write(before); stream.flush(); os.fsync(stream.fileno())
                os.replace(temporary,self.canonical)
                temporary = None
                self.cached = None
            finally:
                if temporary is not None: Path(temporary).unlink(missing_ok=True)
                lock_path.unlink(missing_ok=True)
            try:
                state = self._snapshot()
                return dict(saved=True,message=f'Saved {decision} for {identity}',state=state)
            except Exception as exc:
                # Never imply the already-durable save failed if validation fails afterwards.
                return dict(saved=True,message=f'Saved {decision} for {identity}; validation needs attention: {exc}',state=None)


def make_server(store, port=8766):
    token = secrets.token_urlsafe(32)
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type='application/json; charset=utf-8'):
            self.send_response(status)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers(); self.wfile.write(body)
        def local(self):
            expected = '127.0.0.1:'+str(self.server.server_port)
            return self.headers.get('Host') == expected
        def do_GET(self):
            if not self.local(): return self.send(403,b'{}')
            path = urlsplit(self.path).path
            try:
                if path == '/':
                    body = Path(__file__).with_name('review_benchmark_v5.html').read_text(encoding='utf-8').replace('__TOKEN__',token)
                    return self.send(200,body.encode(),'text/html; charset=utf-8')
                if path == '/api/state':
                    return self.send(200,json.dumps(store.snapshot()).encode())
                if path.startswith('/asset/') and path[7:] in store.assets:
                    return self.send(200,store.assets[path[7:]].read_bytes(),'image/jpeg')
                return self.send(404,b'{}')
            except Exception as exc: return self.send(409,json.dumps({'error':str(exc)}).encode())
        def do_POST(self):
            expected = 'http://127.0.0.1:'+str(self.server.server_port)
            if not self.local() or self.headers.get('X-Review-Token') != token or self.headers.get('Origin') not in (None,expected):
                return self.send(403,b'{}')
            if self.path != '/api/decision': return self.send(404,b'{}')
            try:
                length = int(self.headers.get('Content-Length','0'))
                if not 0 < length <= 100000: raise ValueError('Invalid request size')
                value = json.loads(self.rfile.read(length))
                result = store.save(value['review_id'],value['decision'],value['revision'],value.get('notes'),value.get('selected_annotation'))
                return self.send(200,json.dumps(result).encode())
            except (ValueError,KeyError,TypeError,OSError) as exc:
                return self.send(409,json.dumps({'error':str(exc)}).encode())
        def log_message(self,*args): pass
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Local benchmark_v5 review; decisions only, no dataset build')
    parser.add_argument('--port',type=int,default=8766)
    args=parser.parse_args()
    store=ReviewStore()
    server=make_server(store,args.port)
    print(f'Review app: http://127.0.0.1:{server.server_port}',flush=True)
    print('Decisions are saved directly to '+str(store.canonical),flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
