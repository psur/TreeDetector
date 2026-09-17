import json
import threading
from urllib.request import Request,urlopen
from urllib.error import HTTPError

import pytest
from src.dataset.final_decisions import read_table,write_table,CANONICAL_FIELDS
from scripts.review_benchmark_v5 import ReviewStore,make_server


@pytest.fixture
def app_data(tmp_path):
    root=tmp_path/'results/benchmark_v5_review';root.mkdir(parents=True)
    config=tmp_path/'config';config.mkdir()
    items=[]
    for identity,kind in [('C-one','ANNOTATION_CONFLICT'),('G-one','GEOMETRY')]:
        item=dict(review_id=identity,issue_type=kind,content_sha256=identity+'-hash',annotation_a='/raw/a.json',annotation_b='/raw/b.json' if identity.startswith('C') else '',review_status='UNRESOLVED',human_decision='',selected_annotation='',notes='original note',self_intersection_severity='MINOR')
        items.append(item)
    write_table(root/'human_review_required.csv',items,list(items[0]))
    write_table(config/'benchmark_v5_decisions.csv',items,CANONICAL_FIELDS)
    (config/'benchmark_v5_review_catalog.json').write_text(json.dumps(items))
    def validator(project):
        rows,_=read_table(project/'config/benchmark_v5_decisions.csv')
        results=[]
        for r in rows:
            status='COMPLETE' if r['human_decision'] else 'PENDING'
            if r['human_decision']=='MANUAL_FIX' and not r['selected_annotation']: status='INVALID'
            results.append(dict(review_id=r['review_id'],decision_status=status,errors=['Manual copy required'] if status=='INVALID' else []))
        return dict(items=results,total_review_items=2,complete=sum(r['decision_status']=='COMPLETE' for r in results),pending=sum(r['decision_status']=='PENDING' for r in results),invalid=sum(r['decision_status']=='INVALID' for r in results),conflicts_complete=int(results[0]['decision_status']=='COMPLETE'),geometry_complete=int(results[1]['decision_status']=='COMPLETE'))
    return tmp_path,validator


def store(app_data): return ReviewStore(*app_data)


def test_load_and_resume_existing(app_data):
    s=store(app_data);before=s.canonical.read_bytes()
    assert len(s.snapshot()['items'])==2
    assert s.canonical.read_bytes()==before
    s.save('C-one','KEEP_B',s.snapshot()['revision'],notes='Reviewed český text')
    resumed=store(app_data).snapshot()
    assert resumed['items'][0]['human_decision']=='KEEP_B'
    assert resumed['items'][0]['notes']=='Reviewed český text'
    assert resumed['summary']['complete']==1
    assert resumed['items'][1]['human_decision']==''


@pytest.mark.parametrize('decision,path',[('KEEP_A','/raw/a.json'),('KEEP_B','/raw/b.json'),('EXCLUDE','')])
def test_conflict_save_and_exact_path(app_data,decision,path):
    s=store(app_data);result=s.save('C-one',decision,s.snapshot()['revision'])
    assert result['saved']
    row=read_table(s.canonical)[0][0]
    assert row['human_decision']==decision and row['selected_annotation']==path
    assert row['notes']=='original note'
    assert read_table(s.canonical)[0][1]['human_decision']==''
    assert list((s.canonical.parent/'benchmark_v5_decision_history').glob('*.csv'))


@pytest.mark.parametrize('decision',['KEEP_AS_IS','AUTO_FIX_APPROVED','MANUAL_FIX','EXCLUDE'])
def test_geometry_decisions_no_repairs(app_data,decision):
    s=store(app_data)
    result=s.save('G-one',decision,s.snapshot()['revision'])
    row=read_table(s.canonical)[0][1]
    assert row['human_decision']==decision
    assert not (s.project/'data_staging').exists()
    if decision=='MANUAL_FIX': assert result['state']['summary']['invalid']==1


def test_replacement_and_stale_browser_rejected(app_data):
    s=store(app_data);revision=s.snapshot()['revision']
    s.save('C-one','KEEP_A',revision)
    with pytest.raises(ValueError,match='another window'): s.save('C-one','KEEP_B',revision)
    s.save('C-one','KEEP_B',s.snapshot()['revision'])
    assert read_table(s.canonical)[0][0]['human_decision']=='KEEP_B'


def test_atomic_replace_failure_preserves_original(app_data,monkeypatch):
    s=store(app_data);before=s.canonical.read_bytes()
    def fail(*args): raise OSError('simulated replace failure')
    monkeypatch.setattr('scripts.review_benchmark_v5.os.replace',fail)
    with pytest.raises(OSError): s.save('C-one','KEEP_A',s.snapshot()['revision'])
    assert s.canonical.read_bytes()==before
    assert not list(s.canonical.parent.glob('.review-*'))
    assert not s.canonical.with_suffix('.csv.lock').exists()


def test_invalid_id_and_wrong_issue_decision(app_data):
    s=store(app_data);before=s.canonical.read_bytes()
    with pytest.raises(ValueError,match='Unknown'): s.save('unknown','EXCLUDE',s.snapshot()['revision'])
    with pytest.raises(ValueError,match='issue type'): s.save('C-one','KEEP_AS_IS',s.snapshot()['revision'])
    assert s.canonical.read_bytes()==before


def test_http_local_binding_csrf_and_save(app_data):
    s=store(app_data);server=make_server(s,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base='http://127.0.0.1:'+str(server.server_port)
    try:
        assert server.server_address[0]=='127.0.0.1'
        page=urlopen(base).read().decode()
        token=page.split("const token='")[1].split("'")[0]
        state=json.loads(urlopen(base+'/api/state').read())
        body=json.dumps(dict(review_id='C-one',decision='KEEP_A',revision=state['revision'])).encode()
        with pytest.raises(HTTPError) as error: urlopen(Request(base+'/api/decision',data=body))
        assert error.value.code==403
        result=json.loads(urlopen(Request(base+'/api/decision',data=body,headers={'X-Review-Token':token,'Content-Type':'application/json'})).read())
        assert result['saved'] and result['state']['summary']['complete']==1
        with pytest.raises(HTTPError): urlopen(base+'/asset/../../config/benchmark_v5_decisions.csv')
    finally:
        server.shutdown();server.server_close();thread.join()
