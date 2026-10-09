"""Sequence-boundary changes and graph rebuilds must refresh reusable inputs."""
import argparse
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perf'))
from compare_llamacpp import Endpoint, ManagedServer

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',required=True,type=Path)
p.add_argument('--model',required=True,type=Path)
p.add_argument('--backend',default='cpu')
a=p.parse_args()

def text(index,word,tokens):
    return 'q'+str(index)+(' '+word)*(tokens-4)

cases=[
    ('initial',[text(0,'x',682),text(1,'y',683),text(2,'z',684)]),
    ('changed boundaries',[text(3,'w',684),text(4,'a',682),text(5,'b',683)]),
    ('changed inputs',[text(6,'c',684),text(7,'d',682),text(8,'e',683)]),
    ('rebuild',[text(0,'x',263)]),
    ('return after rebuild',[text(3,'w',684),text(4,'a',682),text(5,'b',683)]),
    ('short',[text(0,'x',32)]),
    ('short two',[text(1,'y',32),text(2,'z',32)]),
    ('short three',[text(3,'a',32),text(4,'b',32),text(5,'c',32)]),
    ('return after eviction',[text(6,'d',32)]),
]
with socket.socket() as probe:
    probe.bind(('127.0.0.1',0))
    port=probe.getsockname()[1]
endpoint=Endpoint('127.0.0.1',port,'/api/embed','embeddinggemma')
command=[str(a.binary.resolve()),'--backend',a.backend,'--model',str(a.model.resolve()),
    '--bind','127.0.0.1','--port',str(port),'--cache-entries','0','--response-cache-mb','0']

def request(inputs):
    with urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{port}/api/embed',
        json.dumps({'input':inputs}).encode(),headers={'Content-Type':'application/json'}),timeout=180) as response:
        result=json.load(response)
    assert len(result['embeddings'])==len(inputs)
    for vector in result['embeddings']:
        assert len(vector)==768 and all(math.isfinite(v) for v in vector)
        assert abs(math.fsum(v*v for v in vector)-1)<1e-5
    return result

os.environ['EI_PROFILE_BACKBONE2']='1'
with tempfile.TemporaryDirectory(prefix='reuse-inputs2-') as directory:
    work=Path(directory)
    os.environ['EI_REUSE_INPUTS2']='0'
    with ManagedServer(command,endpoint,'/healthz',work/'off.log'):
        expected=[request(inputs) for _,inputs in cases]
    os.environ['EI_REUSE_INPUTS2']='1'
    with ManagedServer(command,endpoint,'/healthz',work/'on.log'):
        for (label,inputs),before in zip(cases,expected):
            actual=request(inputs)
            assert actual==before,f'reusable inputs changed output: {label}'
            print(label,'bit-identical',flush=True)
    log=(work/'on.log').read_text(errors='replace')
    assert 'Backbone static input reuse:' in log,'input reuse was not enabled'
    assert 'tokens=2049 batch=3' in log,'large batch was not exercised as one forward'
    assert 'layout_reused=1' in log,'sequence layout was never reused'
print('Reusable-input boundary and graph-lifetime isolation passed')
