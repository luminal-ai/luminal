import collections, json, sys
from pathlib import Path
p=Path(sys.argv[1]); events=json.loads(p.read_text())
children=collections.defaultdict(list)
for e in events:
    children[e['parent']].append(e['id'])
    e['seconds']=e['duration_ns']/1e9
for e in events:
    e['exclusive_seconds']=e['seconds']-sum(events[c]['seconds'] for c in children[e['id']])
    assert e['exclusive_seconds'] > -1e-6, (e['name'],e['exclusive_seconds'])
def descendants(i):
    result=[]
    for c in children[i]:
        result.append(events[c]);result.extend(descendants(c))
    return result
def totals(es):
    result={}
    for e in es:
        r=result.setdefault(e['name'],dict(count=0,inclusive_seconds=0,exclusive_seconds=0,value=0))
        r['count']+=1;r['inclusive_seconds']+=e['seconds'];r['exclusive_seconds']+=e['exclusive_seconds'];r['value']+=e['value'] or 0
    return result
summary={'trace':p.name,'root_seconds':events[0]['seconds'],'all':totals(events),'candidates':[]}
for e in events:
    if e['name'].startswith('candidate.'):
        t=totals(descendants(e['id']))
        summary['candidates'].append(dict(name=e['name'],seconds=e['seconds'],exclusive_seconds=e['exclusive_seconds'],phases=t))
print('Overall root and direct children')
for e in [events[0]]+[events[c] for c in children[0]]:
    print('%-35s %9.3fs excl %9.3fs'%(e['name'],e['seconds'],e['exclusive_seconds']))
print('All events by exclusive wall time (nested inclusive values must not be added)')
for n,t in sorted(summary['all'].items(),key=lambda p:p[1]['exclusive_seconds'],reverse=True)[:45]:
    print('%-35s %7d inclusive %9.3fs exclusive %9.3fs value %s'%(n,t['count'],t['inclusive_seconds'],t['exclusive_seconds'],t['value']))
print('Candidates')
for e in summary['candidates']:
    print(e['name'],'total',round(e['seconds'],3))
    for n,t in e['phases'].items():
        if t['inclusive_seconds']>.05 or t['value']:
            print('  %-33s %6d %8.3fs excl %8.3fs value %s'%(n,t['count'],t['inclusive_seconds'],t['exclusive_seconds'],t['value']))
p.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
