import json

nb = json.load(open('mu_pc.ipynb'))
print('TOTAL CELLS', len(nb['cells']))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    ec = c.get('execution_count')
    outs = c.get('outputs', [])
    kinds = [o.get('output_type') for o in outs]
    print('--- cell %d exec=%s outputs=%s' % (i, ec, kinds))
    for o in outs:
        if o.get('output_type') == 'stream':
            s = ''.join(o['text'])
            print('STREAM:\n' + s[:2500])
        elif o.get('output_type') == 'error':
            print('ERROR: %s: %s' % (o['ename'], str(o['evalue'])[:300]))
