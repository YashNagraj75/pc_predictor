import re

t = open('mpc_meta.xml').read()
for tag in ['title', 'summary', 'published', 'updated']:
    m = re.findall('<' + tag + '>(.*?)</' + tag + '>', t, re.S)
    if m:
        print(tag.upper() + ':', re.sub(r'\s+', ' ', m[-1]).strip()[:3000])
        print()
print('AUTHORS:', re.findall('<name>(.*?)</name>', t))
print('CATS:', re.findall(r'term="([^"]+)"', t))
