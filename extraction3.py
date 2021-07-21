import numpy as np
import re

def find(string, keywords):
    STR = string.find(keywords)
    if STR == -1:
        return 0
    else:
        return 1
# code 200: Aggregated
keywords = 'Endedafter'
#auxdelete = '(%)'
auxdelete = ''
s = []
f  = open('raw/k8s5.txt','r')
for lines in f:
    ls = lines.strip('\n').replace(' ','').replace('、','/').replace('?','').split('/')
    for i in ls:
        s.append(i)
f.close()

a200 = []
for i in range(len(s)):
    if find(s[i], keywords) == 1:
        a200.append(s[i].strip(keywords+auxdelete))
print('a200', len(a200))

b200 = []
afterkey = 'ms:300calls.qps='
for i in range(len(a200)):
    b200.append(a200[i].strip(afterkey))
print('b200', b200)
#print(results)
#results = list(map(float, b200))

# For obtaining qps
time = []
for i in range(len(a200)):
    time.append(a200[i][-6:])
print('time', time)
non_decimal = re.compile(r'[^\d.]+')
for i in range(len(time)):
    time[i] = non_decimal.sub('', time[i])
qps = list(map(float, time))
print('qps', qps)


# For obtaining thread OK
thlist = []
thread = 'Startingatmaxqpswith'
for i in range(len(s)):
    if find(s[i], thread) == 1:
        thlist.append((s[i].strip(thread))[0])
thlist = list(map(int, thlist))
print('thlist', thlist)

# For iterf OK
iterf = []
iterkey = 'iterationf'
for i in range(len(s)):
    if find(s[i], iterkey) == 1:
        iterf.append((s[i].strip(iterkey)))
iterf = list(map(float, iterf))
print('iterf',iterf)

# For iterf OK
itere = []
iterkey = 'iteratione'
for i in range(len(s)):
    if find(s[i], iterkey) == 1:
        itere.append((s[i].strip(iterkey)))
itere = list(map(float, itere))
print('itere',itere)


# For iterj OK
iterj = []
iterkey = 'iterationj'
for i in range(len(s)):
    if find(s[i], iterkey) == 1:
        iterj.append((s[i].strip(iterkey)))
iterj = list(map(float, iterj))
print('iterj',iterj)


# For iterk OK
iterk = []
iterkey = 'iterationk'
for i in range(len(s)):
    if find(s[i], iterkey) == 1:
        iterk.append((s[i].strip(iterkey)))
iterk = list(map(float, iterk))
print('iterk', iterk)


#For obtaining Code 200
code200 = []
code200rate = []
codesuccess = 'Code200:'
for i in range(len(s)):
    if find(s[i], codesuccess) == 1:
        code200.append((s[i].strip(codesuccess))[:3])
        #code200.append((s[i].strip(codesuccess)))
        code200rate.append((s[i].strip(codesuccess))[4:10])
print('code200', code200)
for i in range(len(code200)):
    code200[i] = non_decimal.sub('',code200[i])
code200 = list(map(float, code200))
code200 = np.array(code200)
length = len(code200)
print('code200', code200)
code200rate = code200/np.array(iterk[:length])
print('code200rate', code200rate)

#For obtaining Code 503
if len(list(code200))< 1:
    code503 = []
    code503rate = []
    codesuccess = 'Code503'
    for i in range(len(s)):
        if find(s[i], codesuccess) == 1:
            code503.append((s[i].strip(codesuccess))[1:3])
            code503rate.append((s[i].strip(codesuccess))[4:9])
    code503 = list(map(int, code503))
    code503rate = list(map(float, code503rate))
    code503 = np.array(code503)
    code503rate = np.array(code503rate)/100
    print('code503',code503)
    calls = np.divide(code503,code503rate)
    code200rate = 1-code503rate

# For obtaining traffic policy
maxpedre = []
maxpedrekey = 'MaxPendingRequests:'
for i in range(len(s)):
    if find(s[i], maxpedrekey) == 1:
        maxpedre.append((s[i].strip(maxpedrekey)))
for i in range(len(maxpedre)):
    maxpedre[i] = non_decimal.sub('', maxpedre[i])
maxpedre = list(map(float, maxpedre))
maxpedre = (np.array(maxpedre)-10).tolist()
print('maxpredre', maxpedre)


maxrecon = []
maxreconkey = 'maxRequestsPerConnection:'
for i in range(len(s)):
    if find(s[i], maxreconkey) == 1:
        maxrecon.append((s[i].strip(maxreconkey)))
maxrecon = list(map(float, maxrecon))
print('maxrecon', maxrecon)



mcon = []
mconkey = 'maxConnections:'
for i in range(len(s)):
    if find(s[i], mconkey) == 1:
       mcon.append((s[i].strip(mconkey)))
mcon = list(map(float, mcon))
print('mcon', mcon)


basej = []
basejkey = 'baseEjectionTime:'
for i in range(len(s)):
    if find(s[i], basejkey) == 1:
       basej.append((s[i].strip(basejkey)))
basej = list(map(float, basej))


conse = []
consekey = 'consecutive5xxErrors:'
for i in range(len(s)):
    if find(s[i], consekey) == 1:
       conse.append((s[i].strip(consekey)))
conse = list(map(float, conse))


inv = []
invkey = 'interval:'
for i in range(len(s)):
    if find(s[i], invkey) == 1:
        s[i] = s[i].strip(invkey)
        inv.append((s[i].strip('s')))
inv = list(map(float, inv))


maxej = []
maxejkey = 'maxEjectionPercent:'
for i in range(len(s)):
    if find(s[i], maxejkey) == 1:
       maxej.append((s[i].strip(maxejkey)))
maxej = list(map(float, maxej))


code503rate = 1 - np.array(code200rate)
print('code503rate', code503rate)
# k8s2 special case
maxpedre = iterf
maxrecon = itere
data = np.vstack((np.array(maxpedre)[:length], np.array(maxrecon)[:length], np.array(mcon)[:length], np.array(basej)[:length], \
                  np.array(conse)[:length], np.array(inv)[:length], np.array(maxej)[:length], iterj[:length], iterk[:length], \
                  code503rate[:length], np.array(qps)[:length]))
data = data.T
np.savetxt('data/datak5.txt', data)
