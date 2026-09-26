#!/usr/bin/env python3
"""Emit DIRECT evaluator fixtures from supplied YACE, not a replacement loader.
Build splines/DAG with verbatim production C++ helpers; only YAML decoding and
field construction are performed here. Deployment/schema validation is NOT tested.
"""
import argparse, pathlib, yaml, collections, hashlib
class Loader(yaml.SafeLoader):
 def construct_mapping(self,node,deep=False):
  d={}
  for k,v in node.value:
   key=self.construct_object(k,deep=True)
   if isinstance(key,list): key=tuple(key)
   if key in d: raise ValueError('duplicate mapping key')
   d[key]=self.construct_object(v,deep=deep)
  return d

def emit(path,out):
 x=yaml.load(path.read_text(),Loader=Loader); lines=['ye3t_native_cpu_fixture_v1',str(len(x['elements'])),repr(x['deltaSplineBins'])]
 def vec(v):lines.append(str(len(v))+' '+' '.join(map(str,v)))
 for s,e in enumerate(x['elements']):
  emb=x['embeddings'][s]; assert emb['FS_parameters'][1]==1
  # PyYAML parses exponent forms without a mantissa dot (1e+20) as strings.
  lines.append(f'{e} {x["E0"][s]} {emb["FS_parameters"][0]} {float(emb["rho_core_cutoff"])-float(emb["drho_core_cutoff"])}')
  channels={}; monomials={}; coeff=[]; desc_off=[0]; desc_terms=[]; desc_coeff=[]; rank=0
  for f in x['functions'][s]:
   rank=max(rank,f['rank'])
   for row,c in enumerate(f['ctildes']):
    keys=[]
    for pos in range(f['rank']):
     ch=(int(f['rank']!=1),f['mus'][pos],f['ns'][pos]-1,f['ls'][pos],f['ms_combs'][row*f['rank']+pos])
     keys.append(channels.setdefault(ch,len(channels)))
    mono=tuple(sorted(collections.Counter(keys).items())); term=monomials.setdefault(mono,len(monomials))
    if term==len(coeff):coeff.append(0.)
    coeff[term]+=c; desc_terms.append(term);desc_coeff.append(c)
   desc_off.append(len(desc_terms))
  lines.append(str(len(channels)))
  for ch in channels:lines.append(' '.join(map(str,ch)))
  lines.append(str(rank)); off=[0];idx=[]; exp=[]
  for mono in monomials:
   for i,e in mono:idx.append(i);exp.append(e)
   off.append(len(idx))
  for v in (off,idx,exp,coeff,desc_off,desc_terms,desc_coeff):vec(v)
 for i in range(len(x['elements'])):
  for j in range(len(x['elements'])):
   b=x['bonds'][(i,j)];assert b['radbasename']=='ChebExpCos'
   lines.append(' '.join(map(str,[i,j,b['nradmax'],b['lmax'],b['nradbasemax'],b['radparameters'][0],b['rcut'],b['dcut']])))
   vec([c for rad in b['radcoefficients'] for ang in rad for c in ang])
 out.write_text('\n'.join(lines)+'\n');return hashlib.sha256(path.read_bytes()).hexdigest()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('model',type=pathlib.Path);p.add_argument('output',type=pathlib.Path);a=p.parse_args();print(emit(a.model,a.output))
