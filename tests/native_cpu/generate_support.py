#!/usr/bin/env python3
"""Compile the production native model helpers without the YAML deployment TU.
No runtime or compiler algorithm is mocked. YAML/sidecar schema validation is not
part of this test target. Fixture provenance is verified before building.
"""
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);here=root/'tests/native_cpu'
for record in json.loads((here/'provenance.json').read_text()):
 checks=[(here/record['fixture'],record['fixture_sha256'])]
 if record.get('source_shipped',True):checks.insert(0,(root/record['source'],record['source_sha256']))
 for path,expected in checks:
  if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError(f'stale native fixture: {path}')
text=(root/'src/ye3t_yace_model.cpp').read_text()
pure=text[text.index('void build_splines('):text.index('int full_channel_index(')]
accessors=text[text.index('const YACESpecies &YACEModel::species('):text.index('double YACEModel::memory_usage()')]
pure+=text[text.index('YACESparsePolynomial without_descriptors('):text.index('std::int64_t add_operations(')]
header='''#include "ye3t_yace_model.h"
#include "ye3t_runtime_core.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>
namespace YE3T_LAMMPS { namespace { using PowerKey=std::pair<std::int64_t,std::int64_t>;
'''
tail=(here/'factory_tail.inc').read_text();start=tail.index('struct YACEModelTestAccess {')
out.write_text(header+pure+'}\n'+accessors+tail[start:])
