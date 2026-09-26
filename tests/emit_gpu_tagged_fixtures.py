#!/usr/bin/env python3
"""Decode only source bindings of two reference fixtures for HOST math tests.
Not a replacement for the deployment loader or its hash/certificate checks.
"""
from pathlib import Path
import json
import math
import sys


def inverse(matrix):
    n = len(matrix)
    a = [list(row) + [complex(i == j) for j in range(n)] for i, row in enumerate(matrix)]
    for j in range(n):
        pivot = max(range(j, n), key=lambda i: abs(a[i][j]))
        if abs(a[pivot][j]) < 1e-15:
            raise ValueError("singular serialized real form")
        a[j], a[pivot] = a[pivot], a[j]
        divisor = a[j][j]
        a[j] = [v/divisor for v in a[j]]
        for i in range(n):
            if i != j:
                scale = a[i][j]
                a[i] = [u-scale*v for u,v in zip(a[i], a[j])]
    return [row[n:] for row in a]


def number(x):
    value = float(x)
    if not math.isfinite(value):
        raise ValueError("non-finite fixture coefficient")
    return format(value, '.17g')


def main():
    root, output = map(Path, sys.argv[1:])
    lines = ['// Generated from supplied source-binding fixtures. No readout/loader validation.',
             'std::vector<std::pair<std::string, TaggedCauchyModel>> gpu_tagged_fixtures() {',
             'std::vector<std::pair<std::string, TaggedCauchyModel>> result;']
    for name in ('tagged_cauchy_multispecies_model.json', 'tagged_cauchy_physical_image_v3.json'):
        data = json.loads((root/'tests/fixtures'/name).read_text())
        v3 = 'source_binding' in data
        source = data['source_binding']['payload'] if v3 else data
        forms = source['real_forms'] if v3 else data['compiled_artifact']['payload']['real_forms']
        channels = source['channels'] if v3 else data['channel_real_forms']
        species = source['species_order']
        cutoff = source['cutoff']
        radial = {} if v3 else source['radial_definition']['parameters']
        lines += ['{ TaggedCauchyModel m;', 'm.species_order = {'+','.join(json.dumps(s) for s in species)+'};',
                  'm.cutoff = '+number(cutoff)+';',
                  f'm.deployment_kind = TaggedCauchyDeploymentKind::{"PhysicalImageV3" if v3 else "LegacyMomentV2"};',
                  'm.radial_count = '+str(radial.get('radial_count',0))+';',
                  'm.radial_cutoff_width = '+number(radial.get('cutoff_width',0))+';',
                  'm.radial_lambda = '+number(radial.get('lmbda',0))+';']
        form_ids = []
        for form in forms:
            form_ids.append(form['real_form_id'])
            l = form['angular_l']; n = 2*l+1
            if form['magnetic_order'] != list(range(-l,l+1)):
                raise ValueError('fixture magnetic order differs from existing GPU contract')
            matrix = [[complex(*c['binary64']) for c in row] for row in form['real_to_complex_matrix']]
            inv = [[matrix[j][i].conjugate() for j in range(n)] for i in range(n)] if v3 else inverse(matrix)
            values = ','.join('Complex('+number(z.real)+','+number(z.imag)+')' for row in inv for z in row)
            lines += ['{ TaggedCauchyRealForm f;', f'f.l={l}; f.width={n};', 'f.inverse={'+values+'};',
                      'm.real_forms.push_back(f); }']
        offset = 0
        for c in sorted(channels,key=lambda c:c['channel_index']):
            lines += ['{ TaggedCauchyChannel c;',f'c.channel_index={c["channel_index"]}; c.l={c["l"]};',
                      f'c.radial_channel={c["q"] if v3 else c["radial_channel"]};',
                      f'c.neighbor_species_index={species.index(c["neighbor_species"])};',
                      f'c.real_form_index={form_ids.index(c["real_form_id"])}; c.component_offset={offset};',
                      'c.normalization='+number(c.get('binary64_normalization',1))+';',
                      'c.angular_scale='+number(c.get('angular_racah_scale',1))+';', 'm.channels.push_back(c); }']
            offset += 2*c['l']+1
        lines += [f'm.total_component_count={offset};', 'result.emplace_back('+json.dumps(name)+',std::move(m)); }']
    lines += ['return result; }']
    output.write_text('\n'.join(lines)+'\n')

if __name__ == '__main__':
    main()
