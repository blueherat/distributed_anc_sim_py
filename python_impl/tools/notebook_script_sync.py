#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple bidirectional sync tool between an .ipynb and a plain script using
"# %%" cell markers. This tool does not require external packages and
uses the JSON notebook format directly.

Usage:
    python notebook_script_sync.py --nb2py --nb <notebook.ipynb> --py <script.py>
    python notebook_script_sync.py --py2nb --py <script.py> --nb <notebook.ipynb>

This script is intentionally conservative: it preserves cell ordering
and writes a minimal notebook structure when converting from .py -> .ipynb.
"""

import json
import re
from pathlib import Path
import argparse

MARKER_RE = re.compile(r'^\s*#\s*%%\s*(?:\[(?P<type>\w+)\])?')


def nb_to_py(nb_path, py_path):
    nb_path = Path(nb_path)
    py_path = Path(py_path)
    with nb_path.open('r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    out_lines = []
    for cell in cells:
        ctype = cell.get('cell_type', 'code')
        source = cell.get('source', [])
        if isinstance(source, str):
            source_lines = source.splitlines(True)
        else:
            source_lines = source

        if ctype == 'markdown':
            out_lines.append('# %% [markdown]')
            for s in source_lines:
                s = s.rstrip('\n')
                if s.strip() == '':
                    out_lines.append('#')
                else:
                    out_lines.append('# ' + s)
            out_lines.append('')
        else:
            out_lines.append('# %%')
            for s in source_lines:
                out_lines.append(s.rstrip('\n'))
            out_lines.append('')

    py_path.parent.mkdir(parents=True, exist_ok=True)
    with py_path.open('w', encoding='utf-8') as f:
        f.write('\n'.join(out_lines))
    print(f'Wrote: {py_path}')


def py_to_nb(py_path, nb_path):
    py_path = Path(py_path)
    nb_path = Path(nb_path)
    text = py_path.read_text(encoding='utf-8')
    lines = text.splitlines()

    cells = []
    idx = 0
    n = len(lines)
    while idx < n:
        m = MARKER_RE.match(lines[idx])
        if m:
            cell_type = m.group('type') or 'code'
            is_md = (cell_type == 'markdown')
            idx += 1
            src = []
            while idx < n and not MARKER_RE.match(lines[idx]):
                l = lines[idx]
                if is_md:
                    if l.startswith('# '):
                        src.append(l[2:] + '\n')
                    elif l.strip() == '#' or l.strip() == '':
                        src.append('\n')
                    elif l.startswith('#'):
                        src.append(l[1:].lstrip() + '\n')
                    else:
                        src.append(l + '\n')
                else:
                    src.append(l + '\n')
                idx += 1
            cells.append({'cell_type': 'markdown' if is_md else 'code', 'source': src})
        else:
            # no marker: consume until marker or EOF as a code cell
            src = []
            while idx < n and not MARKER_RE.match(lines[idx]):
                src.append(lines[idx] + '\n')
                idx += 1
            cells.append({'cell_type': 'code', 'source': src})

    nb = {
        'cells': [],
        'metadata': {
            'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
            'language_info': {'name': 'python'}
        },
        'nbformat': 4,
        'nbformat_minor': 5
    }

    for c in cells:
        nb_cell = {'cell_type': c['cell_type'], 'metadata': {}, 'source': c['source']}
        if c['cell_type'] == 'code':
            nb_cell['outputs'] = []
            nb_cell['execution_count'] = None
        nb['cells'].append(nb_cell)

    nb_path.parent.mkdir(parents=True, exist_ok=True)
    with nb_path.open('w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f'Wrote: {nb_path}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nb2py', action='store_true', help='Convert notebook -> python (use markers)')
    ap.add_argument('--py2nb', action='store_true', help='Convert python -> notebook')
    ap.add_argument('--nb', type=str, default=str(Path('python_impl') / 'notebooks' / 'train_anc.ipynb'))
    ap.add_argument('--py', type=str, default=str(Path('python_impl') / 'notebooks' / 'train_anc.py'))
    args = ap.parse_args()

    if args.nb2py:
        nb_to_py(args.nb, args.py)
    if args.py2nb:
        py_to_nb(args.py, args.nb)
    if not (args.nb2py or args.py2nb):
        ap.print_help()
