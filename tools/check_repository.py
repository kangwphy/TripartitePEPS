#!/usr/bin/env python3
"""Static publication audit. Does not import or run scientific code."""
import ast
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    staged = '--staged' in sys.argv
    args = ['git', 'ls-files', '-z', '--cached']
    if not staged:
        args += ['--others', '--exclude-standard']
    files = sorted(set(subprocess.check_output(args, cwd=ROOT).decode().split('\0')) - {''})
    errors, total_bytes = [], 0
    secrets = [r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
               r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{40,}',
               r'AKIA[0-9A-Z]{16}']
    for name in files:
        path = ROOT / name
        if path.is_symlink():
            errors.append([name, 'external/local symlink must not be published'])
            continue
        if not path.is_file():
            errors.append([name, 'missing file'])
            continue
        size = path.stat().st_size
        total_bytes += size
        if size > 10 * 1024**2:
            errors.append([name, 'file exceeds 10 MiB review threshold'])
        if name.split('/')[0] in {'data', 'trash', 'verification'} or path.suffix in {
                '.jls','.npz','.npy','.pt','.pth','.h5','.hdf5','.ckpt','.bson'}:
            errors.append([name, 'local data/artifact selected for publication'])
        if '/chi6_report_python/' in name or '/site-packages/' in name:
            errors.append([name, 'installed third-party package is not project source'])
        try:
            body = path.read_text()
        except UnicodeDecodeError:
            continue
        if any(re.search(pattern, body) for pattern in secrets):
            errors.append([name, 'possible credential; inspect locally before upload'])
        if path.suffix == '.py':
            try:
                ast.parse(body, filename=name)
            except SyntaxError as exc:
                errors.append([name, f'Python syntax error at line {exc.lineno}'])
        if path.suffix == '.jl':
            refs = re.findall(r'(?m)^\s*(?:isdefined[^\n]*?\|\|\s*)?include\("([^"\n]+)"\)', body)
            refs += ['/'.join(re.findall(r'"([^"\n]+)"', match)) for match in
                     re.findall(r'include\(joinpath\(@__DIR__,\s*((?:"[^"\n]+"\s*,?\s*)+)\)\)', body)]
            for ref in refs:
                target = path.parent / ref
                if not target.is_file():
                    errors.append([name, 'missing Julia include: ' + ref])
                elif staged:
                    try:
                        relative = str(target.resolve().relative_to(ROOT))
                    except ValueError:
                        relative = ''
                    if relative not in files:
                        errors.append([name, 'Julia include is not in publication: ' + ref])
    print(json.dumps(dict(files=len(files), bytes=total_bytes, errors=errors,
                          passed=not errors, scope='static audit, not numerical validation'), indent=2))
    if errors:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
