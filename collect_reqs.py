#!/Users/owner/miniconda3/envs/ml/bin/python
# -*- coding: utf-8 -*-

'''
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
'''

import os
import re
import sys
import glob
import subprocess

if __name__ == "__main__":
    # Run this script as: `python collect_reqs [conda | pip]`
    manager = sys.argv[1]
    cwd = os.path.dirname(os.path.abspath(__file__))
    packages = ['scikit-learn', 'scikit-optimize', 'telegram-send'] # special names
    with open(manager + '_requirements.txt', 'w') as f:
        # Go over all files ending in *.py and collect only trimmed lines starting with `import` and `from`
        for fname in glob.iglob(cwd + '/**/*.py', recursive=True):
            # Disregard this file
            if fname == os.path.abspath(__file__):
                continue
            with open(fname, 'r') as g:
                # Read all lines of every files ending in *.py
                for l in g.readlines():
                    tokenized_line = re.split(r"\W", l.strip())
                    if len(tokenized_line) > 0:
                        if tokenized_line[0] in ['import', 'from']:
                            complex_package = tokenized_line[1]
                            # Append the package only
                            packages.append(complex_package.split('.')[0].lower())
        if manager == 'conda':
            packages.extend(['cudatoolkit', 'cudnn'])
            cmd = ['conda', 'list', '--export']
        elif manager == 'pip':
            cmd = ['pip', 'freeze']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        installed_packages = p.stdout.read().decode().lower().split('\n')
        installed_pkg_names = [re.split(r"=+", pkg)[0] for pkg in installed_packages if len(re.split(r"=+", pkg)) > 1]
        output_packages = [pkg for pkg in packages if pkg in installed_pkg_names]
        f.writelines([pkg + '\n' for pkg in installed_packages if re.split(r"=+", pkg)[0] in output_packages])
