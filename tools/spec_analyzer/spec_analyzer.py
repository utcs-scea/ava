#!/usr/bin/env python3
# CUDA spec analyzer
# Dumps CSV spreadsheet summary to specs.csv
# Dumps combined spec information function content to specs.txt
#

AVA_SPECS_PATH = '../../cava/samples/'
OUTPUT_CSV = 'specs.csv'
OUTPUT_SPECS = 'specs.txt'

import re
import csv
import hashlib
from glob import glob

funcInfo = {}           # Dict of function names to info dict {'preamble' : str, 'args' : str, 'sigs' : {SIG : {'index' : int, 'body' : str, 'specs' : set ([1, 2, ..])}}}
specIndex = 0
cudaSpecNames = []      # Array of spec names (base name of cpp file)

fNames = sorted (glob (AVA_SPECS_PATH + '**/*.cpp', recursive=True), key=lambda x: re.sub ('[^/]+/', '', x).replace ('.cpp', ''))

for srcFile in fNames:
  cSrc = open (srcFile).read ()

  # Skip spec files which aren't CUDA related (look for cuda_runtime(_api).h)
  if 'cuda_runtime' not in cSrc:
    continue

  cudaSpecNames.append (re.sub ('[^/]+/', '', srcFile).replace ('.cpp', ''))

  cSrc = re.sub (r'/\*.*?\*/', '', cSrc, flags=re.DOTALL)     # Remove multi-line comments
  cSrc = re.sub (r'//.*?\n' ,'\n', cSrc)                      # Remove single line comments
  cSrc = re.sub (r' +__dv *\([^)]+?\) *' ,'', cSrc)           # Remove __dv(VAL) from function arguments
  cSrc = re.sub (r'\n#[^\n]+', '', cSrc)                      # Remove preprocessor commands

  # Find functions ([0] is function preamble - content up to and including name, [1] is arguments, [2] is the function body)
  finfo = re.findall (r'\n\n([^\s][^(]+)(\(.*?\))(?:;|\s*{(.*?)\n})', cSrc, flags=re.M | re.DOTALL)

  # Add to function info dictionary
  for preamble, args, body in finfo:
    name = re.split (r'\s', preamble.strip ())[-1]              # Parse out function names from function preamble

    if not name.startswith ('cu'): continue

    # Create or fetch function info by name
    if name not in funcInfo:
      info = {'preamble' : preamble, 'args' : args, 'sigs' : {}}
      funcInfo[name] = info
    else: info = funcInfo[name]

    # Is function implemented?
    if "%s is not implemented" not in body:
      sig = re.sub ('\s', '', body)                             # Use the function body content without whitespace as its "signature"
    else: sig = ''                                              # Use empty signature for unimplemented

    # Add new body signature if unique, or add spec to the list which implements the same function content
    if sig not in info['sigs']:
      info['sigs'][sig] = {'index' : len (info['sigs']), 'body' : body, 'specs' : set ([specIndex])}
    else: info['sigs'][sig]['specs'].update ([specIndex])

  specIndex += 1

# Use specification base names for CSV columns
csvRows = [['Function'] + cudaSpecNames]

counts = [0] * len (cudaSpecNames)

funcNames = sorted (funcInfo.keys ())

# Construct CSV rows
for name in funcNames:
  row = [name]
  sigDict = funcInfo[name]['sigs']

  for i in range (len (cudaSpecNames)):
    for sig, sigInfo in sigDict.items ():
      if i in sigInfo['specs']:
        if sig != '':
          row.append (str (sigInfo['index']))
          counts[i] += 1
        else: row.append ('STUB')
        break
    else: row.append ('')

  csvRows.append (row)

csvRows.append (['Totals:'] + counts)

csv.writer (open (OUTPUT_CSV, 'w')).writerows (csvRows)

txt = ''

# Write out specs function overview document
for name in funcNames:
  info = funcInfo[name]
  txt += '// ' + 80 * '=' + '\n'
  txt += info['preamble'] + info['args'] + '\n'

  notPresent = set (range (len (cudaSpecNames)))

  for sigInfo in info['sigs'].values ():
    notPresent -= sigInfo['specs']

  if notPresent:
    txt += '// UNAVAILABLE: ' + ','.join ([cudaSpecNames[i] for i in sorted (list (notPresent))])

  if '' in info['sigs']:
    txt += '// UNIMPLEMENTED: ' + ','.join ([cudaSpecNames[i] for i in sorted (list (info['sigs']['']['specs']))])

  first = True

  for sig, sigInfo in info['sigs'].items ():
    if sig == '': continue

    if not first:
      first = False
    else: txt += '// ' + 20 * '-' + '\n'

    txt += '// Implemented: ' + ','.join ([cudaSpecNames[i] for i in sorted (list (sigInfo['specs']))])
    txt += sigInfo['body'] + '\n'

open (OUTPUT_SPECS,'w').write (txt)

