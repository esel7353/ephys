#!/usr/bin/python3
################################################################################
#
# Tex Export
#
# Copyright (C) 2013, Frank Sauerburger
#   published under MIT license (see below)
#
################################################################################
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################

def dir(file='pyexp.tex'):
  data = read(file)
  for d in data:
    print('{:16}->  {}'.format(d, data[d]))

def read(file='pyexp.tex'):
  data = {}
  f = open(file, 'r')
  for line in f:
    if line.startswith('%#'): continue
    line = line.split(':', maxsplit=1)
    if len(line) == 2 and len(line[0]) > 0 and line[0][0] == '%':
      key = line[0][1:].strip()
      val = line[1].strip()
      data[key] = val
  f.close()
  return data

def push(id, s, file='pyexp.tex'):
  data = {}

  try:
    data = read(file='pyexp.tex')
  except FileNotFoundError:
    pass

  data[id] = s.replace('\n', ' ')

  f = open(file, 'w')
  print('%# DO NOT EDIT #%', file=f)
  for key in data:
    print('% {}: {}'.format(key, data[key]), file=f)

  print(r'\RequirePackage{ifthen}', file=f)
  print(r'\newcommand{\pyexp}[1]{%', file=f)
  for key in data:
    l = r'\ifthenelse{\equal{#1}{KEY}}{CODE}{}%'.replace('KEY', key)
    l = l.replace('CODE', data[key])
    print(l, file=f)
