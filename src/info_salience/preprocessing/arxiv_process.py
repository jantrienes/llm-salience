"""
Convert LaTeX sources to Markdown.

Code credits: Sebastian Joseph (UT Austin).
"""

import pypandoc
import os
import shutil
import tarfile
import magic
import tempfile
import re
import chardet
import subprocess
import signal
import argparse

from tqdm import tqdm

from directory_tree import DisplayTree

def get_doc_class_tex(dirname):

  fname_add = []
  for fname in os.listdir(dirname):
    full_name = os.path.join(dirname, fname)
    if full_name.endswith('.tex'):
        if convert_to_utf8(full_name, full_name):
          with open(full_name) as f:
            search = f.read()
            if '\\documentclass' in search:
              fname_add.append((full_name, dirname))
    if os.path.isdir(full_name):
        fname_add.extend(get_doc_class_tex(full_name))
  return fname_add

def make_tarfile(output_fname, source):
  with tarfile.open(output_fname, 'w:gz') as tar:
    tar.add(source, arcname=os.path.basename(source))

def convert_to_utf8(inputf, outputf):
  with open(inputf, 'rb') as f:
      raw_data = f.read()
      result = chardet.detect(raw_data)
      encoding = result['encoding']
  try:
    with open(inputf, encoding=encoding) as inf:
      content = inf.read()
  except:
    return False

  with open(outputf, 'w', encoding='utf-8') as outf:
    outf.write(content)

  return True

class TimeoutException(Exception):
  pass

def timeout_handler(signum, frame):
  raise TimeoutException()

def extract_pandoc_save_per(tar_path, outdir):

    #first extract to tempfile
    if tarfile.is_tarfile(tar_path):

        with tempfile.TemporaryDirectory() as temp_dir_name:
          print("starting")
          extract_files(tar_path, temp_dir_name)
          all_docs = get_doc_class_tex(temp_dir_name)
          cwd = os.getcwd()
          for doc, doc_dir in all_docs:
            doc_str = None

            with open(doc, 'r') as d:
              doc_str = d.read()
              doc_str = doc_str.replace('figure*', 'figure')
              doc_str = doc_str.replace('table*', 'table')

              doc_lines = doc_str.split('\n')
              new_doc_lines = []
              for line in doc_lines:
                cleaned_line = re.sub(r'(?<!\\)%.*$', '', line).rstrip()
                if cleaned_line == '' and line != '':
                  continue
                new_doc_lines.append(cleaned_line)

              doc_str = '\n'.join(new_doc_lines)
              #doc_str_fin = re.sub(r'{([^{}]*)}', lambda m: '{' + m.group(1).replace('\n', '') + '}', doc_str)
              #doc_str = doc_str_fin

            with open(doc, 'w') as d:
              if doc_str is not None:
                d.write(doc_str)
            os.chdir(temp_dir_name)
            try:
              #print('reached_here')
              signal.signal(signal.SIGALRM, timeout_handler)
              signal.alarm(15)
              output = pypandoc.convert_file(doc, 'gfm', format='latex', extra_args=['--standalone', '--citeproc', '--trace'])

              #exit()
              #print("not stuck")
            except Exception as e:
              try:
                print(e)
                print('first fail')
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(15)
                output = pypandoc.convert_file(doc, 'gfm', format='latex', extra_args=['--standalone', '--trace'])
              except:
                print(e)
                print('total failure')
                DisplayTree('.')
                os.chdir(cwd)
                fname = os.path.basename(doc)
                with open(f'error_{fname}.tex', 'w') as f:
                  f.write(doc_str)
                continue

            fname = os.path.basename(doc)
            with open(f"{fname}_pandoc.md", "w") as f:
              f.write(output)
            os.chdir(cwd)

          tar_base = os.path.basename(tar_path)
          out_path = os.path.join(outdir, tar_base)

          print("pandoc ending")

          make_tarfile(out_path, temp_dir_name)
          print('tar ending')
    else:
      print(f'{tar_path} is not a tarfile.')


def extract_pandoc_save(dirname, outdir):

    if not os.path.exists(outdir):
        os.mkdir(outdir)
    skipper = 0
    i = 0
    for tar_path in os.listdir(dirname):
      if tar_path not in os.listdir(outdir):
        if i < skipper:
          i += 1
          continue
        full_name = os.path.join(dirname, tar_path)
        extract_pandoc_save_per(full_name, outdir)
        i += 1


def get_ftype(fname):
  if fname.endswith('pandoc.md'):
    return "pandoc_mds"
  mime = magic.Magic(mime=True)
  return mime.from_file(fname)


def extract_files(fname, fpath):
  types = ['r', 'r:gz', 'r:bz2']
  for t in types:
    try:
      tar = tarfile.open(fname, t)
      tar.extractall(path=fpath)
      tar.close()
      break
    except:
      pass


def extract_cat_transfer(dirname, out_dirname):

  if not os.path.exists(out_dirname):
    os.mkdir(out_dirname)
    os.mkdir(os.path.join(out_dirname, 'temp'))
    os.mkdir(os.path.join(out_dirname, 'non_tar'))

  for fname in tqdm(os.listdir(dirname)):
    fname_full = os.path.join(dirname, fname)

    #first extract everything to a temp dir

    temp_dir = os.path.join(out_dirname, 'temp')
    if not os.path.exists(temp_dir):
      os.mkdir(temp_dir)

    if tarfile.is_tarfile(fname_full):
      extract_files(fname_full, temp_dir)
    else:
      shutil.copy(fname_full, os.path.join(out_dirname, 'non_tar'))

    arxiv_id_pattern = r"arXiv-(.*?)(v\d+)"
    id_match = re.findall(arxiv_id_pattern, fname, flags=re.DOTALL)
    arxiv_id = "arXiv-" + "".join([s.replace('-', '.') for s in id_match[0]])

    #arxiv_id = fname.split('.')[0]

    #now category all the extracted files

    def categorize(fname, basedir, arxiv_id, outdir):

      if os.path.isdir(fname):

        for fpath in os.listdir(fname):
          fpath = os.path.join(fname, fpath)
          categorize(fpath, basedir, arxiv_id, outdir)

      else:
        ftype = get_ftype(fname)

        # create file encoding path and arxiv id

        new_fname = os.path.normpath(fname)
        norm_bd_with_sep = os.path.normpath(basedir) + os.path.sep

        if new_fname.startswith(norm_bd_with_sep):
          new_fname = new_fname[len(norm_bd_with_sep):]

        path_list = new_fname.split(os.path.sep)
        mod_path_list = ['PATH_START'] + path_list + ['PATH_END']

        fin_fname = arxiv_id + "_" + '.'.join(mod_path_list) + '.' + os.path.basename(new_fname)

        if len(fin_fname) > 200:


          with open(os.path.join(outdir, 'long_name_list.txt'), 'a') as f:
            f.write(fin_fname + '\n')

          name_index = -1

          with open(os.path.join(outdir, 'long_name_list.txt'), 'r') as f:
            lines = f.readlines()
            name_index = len(lines) - 1

          fin_fname = "LONG_NAME_" + str(name_index) + "_" + os.path.basename(new_fname)

        ftype_dir = ftype.replace(os.path.sep, '_')

        if not os.path.exists(os.path.join(outdir, ftype_dir)):
          os.mkdir(os.path.join(outdir, ftype_dir))

        shutil.move(fname, os.path.join(outdir, ftype_dir, fin_fname))

    categorize(temp_dir, temp_dir, arxiv_id, out_dirname)

    shutil.rmtree(temp_dir)


def main():

  parser = argparse.ArgumentParser()

  parser.add_argument('filename')

  args = parser.parse_args()

  record_list = [args.filename]

  for record_col in record_list:
    extract_pandoc_save(record_col, f'{record_col}.pandoced')
    extract_cat_transfer(f'{record_col}.pandoced', f'{record_col}.categorized')


if __name__ == "__main__":
  main()
