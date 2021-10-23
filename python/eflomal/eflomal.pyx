cimport cython
from cpython cimport bool
cimport numpy as np
from libc.stdio cimport fprintf, fdopen, fputc, fflush, FILE

import os
import sys
import math
import subprocess
from tempfile import NamedTemporaryFile

import numpy as np
from pyvi import ViTokenizer
from underthesea import word_tokenize


def ignore_spec_chars(str text):
    '''Loại bỏ các ký tự không cần thiết
    '''
    skip_chars = ['-', ':', ',', '.', '+', ';', '<', '>', "'", '*', '!',
                  '(', ')', '"', '/', '‰', '%', '…', '‘', '–', '?', '@', '°']
    return ''.join([c for c in text if c not in skip_chars and not c.isdigit()])


def tnkey_to_unicode(str text):
    '''Chuyển một vài ký tự đặc biệt trong font TNKey sang Unicode'''
    if not text:
        return ''
    list_chars = [('[', 'ƀ'), ('_', 'Ƀ'), ('}', 'Č'), (']', 'č'),
                  ('E|', 'Ĕ'), ('e\\', 'ĕ'), ('I|', 'Ĭ'), ('i\\', 'ĭ'),
                  ('~', 'Ñ'), ('`', 'ñ'), ('O|', 'Ŏ'), ('o\\', 'ŏ'),
                  ('U|', 'Ŭ'), ('u\\', 'ŭ'), ('^', 'ĭ'), ('A|', 'Ă'),
                  ('a\\', 'ă'), ('  ', ' '), ('‘', '\'')]
    for c in list_chars:
        text = text.replace(c[0], c[1])
    return text.strip()


def replace_or_recover_spec_kho_chars(str text, bool recover = False):
    '''Thay thế một số ký tự đặc thù của K'Ho và khôi phục lại trạng thái'''
    if not text:
        return ''
    spec_chars = [('a#', 'ȁ'), ('e#', 'ȅ'), ('o#', 'ȍ'),
                  ('A$', 'Ȁ'), ('E$', 'Ȅ'), ('O$', 'Ȍ'),
                  ('ơ\\', 'ō'), ('ư\\', 'ū'), ('Ơ|', 'Ō'), ('Ư|', 'Ū')]
    if recover:
        fid = 1
        rid = 0
    else:
        fid = 0
        rid = 1
    for c in spec_chars:
        text = text.replace(c[fid], c[rid])
    return text


def tokenize(str sentence, bool lower = True, bool tnkey_to_unicode = False, int tokenizer_id = 1):
    '''Tách (các) câu thành các từ/cụm từ dựa trên khoảng trắng (0), pyvi (1) hoặc underthesea (2)'''
    if lower:
        sentence = sentence.lower()
    if tnkey_to_unicode:
        sentence = tnkey_to_unicode(sentence)

    # Bỏ qua các ký tự không cần thiết
    sentence = ignore_spec_chars(sentence).strip()
    sentence = replace_or_recover_spec_kho_chars(sentence)
    if tokenizer_id == 0:
        tokens = sentence.split()
    elif tokenizer_id == 1:
        tokens = [i.replace('_', ' ')
                for i in ViTokenizer.tokenize(sentence).split()]
    elif tokenizer_id == 2:
        tokens = word_tokenize(sentence)
    
    tokens = [replace_or_recover_spec_kho_chars(token, True) for token in tokens]
    return tokens


cpdef tuple read_text(pyfile, bool lowercase, int prefix_len, int suffix_len, int tokenizer = 0):
    """Read a tokenized text file as a list of indexed sentences.
    
    Optionally transform the vocabulary according to the parameters.
    
    pyfile -- file to read
    lowercase -- if True, all tokens are lowercased
    prefix_len -- if non-zero, all tokens are cut of after so many characters
    suffix_len -- if non-zero, as above, but cutting from the right side
    tokenizer -- 0 if using default tokenizer (WhiteSpace tokenizer), 1 if using PyVi or 2 if using Underthesea

    Returns:
    a tuple (list sents, dict index) containing the actual sentences and the
    string-to-index mapping used.
    """
    cdef:
        np.ndarray[np.uint32_t, ndim=1] sent
        list sents, tokens
        str line, token
        dict index
        int i, n, idx

    index = {}
    sents = []
    for line in pyfile:
        tokens = tokenize(line, lowercase, False, tokenizer)
        n = len(tokens)
        sent = np.empty(n, dtype=np.uint32)

        for i in range(n):
            token = tokens[i]
            if prefix_len != 0: token = token[:prefix_len]
            elif suffix_len != 0: token = token[-suffix_len:]
            idx = index.get(token, -1)
            if idx == -1:
                idx = len(index)
                index[token] = idx
            sent[i] = idx

        sents.append(sent)

    return (sents, index)


cpdef write_text(pyfile, tuple sents, int voc_size):
    """Write a sequence of sentences in the format expected by eflomal

    Arguments:
    pyfile -- Python file object to write to
    sents -- tuple of sentences, each encoded as np.ndarray(uint32)
    voc_size -- size of vocabulary
    """
    cdef int token, i, n
    cdef FILE *f
    cdef np.ndarray[np.uint32_t, ndim=1] sent

    f = fdopen(pyfile.fileno(), 'wb')
    fprintf(f, '%d %d\n', len(sents), voc_size)
    for sent in sents:
        n = len(sent)
        if n < 0x400:
            i = 0
            fprintf(f, '%d', n)
            while i < n:
                fprintf(f, ' %d', sent[i])
                i += 1
            fputc(10, f)
        else:
            fputc(48, f)
            fputc(10, f)
    fflush(f)


def align(
        str source_filename,
        str target_filename,
        str links_filename_fwd=None,
        str links_filename_rev=None,
        str statistics_filename=None,
        str scores_filename_fwd=None,
        str scores_filename_rev=None,
        str priors_filename=None,
        int model=3,
        int score_model=0,
        tuple n_iterations=None,
        int n_samplers=1,
        bool quiet=True,
        double rel_iterations=1.0,
        double null_prior=0.2,
        bool use_gdb=False):
    """Call the eflomal binary to perform word alignment

    Arguments:
    source_filename -- str with source text filename, this and the target
                       text should both be written using write_text()
    target_filename -- str with target text filename
    links_filename_fwd -- if given, write links here (forward direction)
    links_filename_rev -- if given, write links here (reverse direction)
    statistics_filename -- if given, write alignment statistics here
    scores_filename -- if given, write sentence alignment scoeres here
    priors_filename -- if given, read Dirichlet priors from here
    model -- alignment model (1 = IBM1, 2 = HMM, 3 = HMM+fertility)
    n_iterations -- 3-tuple with number of iterations per model, if this is
                    not given the numbers will be computed automatically based
                    on rel_iterations
    n_samplers -- number of independent samplers to run
    quiet -- if True, suppress output
    rel_iterations -- number of iterations relative to the default
    """

    with open(source_filename, 'rb') as f:
        n_sentences = int(f.readline().split()[0])

    if n_iterations is None:
        iters = max(2, int(round(
            rel_iterations*5000 / math.sqrt(n_sentences))))
        iters4 = max(1, iters//4)
        if model == 1:
            n_iterations = (iters, 0, 0)
        elif model == 2:
            n_iterations = (max(2, iters4), iters, 0)
        else:
            n_iterations = (max(2, iters4), iters4, iters)

    possible_paths = [os.path.dirname(os.path.realpath(sys.argv[0]))] + \
                     os.environ['PATH'].split(':')
    executable = None
    for path in possible_paths:
        full_path = os.path.join(path, 'eflomal')
        if os.path.exists(full_path):
            executable = full_path
            break
    if executable is None:
        sys.stderr.write('ERROR: eflomal binary not found in either of: ' +
                         ' '.join(possible_paths) + '\n')
        sys.exit(1)
    args = [executable,
            '-m', str(model),
            '-s', source_filename,
            '-t', target_filename,
            '-n', str(n_samplers),
            '-N', str(null_prior),
            '-1', str(n_iterations[0])]
    if quiet: args.append('-q')
    if model >= 2: args.extend(['-2', str(n_iterations[1])])
    if model >= 3: args.extend(['-3', str(n_iterations[2])])
    if links_filename_fwd: args.extend(['-f', links_filename_fwd])
    if links_filename_rev: args.extend(['-r', links_filename_rev])
    if statistics_filename: args.extend(['-S', statistics_filename])
    if score_model > 0: args.extend(['-M', str(score_model)])
    if scores_filename_fwd: args.extend(['-F', scores_filename_fwd])
    if scores_filename_rev: args.extend(['-R', scores_filename_rev])
    if priors_filename: args.extend(['-p', priors_filename])
    if not quiet: sys.stderr.write(' '.join(args) + '\n')
    if use_gdb: args = ['gdb', '-ex=run', '--args'] + args
    subprocess.call(args)

