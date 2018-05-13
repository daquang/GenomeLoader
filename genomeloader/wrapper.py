import numpy as np
import pandas as pd
from pyfaidx import Fasta
import py2bit
import pyBigWig
from tqdm import tqdm
import sklearn.utils


class TwoBitWrapper:
    def __init__(self, twobit_file, alpha='dna', one_hot=True, channel_last=True, in_mem=False, thread_safe=False):
        self.twobit = py2bit.open(twobit_file)
        self.one_hot = one_hot
        self._chroms = list(self.twobit.chroms().keys())
        self._chroms_size = self.twobit.chroms()
        self.channel_last = channel_last
        self.thread_safe = thread_safe
        if alpha == 'dna':
            self.residues = np.array(['A', 'C', 'G', 'T'])
        elif alpha == 'rna':
            self.residues = np.array(['A', 'C', 'G', 'U'])
        elif alpha == 'protein':
            self.residues = np.array(['A', 'C', 'D', 'E',
                                      'F', 'G', 'H', 'I',
                                      'K', 'L', 'M', 'N',
                                      'P', 'Q', 'R', 'S',
                                      'T', 'V', 'W', 'Y'])
        self.in_mem = in_mem
        if in_mem:
            twobit_onehot_dict = self._encode_seqs(self.twobit)
            self.twobit.close()
            self.twobit = twobit_onehot_dict
            self.thread_safe = True
        else:
            if thread_safe:
                self.twobit.close()
                self.twobit = twobit_file

    def __del__(self):
        self.close()

    def close(self):
        if not self.thread_safe:
            self.twobit.close()

    def _encode_seqs(self, twobit):
        # Converts a TwoBit object into a dictionary of one-hot coded boolean matrices
        twobit_dict = {}
        pbar = tqdm(twobit.chroms())
        for chrom in pbar:
            pbar.set_description(desc='Loading sequence: ' + chrom)
            seq = twobit.sequence(chrom)
            seq = np.array(list(seq))
            if self.one_hot:
                seq = self.residues == seq[:, np.newaxis]
                if not self.channel_last:
                    seq = seq.T
            twobit_dict[chrom] = seq
        return twobit_dict

    def __getitem__(self, item):
        if len(item) == 2:
            chrom, coords = item
        else:
            chrom = item
            coords = slice(0, self._chroms_size[chrom], None)
        if self.in_mem:
            seq = self.twobit[chrom][coords]
        else:
            if self.thread_safe:
                twobit = py2bit.open(self.twobit)
                seq = np.array(list(twobit.sequence(chrom, coords.start, coords.stop)))
                twobit.close()
            else:
                seq = np.array(list(self.twobit.sequence(chrom, coords.start, coords.stop)))
            if self.one_hot:
                seq = self.residues == seq[:, np.newaxis]
                if not self.channel_last:
                    seq = seq.T
        return seq

    def chroms(self):
        return self._chroms

    def chroms_size(self):
        return self._chroms_size


class FastaWrapper:
    def __init__(self, fasta_file, alpha='dna', read_ahead=10000, one_hot=True, in_mem=False, thread_safe=False):
        self.fasta = Fasta(fasta_file, as_raw=True, sequence_always_upper=True, read_ahead=read_ahead)
        self.one_hot = one_hot
        self._chroms = list(self.fasta.keys())
        seq_lens = [len(self.fasta[chrom]) for chrom in self._chroms]
        self._chroms_size = dict(zip(self._chroms, seq_lens))
        self.read_ahead = read_ahead
        self.thread_safe = thread_safe
        if alpha == 'dna':
            self.residues = np.array(['A', 'C', 'G', 'T'])
        elif alpha == 'rna':
            self.residues = np.array(['A', 'C', 'G', 'U'])
        elif alpha == 'protein':
            self.residues = np.array(['A', 'C', 'D', 'E',
                                      'F', 'G', 'H', 'I',
                                      'K', 'L', 'M', 'N',
                                      'P', 'Q', 'R', 'S',
                                      'T', 'V', 'W', 'Y'])
        self.in_mem = in_mem
        if in_mem:
            fasta_onehot_dict = self._encode_seqs(self.fasta)
            self.fasta.close()
            self.fasta = fasta_onehot_dict
            self.thread_safe = True
        else:
            if thread_safe:
                self.fasta.close()
                self.fasta = fasta_file

    def __del__(self):
        self.close()

    def close(self):
        if not self.thread_safe:
            self.fasta.close()

    def _encode_seqs(self, fasta):
        # Converts a FASTA object into a dictionary of one-hot coded boolean matrices
        fasta_dict = {}
        pbar = tqdm(fasta)
        for record in pbar:
            pbar.set_description(desc='Loading sequence: ' + record.name)
            seq = record[:]
            seq = np.array(list(seq))
            if self.one_hot:
                seq = self.residues == seq[:, np.newaxis]
            fasta_dict[record.name] = seq
        return fasta_dict

    def __getitem__(self, item):
        if len(item) == 2:
            chrom, coords = item
        else:
            chrom = item
            coords = slice(0, self._chroms_size[chrom], None)
        if self.in_mem:
            seq = self.fasta[chrom][coords]
        else:
            if self.thread_safe:
                fasta = Fasta(self.fasta, as_raw=True, sequence_always_upper=True, read_ahead=self.read_ahead)
                seq = np.array(list(fasta[chrom][coords]))
                fasta.close()
            else:
                seq = np.array(list(self.fasta[chrom][coords]))
            if self.one_hot:
                seq = self.residues == seq[:, np.newaxis]
        return seq

    def chroms(self):
        return self._chroms

    def chroms_size(self):
        return self._chroms_size


class BigWigWrapper:
    def __init__(self, bigwig_file, default_value=0, in_mem=False, thread_safe=False):
        self.bigwig = pyBigWig.open(bigwig_file, 'r')
        self._chroms_size = self.bigwig.chroms()
        self._chroms = list(self._chroms_size.chroms())
        self.default_value = default_value
        self.thread_safe = thread_safe
        self.in_mem = in_mem
        if in_mem:
            seqs_dict = self._encode(self.bigwig)
            self.bigwig.close()
            self.bigwig = seqs_dict
            self.thread_safe = True
        else:
            if thread_safe:
                self.bigwig.close()
                self.bigwig = bigwig_file

    def __del__(self):
        self.close()

    def close(self):
        if not self.thread_safe:
            self.bigwig.close()

    def _encode_seqs(self, bigwig):
        # Converts a bigWig object into a dictionary of numpy arrays
        seqs_dict = {}
        pbar = tqdm(self._chroms)
        for chrom in pbar:
            pbar.set_description(desc='Loading sequence: ' + chrom)
            seqs_dict[chrom] = self.bigwig.values(chrom, 0, self._chroms_size[chrom], numpy=True)
        return seqs_dict

    def __getitem__(self, item):
        if len(item) == 2:
            chrom, coords = item
        else:
            chrom = item
            coords = slice(0, self._chroms_size[chrom], None)
        if self.in_mem:
            seq = self.bigwig[chrom][coords]
        else:
            if self.thread_safe:
                bigwig = pyBigWig.open(self.bigwig, 'r')
                seq = bigwig.values(chrom, coords.start, coords.stop, numpy=True)
                bigwig.close()
            else:
                seq = self.bigwig.values(chrom, coords.start, coords.stop, numpy=True)
        seq[np.isnan(seq)] = self.default_value
        return seq

    def chroms(self):
        return self._chroms


class BedGraphWrapper:
    def __init__(self, bedgraph_file):
        self.df = pd.read_table(bedgraph_file,
                           names=['chrom', 'chromStart', 'chromEnd', 'dataValue'])

    def __getitem__(self, item):
        return self.df.iloc[item]

    def __len__(self):
        return len(self.df)

    def shuffle(self):
        self.df = sklearn.utils.shuffle(self.df)


class NarrowPeakWrapper:
    def __init__(self, narrowpeak_file):
        self.df = pd.read_table(narrowpeak_file,
                           names=['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'signalValue', 'pValue', 'qValue', 'peak'])

    def __getitem__(self, item):
        return self.df.iloc[item]

    def __len__(self):
        return len(self.df)

    def shuffle(self):
        self.df = sklearn.utils.shuffle(self.df)

