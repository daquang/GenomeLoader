import numpy as np
import pandas as pd
import pybedtools as pbt
from collections import OrderedDict
from pyfaidx import Fasta
import py2bit
import pyBigWig
from tqdm import tqdm
import sklearn.utils


class SignalWrapper:
    def __init__(self, in_mem=False, thread_safe=False):
        self.in_mem = in_mem
        self.thread_safe = thread_safe

    def __del__(self):
        self.close()

    def __getitem__(self, item):
        return

    def close(self):
        return

    def chroms(self):
        return self._chroms

    def chroms_size(self):
        return self._chroms_size

    def chroms_size_pybedtools(self):
        chromsizes = OrderedDict(self._chroms_size)
        for chrom in chromsizes:
            chromsizes[chrom] = (0, chromsizes[chrom])
        return chromsizes

    def __getitem__(self, item):
        if len(item) == 2:
            chrom, coords = item
        else:
            chrom = item
            coords = slice(0, self._chroms_size[chrom], None)
        # In practice, never have to worry if coords.start > chrom_size or coords.stop < 0
        chrom_size = self._chroms_size[chrom]
        start = int(max(0, coords.start))
        stop = int(min(coords.stop, chrom_size))
        seq = self._get_seq(chrom, start, stop)
        seq_len = stop - start
        orig_len = coords.stop - coords.start
        if seq_len < orig_len:
            left_seq = np.array((start - coords.start) * [self.default_value], dtype=seq.dtype)
            right_seq = np.array((coords.stop - stop) * [self.default_value], dtype=seq.dtype)
            seq = np.concatenate([left_seq, seq, right_seq])
        return seq

    def _get_seq(self, chrom, start, stop):
        return None


class GenomeWrapper(SignalWrapper):
    def __init__(self, alpha='dna', one_hot=True, channel_last=True, in_mem=False, thread_safe=False):
        super().__init__(in_mem, thread_safe)
        alphabets = {
            'dna': np.array(['A', 'C', 'G', 'T']),
            'rna': np.array(['A', 'C', 'G', 'U']),
            'protein': np.array(['A', 'C', 'D', 'E',
                                 'F', 'G', 'H', 'I',
                                 'K', 'L', 'M', 'N',
                                 'P', 'Q', 'R', 'S',
                                 'T', 'V', 'W', 'Y'])
        }
        ambiguous_letters = {
            'dna': 'N',
            'rna': 'N',
            'protein': 'X'
        }
        self.channel_last = channel_last
        self.residues = alphabets[alpha]
        self.default_value = ambiguous_letters[alpha]
        self.one_hot = one_hot

    def __getitem__(self, item):
        seq = super().__getitem__(item)
        if self.one_hot:
            seq = self.residues == seq[:, np.newaxis]
            if not self.channel_last:
                seq = seq.T
        return seq


class TwoBitWrapper(GenomeWrapper):
    def __init__(self, twobit_file, alpha='dna', one_hot=True, channel_last=True, in_mem=False, thread_safe=False):
        super().__init__(alpha, one_hot, channel_last, in_mem, thread_safe)
        self.twobit = py2bit.open(twobit_file)
        self._chroms = list(self.twobit.chroms().keys())
        self._chroms_size = self.twobit.chroms()
        if in_mem:
            twobit_onehot_dict = self._encode_seqs(self.twobit)
            self.twobit.close()
            self.twobit = twobit_onehot_dict
            self.thread_safe = True
        else:
            if thread_safe:
                self.twobit.close()
                self.twobit = twobit_file

    def close(self):
        if not self.thread_safe:
            self.twobit.close()

    @staticmethod
    def _encode_seqs(twobit):
        # Converts a TwoBit object into a dictionary of one-hot coded boolean matrices
        twobit_dict = {}
        pbar = tqdm(twobit.chroms())
        for chrom in pbar:
            pbar.set_description(desc='Loading sequence: ' + chrom)
            seq = twobit.sequence(chrom)
            seq = np.array(list(seq))
            twobit_dict[chrom] = seq
        return twobit_dict

    def _get_seq(self, chrom, start, stop):
        if self.in_mem:
            seq = self.twobit[chrom][start:stop]
        else:
            if self.thread_safe:
                twobit = py2bit.open(self.twobit)
                seq = np.array(list(twobit.sequence(chrom, start, stop)))
                twobit.close()
            else:
                seq = np.array(list(self.twobit.sequence(chrom, start, stop)))
        return seq


class FastaWrapper(GenomeWrapper):
    def __init__(self, fasta_file, alpha='dna', one_hot=True, channel_last=True, in_mem=False, thread_safe=False,
                 read_ahead=10000):
        super().__init__(alpha, one_hot, channel_last, in_mem, thread_safe)
        self.fasta = Fasta(fasta_file, as_raw=True, sequence_always_upper=True, read_ahead=read_ahead)
        self._chroms = list(self.fasta.keys())
        seq_lens = [len(self.fasta[chrom]) for chrom in self._chroms]
        self._chroms_size = dict(zip(self._chroms, seq_lens))
        self.read_ahead = read_ahead
        if in_mem:
            fasta_onehot_dict = self._encode_seqs(self.fasta)
            self.fasta.close()
            self.fasta = fasta_onehot_dict
            self.thread_safe = True
        else:
            if thread_safe:
                self.fasta.close()
                self.fasta = fasta_file

    def close(self):
        if not self.thread_safe:
            self.fasta.close()

    @staticmethod
    def _encode_seqs(fasta):
        # Converts a FASTA object into a dictionary of one-hot coded boolean matrices
        fasta_dict = {}
        pbar = tqdm(fasta)
        for record in pbar:
            pbar.set_description(desc='Loading sequence: ' + record.name)
            seq = record[:]
            seq = np.array(list(seq))
            fasta_dict[record.name] = seq
        return fasta_dict

    def _get_seq(self, chrom, start, stop):
        if self.in_mem:
            seq = self.fasta[chrom][start:stop]
        else:
            if self.thread_safe:
                fasta = Fasta(self.fasta, as_raw=True, sequence_always_upper=True, read_ahead=self.read_ahead)
                seq = np.array(list(fasta[chrom][start:stop]))
                fasta.close()
            else:
                seq = np.array(list(self.fasta[chrom][start:stop]))
        return seq


class BigWigWrapper(SignalWrapper):
    def __init__(self, bigwig_file, in_mem=False, thread_safe=False, default_value=0):
        super().__init__(in_mem, thread_safe)
        self.bigwig = pyBigWig.open(bigwig_file, 'r')
        self._chroms_size = self.bigwig.chroms()
        self._chroms = list(self._chroms_size.keys())
        self.default_value = default_value
        if in_mem:
            seqs_dict = self._encode(self.bigwig)
            self.bigwig.close()
            self.bigwig = seqs_dict
            self.thread_safe = True
        else:
            if thread_safe:
                self.bigwig.close()
                self.bigwig = bigwig_file

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

    def _get_seq(self, chrom, start, stop):
        if self.in_mem:
            seq = self.bigwig[chrom][start:stop]
        else:
            if self.thread_safe:
                bigwig = pyBigWig.open(self.bigwig, 'r')
                seq = bigwig.values(chrom, start, stop, numpy=True)
                bigwig.close()
            else:
                seq = self.bigwig.values(chrom, start, stop, numpy=True)
        seq[np.isnan(seq)] = self.default_value
        return seq


class BedWrapper:
    def __init__(self, bed_file):
        self.df = pd.read_table(bed_file,
                                names=['chrom', 'chromStart', 'chromEnd'],
                                usecols=[0, 1, 2])
        self.bt = pbt.BedTool(bed_file).sort()

    def __getitem__(self, item):
        return self.df.iloc[item]

    def __len__(self):
        return len(self.df)

    def shuffle(self):
        self.df = sklearn.utils.shuffle(self.df)


class BedGraphWrapper(BedWrapper):
    def __init__(self, bedgraph_file):
        self.df = pd.read_table(bedgraph_file,
                           names=['chrom', 'chromStart', 'chromEnd', 'dataValue'])
        self.bt = pbt.BedTool(bedgraph_file).sort()


class NarrowPeakWrapper(BedWrapper):
    def __init__(self, narrowpeak_file):
        self.df = pd.read_table(narrowpeak_file,
                           names=['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand',
                                  'signalValue', 'pValue', 'qValue', 'peak'])
        self.bt = pbt.BedTool(narrowpeak_file).sort()
