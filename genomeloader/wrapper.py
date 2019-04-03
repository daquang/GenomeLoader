import numpy as np
import pandas as pd
import pybedtools as pbt
from collections import OrderedDict
from pyfaidx import Fasta
import py2bit
import pyBigWig
from intervaltree import IntervalTree
from tqdm import tqdm
import sklearn.utils


class SignalWrapper:
    def __init__(self, channel_last=True, in_mem=False, thread_safe=False):
        self.channel_last = channel_last
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
        super().__init__(channel_last, in_mem, thread_safe)
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
    def __init__(self, bigwig_file, channel_last=True, in_mem=False, thread_safe=False, default_value=0):
        super().__init__(channel_last, in_mem, thread_safe)
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

    def __getitem__(self, item):
        seq = super().__getitem__(item)
        if self.channel_last:
            seq = seq.reshape((-1, 1))
        else:
            seq = seq.reshape((1, -1))
        return seq


class GenomicIntervalTree(dict):
    def add(self, chrom, start, stop, data=None):
        if chrom not in self:
            self[chrom] = IntervalTree()
        self[chrom].addi(start, stop, data)

    def search(self, chrom, start, stop):
        if chrom not in self:
            return []
        return self[chrom].overlap(start, stop)


class BedWrapper:
    def __init__(self, bed_file, col_names=['chrom', 'chromStart', 'chromEnd'], channel_last=True, sort_bed=True,
                 data_col=None, dtype=bool):
        self.col_names = col_names
        self.data_col = data_col
        col_indices = list(range(len(col_names)))
        self.channel_last = channel_last
        self.df = pd.read_csv(bed_file,
                              sep='\t',
                              names=col_names,
                              usecols=col_indices)
        self._chroms = self.df.chrom.unique()
        self.bt = pbt.BedTool(bed_file)
        if sort_bed:
            self.bt = pbt.BedTool(bed_file).sort()
        self.genomic_interval_tree = GenomicIntervalTree()
        use_data = data_col is not None
        self.dtype = dtype
        for interval in self.df.itertuples():
            chrom = interval[1]
            start = interval[2]
            stop = interval[3]
            data = interval[data_col] if use_data else True
            self.genomic_interval_tree.add(chrom, start, stop, data)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        chrom, coords = item
        start = coords.start
        stop = coords.stop
        if chrom in self._chroms:
            intervals = self.search(chrom, start, stop)
        else:
            intervals = set()
        seq = np.zeros((stop - start, 1), dtype=self.dtype)
        for interval in intervals:
            seq[max(0, interval.begin - start):interval.end - start, 0] = interval.data
        if not self.channel_last:
            seq = seq.T
        return seq

    def shuffle(self):
        self.df = sklearn.utils.shuffle(self.df)

    def chroms(self):
        return self._chroms

    def sum_intervals(self):
        return sum(self.df.chromEnd - self.df.chromStart)

    def search(self, chrom, start, stop):
        intervals = self.genomic_interval_tree.search(chrom, start, stop)
        return intervals

    def train_valid_test_split(self, valid_chroms, test_chroms=[]):
        inds_valid = self.df['chrom'].isin(valid_chroms)
        inds_test = self.df['chrom'].isin(test_chroms)
        inds_train = ~inds_valid & ~inds_test
        df_train = self.df.loc[inds_train]
        df_valid = self.df.loc[inds_valid]
        df_test = self.df.loc[inds_test]
        bed_str_train = df_train.to_string(header=False, index=False) if len(df_train) > 0 else ''
        bed_str_valid = df_valid.to_string(header=False, index=False) if len(df_valid) > 0 else ''
        bed_str_test = df_test.to_string(header=False, index=False) if len(df_test) > 0 else ''
        bed_file_train = pbt.BedTool(bed_str_train, from_string=True).fn
        bed_file_valid = pbt.BedTool(bed_str_valid, from_string=True).fn
        bed_file_test = pbt.BedTool(bed_str_test, from_string=True).fn
        bed_train = BedWrapper(bed_file_train, col_names=self.col_names, channel_last=self.channel_last,
                               data_col=self.data_col, dtype=self.dtype)
        bed_valid = BedWrapper(bed_file_valid, col_names=self.col_names, channel_last=self.channel_last,
                               data_col=self.data_col, dtype=self.dtype)
        bed_test = BedWrapper(bed_file_test, col_names=self.col_names, channel_last=self.channel_last,
                              data_col=self.data_col, dtype=self.dtype)
        return bed_train, bed_valid, bed_test


class BedGraphWrapper(BedWrapper):
    def __init__(self, bedgraph_file, channel_last=True):
        super().__init__(bedgraph_file,
                         col_names=['chrom', 'chromStart', 'chromEnd', 'dataValue'],
                         channel_last=channel_last, data_col=4,
                         dtype=np.float32)


class NarrowPeakWrapper(BedWrapper):
    def __init__(self, narrowpeak_file, channel_last=True):
        super().__init__(narrowpeak_file,
                         col_names=['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'signalValue',
                                    'pValue', 'qValue', 'peak'],
                         channel_last=channel_last, data_col=7,
                         dtype=np.float32)


class BroadPeakWrapper(BedWrapper):
    def __init__(self, narrowpeak_file, channel_last=True):
        super().__init__(narrowpeak_file,
                         col_names=['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand', 'signalValue',
                                    'pValue', 'qValue'],
                         channel_last=channel_last, data_col=8,
                         dtype=np.float32)
