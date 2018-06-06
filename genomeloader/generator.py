import numpy as np
import pandas as pd
import pybedtools as pbt
from pybedtools.bedtool import BEDToolsError
import keras
import sklearn.utils
from .wrapper import BedWrapper, BedGraphWrapper


class BedGenerator(keras.utils.Sequence):
    def __init__(self, bed, genome, bigwigs=[], batch_size=128, epochs=10, window_len=200, seq_len=1024,
                 negatives_ratio=1, jitter_mode='sliding', shuffle=True):
        # Initialization
        self.bed = bed
        self.genome = genome
        self.bigwigs = bigwigs
        self.batch_size = batch_size
        self.epochs = epochs
        self.epoch_i = -1
        self.labels_epoch_i = None
        self.intervals_df_epoch_i = None
        self.window_len = window_len
        self.seq_len = seq_len
        assert seq_len > window_len
        self.negatives_ratio = negatives_ratio
        self.jitter_mode = jitter_mode
        self.shuffle = shuffle
        self.bed.bt.set_chromsizes(self.genome.chroms_size_pybedtools())
        if self.negatives_ratio > 1:
            self.negative_windows_epoch_i = BedWrapper(pbt.BedTool.cat(*(self.negatives_ratio * [self.bed.bt]),
                                                                       postmerge=False).saveas().fn)
            self.negative_windows_epoch_i.bt.set_chromsizes(self.genome.chroms_size_pybedtools())
        else:
            self.negative_windows_epoch_i = self.bed
        if self.jitter_mode == 'sliding':
            self.cumulative_excl_bt = self.bed.bt.slop(b=self.window_len/2)
        else:
            self.cumulative_excl_bt = self.bed.bt
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil((1.0 + self.negatives_ratio) * len(self.bed) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Collect genome intervals of the batch
        intervals_batch_df = self.intervals_df_epoch_i[index*self.batch_size:(index+1)*self.batch_size]
        labels_batch = self.labels_epoch_i[index*self.batch_size:(index+1)*self.batch_size]
        x_genome = []
        y = []
        for interval, label in zip(intervals_batch_df.itertuples(), labels_batch):
            chrom = interval[1]
            chrom_start = interval[2]
            chrom_end = interval[3]
            midpt = (chrom_start + chrom_end) / 2
            start = int(midpt - self.seq_len / 2)
            stop = start + self.seq_len
            if self.jitter_mode == 'sliding':
                shift_size = np.max((self.window_len / 2 + 1, chrom_end - self.window_len / 2 - chrom_start + 1))
            elif self.jitter_mode == 'landmark':
                shift_size = midpt - chrom_start
            else:
                shift_size = 0
            s = np.random.randint(-shift_size, shift_size+1)
            start += s
            stop += s
            x_genome.append(self.genome[chrom, start:stop])
            y.append(label)

        x_genome = np.array(x_genome)
        if len(self.bigwigs) == 0:
            x = x_genome
        y = np.array(y)
        return x, y

    def on_epoch_end(self):
        self.epoch_i += 1
        if self.epoch_i > 0 and not self.shuffle:
            return
        'Updates indexes after each epoch if shuffling is desired'
        try:
            self.cumulative_excl_bt = self.cumulative_excl_bt.cat(self.negative_windows_epoch_i.bt)
            negative_windows_bt = self.negative_windows_epoch_i.bt.shuffle(excl=self.cumulative_excl_bt.fn,
                                                                           noOverlapping=True,
                                                                           seed=np.random.randint(
                                                                               np.iinfo(np.uint32).max+1))
            self.negative_windows_epoch_i = BedWrapper(negative_windows_bt.fn)
            self.negative_windows_epoch_i.bt.set_chromsizes(self.genome.chroms_size_pybedtools())
        except BEDToolsError:  # Cannot find any more non-overlapping intervals, reset
            if self.negatives_ratio > 1:
                self.negative_windows_epoch_i = BedWrapper(
                    pbt.BedTool.cat(*(self.negatives_ratio * self.epochs * [self.bed.bt]), postmerge=False).saveas().fn)
                self.negative_windows_epoch_i.bt.set_chromsizes(self.genome.chroms_size_pybedtools())
            else:
                self.negative_windows_epoch_i = self.bed
            if self.jitter_mode == 'sliding':
                self.cumulative_excl_bt = self.bed.bt.slop(b=self.window_len / 2)
            else:
                self.cumulative_excl_bt = self.bed.bt
        labels_epoch_i = np.zeros((self.negatives_ratio + 1) * len(self.bed), dtype=bool)
        labels_epoch_i[:len(self.bed)] = True
        self.intervals_df_epoch_i = pd.concat([self.bed.df, self.negative_windows_epoch_i.df])
        self.intervals_df_epoch_i, self.labels_epoch_i = sklearn.utils.shuffle(self.intervals_df_epoch_i,
                                                                               labels_epoch_i)


class BedGraphGenerator(keras.utils.Sequence):
    def __init__(self, bedgraph, genome, batch_size=128, seq_len=1024, incl_chroms=None, shuffle=True):
        # Initialization
        self.bedgraph = bedgraph
        self.genome = genome
        self.batch_size = batch_size
        self.seq_len = seq_len
        # Remove intervals that are not in included chromosomes (default: use all chromosomes)
        self.incl_chroms = incl_chroms
        if incl_chroms is not None:
            new_bt = self.bedgraph.filter(lambda feature: feature.chrom in incl_chroms).saveas()
            self.bedgraph = BedGraphWrapper(new_bt.fn)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(1.0 * len(self.bedgraph) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Collect genome intervals of the batch
        intervals_df = self.bedgraph[index*self.batch_size:(index+1)*self.batch_size]
        x = []
        y = []
        for interval in intervals_df.itertuples():
            chrom = interval[1]
            chrom_start = interval[2]
            chrom_end = interval[3]
            label = interval[4]
            if self.seq_len is None:
                start = chrom_start
                stop = chrom_end
            else:
                midpt = (chrom_start + chrom_end) / 2
                start = int(midpt - self.seq_len / 2)
                stop = start + self.seq_len
            x.append(self.genome[chrom, start:stop])
            y.append(label)

        x = np.array(x)
        y = np.array(y)
        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.bedgraph.shuffle()
