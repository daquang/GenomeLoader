import numpy as np
import pandas as pd
import pybedtools as pbt
from pybedtools.bedtool import BEDToolsError
import keras
import sklearn.utils
from .wrapper import BedWrapper


class BedGenerator(keras.utils.Sequence):
    def __init__(self, bed, genome, bigwigs=[], blacklist=None, batch_size=128, epochs=10, window_len=200, seq_len=1024,
                 negatives_ratio=1, return_sequences=False, jitter_mode='sliding', shuffle=True):
        # Initialization
        self.bed = bed
        self.genome = genome
        self.bigwigs = bigwigs
        self.blacklist = blacklist
        self.batch_size = batch_size
        self.epochs = epochs
        self.epoch_i = -1
        self.labels_epoch_i = None
        self.intervals_df_epoch_i = None
        self.window_len = window_len
        self.seq_len = seq_len
        assert seq_len > window_len
        self.negatives_ratio = negatives_ratio
        self.unet = unet
        self.jitter_mode = jitter_mode
        self.shuffle = shuffle
        # Will only shuffle intervals within chromosomes occupied by BED intervals
        genome_chromsizes = self.genome.chroms_size_pybedtools()
        bed_chroms = self.bed.chroms()
        self.chromsizes = {}
        for chrom in bed_chroms:
            self.chromsizes[chrom] = genome_chromsizes[chrom]
        self.bed.bt.set_chromsizes(self.chromsizes)
        self.negative_windows_epoch_i = None
        self.cumulative_excl_bt = None
        self._reset_negatives()
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
        x_bigwigs = len(self.bigwigs) * [[]]
        y = []
        for interval, label in zip(intervals_batch_df.itertuples(), labels_batch):
            chrom = interval[1]
            chrom_start = interval[2]
            chrom_end = interval[3]
            midpt = (chrom_start + chrom_end) / 2
            start = int(midpt - self.seq_len / 2)
            stop = start + self.seq_len
            if self.jitter_mode == 'sliding':
                shift_size = midpt - chrom_start
            elif self.jitter_mode == 'landmark':
                shift_size = self.window_len / 2
            else:
                shift_size = 0
            s = np.random.randint(-shift_size, shift_size+1)
            start += s
            stop += s
            x_genome.append(self.genome[chrom, start:stop])
            for i in range(len(self.bigwigs)):
                x_bigwigs[i].append(self.bigwigs[i][chrom, start:stop])
            if self.return_sequences:
                peaks = self.bed.search(chrom, start, stop)
                label = np.zeros((stop - start, 1), dtype=bool)
                for peak in peaks:
                    label[max(0, peak.start - start):peak.end - start, 0] = True
            y.append(label)

        x_genome = np.array(x_genome)
        if len(self.bigwigs) == 0:
            x = x_genome
        else:
            x = [x_genome]
            for x_bigwig in x_bigwigs:
                x.append(np.array(x_bigwig))
        y = np.array(y)
        return x, y

    def _reset_negatives(self):
        if self.negatives_ratio > 1:
            self.negative_windows_epoch_i = BedWrapper(pbt.BedTool.cat(*(self.negatives_ratio * [self.bed.bt]),
                                                                       postmerge=False).saveas().fn)
        elif self.negatives_ratio == 1:
            self.negative_windows_epoch_i = self.bed
        else:
            self.negative_windows_epoch_i = BedWrapper(pbt.BedTool([]).fn)
        self.negative_windows_epoch_i.bt.set_chromsizes(self.chromsizes)
        if self.jitter_mode == 'sliding':
            self.cumulative_excl_bt = self.bed.bt.slop(b=self.window_len/2)
        else:
            self.cumulative_excl_bt = self.bed.bt
        if self.blacklist is not None:
            self.cumulative_excl_bt = self.cumulative_excl_bt.cat(self.blacklist.bt)

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
            self.negative_windows_epoch_i.bt.set_chromsizes(self.chromsizes)
        except BEDToolsError:  # Cannot find any more non-overlapping intervals, reset
            self._reset_negatives()
        labels_epoch_i = np.zeros((self.negatives_ratio + 1) * len(self.bed), dtype=bool)
        labels_epoch_i[:len(self.bed)] = True
        self.intervals_df_epoch_i = pd.concat([self.bed.df, self.negative_windows_epoch_i.df])
        if self.shuffle:
            self.intervals_df_epoch_i, self.labels_epoch_i = sklearn.utils.shuffle(self.intervals_df_epoch_i,
                                                                                   labels_epoch_i)


class BedGraphGenerator(keras.utils.Sequence):
    def __init__(self, bedgraph, genome, bigwigs=[], batch_size=128, seq_len=1024, shuffle=True):
        # Initialization
        self.bedgraph = bedgraph
        self.genome = genome
        self.batch_size = batch_size
        self.bigwigs = bigwigs
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(1.0 * len(self.bedgraph) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Collect genome intervals of the batch
        intervals_df = self.bedgraph[index*self.batch_size:(index+1)*self.batch_size]
        x_genome = []
        x_bigwigs = len(self.bigwigs) * [[]]
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
            x_genome.append(self.genome[chrom, start:stop])
            for i in range(len(self.bigwigs)):
                x_bigwigs[i].append(self.bigwigs[i][chrom, start:stop])
            y.append(label)

        x_genome = np.array(x_genome)
        if len(self.bigwigs) == 0:
            x = x_genome
        else:
            x = [x_genome]
            for x_bigwig in x_bigwigs:
                x.append(np.array(x_bigwig))
        y = np.array(y)
        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.bedgraph.shuffle()
