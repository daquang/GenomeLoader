import numpy as np
import pandas as pd
import pybedtools as pbt
from pybedtools.bedtool import BEDToolsError
import keras
import sklearn.utils
from .wrapper import BedWrapper


class MultiBedGenerator(keras.utils.Sequence):
    def __init__(self, beds, signals, output_signals=None, extra=None, blacklist=None, batch_size=128, window_len=200,
                 seq_len=1024, output_seq_len=None, negatives_ratio=1, return_sequences=False, jitter_mode='sliding',
                 left_justify=False, epochs_reset=10, return_output=True, shuffle=True):
        # Initialization
        self.beds = beds
        beds_bt = [bed.bt for bed in beds]
        if len(beds_bt) == 1:
            master_bed_bt = beds_bt[0]
        else:
            master_bed_bt = pbt.BedTool.cat(*beds_bt, postmerge=True)
        if extra is not None:
            master_bed_bt = master_bed_bt.cat(extra.bt, postmerge=True)
        if len(beds_bt) == 1 and extra is None:
            self.master_bed = beds[0]  # simple case, do not recreate BedWrapper object
        else:
            self.master_bed = BedWrapper(master_bed_bt.fn)
        self.signals = signals
        self.output_signals = [] if output_signals is None else output_signals
        self.blacklist = blacklist
        self.batch_size = batch_size
        self.epoch_i = -1
        self.epochs_reset = epochs_reset
        self.intervals_df_epoch_i = None
        self.window_len = window_len
        if type(window_len) is not int or window_len < 0:
            raise ValueError('window_len must be positive integer')
        self.seq_len = seq_len
        if seq_len <= window_len:
            raise ValueError('seq_len must be > window_len')
        if output_seq_len is None:
            output_seq_len = seq_len
        self.output_seq_len = output_seq_len
        self.negatives_ratio = negatives_ratio
        self.return_sequences = return_sequences
        self.left_justify = left_justify
        self.return_output = return_output
        jitter_modes = ['sliding', 'detection', 'simple', None]
        if jitter_mode not in jitter_modes:
            raise ValueError('Invalid jitter mode. Expected one of: %s' % jitter_modes)
        self.jitter_mode = jitter_mode
        self.shuffle = shuffle
        # Will only shuffle intervals within chromosomes occupied by BED intervals
        genome_chromsizes = signals[0].chroms_size_pybedtools()
        bed_chroms = self.master_bed.chroms()
        self.chromsizes = {}
        for chrom in bed_chroms:
            self.chromsizes[chrom] = genome_chromsizes[chrom]
        self.master_bed.bt.set_chromsizes(self.chromsizes)
        self.negative_windows_epoch_i = None
        self.cumulative_excl_bt = None
        self._reset_negatives()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.intervals_df_epoch_i) / self.batch_size))
        # return int(np.ceil((1.0 + self.negatives_ratio) * len(self.bed) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Collect genome intervals of the batch
        intervals_batch_df = self.intervals_df_epoch_i[index * self.batch_size:(index + 1) * self.batch_size]
        x_signals = [[] for _ in range(len(self.signals))]
        y = []
        for interval in intervals_batch_df.itertuples():
            chrom = interval[1]
            chrom_start = interval[2]
            chrom_end = interval[3]
            if self.left_justify:
                midpt = chrom_start + self.output_seq_len / 2
            else:
                midpt = (chrom_start + chrom_end) / 2
            if self.jitter_mode == 'sliding':
                interval_len = chrom_end - chrom_start
                shift_size = np.max([midpt - chrom_start, int((self.window_len - interval_len) / 2)])
            elif self.jitter_mode == 'detection' or self.jitter_mode == 'sliding':
                shift_size = self.output_seq_len / 2
            else:
                shift_size = 0
            s = np.random.randint(-shift_size, shift_size + 1)
            midpt += s
            start = int(midpt - self.seq_len / 2)
            stop = start + self.seq_len
            for i in range(len(self.signals)):
                x_signals[i].append(self.signals[i][chrom, start:stop])
            label = []
            if self.return_output:
                if len(self.output_signals) == 0:
                    for bed in self.beds:
                        if self.return_sequences:
                            start_output = int(midpt - self.output_seq_len / 2)
                            stop_output = start_output + self.output_seq_len
                            label_i = bed[chrom, start_output:stop_output]
                        else:
                            start_window = int(midpt - self.window_len / 2)
                            stop_window = start_window + self.window_len
                            label_i = bed[chrom, start_window:stop_window].sum() >= self.window_len / 2
                        label.append(label_i)
                else:
                    for output_signal in self.output_signals:
                        if self.return_sequences:
                            start_output = int(midpt - self.output_seq_len / 2)
                            stop_output = start_output + self.output_seq_len
                            label_i = output_signal[chrom, start_output:stop_output]
                        else:
                            start_window = int(midpt - self.window_len / 2)
                            stop_window = start_window + self.window_len
                            label_i = output_signal[chrom, start_window:stop_window].mean()
                        label.append(label_i)
                if self.return_sequences:
                    label = np.concatenate(label, axis=-1)
                y.append(label)

        x = [np.array(x_signal) for x_signal in x_signals]
        if len(x) == 1:
            x = x[0]
        if self.return_output:
            y = np.array(y)
            return x, y
        return x

    def _reset_negatives(self):
        if self.negatives_ratio > 1:
            self.negative_windows_epoch_i = BedWrapper(pbt.BedTool.cat(*(self.negatives_ratio * [self.bed.bt]),
                                                                       postmerge=False).fn)
        elif self.negatives_ratio == 1:
            self.negative_windows_epoch_i = self.master_bed
        else:
            self.negative_windows_epoch_i = BedWrapper(pbt.BedTool([]).fn)
        self.negative_windows_epoch_i.bt.set_chromsizes(self.chromsizes)
        if self.jitter_mode == 'sliding':
            self.cumulative_excl_bt = self.master_bed.bt.slop(b=self.window_len / 2)
        else:
            self.cumulative_excl_bt = self.master_bed.bt
        if self.blacklist is not None:
            self.cumulative_excl_bt = self.cumulative_excl_bt.cat(self.blacklist.bt)

    def on_epoch_end(self):
        self.epoch_i += 1
        if self.epoch_i > 0 and not self.shuffle:
            return
        # Updates indexes after each epoch if shuffling is desired
        try:
            if self.epoch_i != 0 and self.epoch_i % self.epochs_reset == 0:
                self._reset_negatives()
                raise BEDToolsError(cmd=None, msg=None)
            self.cumulative_excl_bt = pbt.BedTool(self.cumulative_excl_bt.cat(self.negative_windows_epoch_i.bt,
                                                                              postmerge=True))
            negative_windows_bt = self.negative_windows_epoch_i.bt.shuffle(excl=self.cumulative_excl_bt.fn,
                                                                           noOverlapping=True,
                                                                           seed=np.random.randint(
                                                                               np.iinfo(np.uint32).max + 1),
                                                                           maxTries=3)
            self.negative_windows_epoch_i = BedWrapper(negative_windows_bt.fn)
            self.negative_windows_epoch_i.bt.set_chromsizes(self.chromsizes)
        except BEDToolsError:  # Cannot find any more non-overlapping intervals or on a 10th epoch, reset
            self._reset_negatives()
        intervals_df_list_epoch_i = [self.master_bed.df, self.negative_windows_epoch_i.df]
        self.intervals_df_epoch_i = pd.concat(intervals_df_list_epoch_i)
        if self.shuffle:
            self.intervals_df_epoch_i = sklearn.utils.shuffle(self.intervals_df_epoch_i)


class BedGraphGenerator(keras.utils.Sequence):
    def __init__(self, bedgraph, genome, signals=[], batch_size=128, seq_len=1024, return_sequences=False, shuffle=True):
        # Initialization
        self.bedgraph = bedgraph
        self.genome = genome
        self.batch_size = batch_size
        self.signals = signals
        self.seq_len = seq_len
        self.return_sequences = return_sequences
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(1.0 * len(self.bedgraph) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Collect genome intervals of the batch
        intervals_df = self.bedgraph.df[index*self.batch_size:(index+1)*self.batch_size]
        x_genome = []
        x_signals = [[] for _ in range(len(self.signals))]
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
            for i in range(len(self.signals)):
                x_signals[i].append(self.signals[i][chrom, start:stop])
            if self.return_sequences:
                label = self.bedgraph[chrom, start:stop]
            y.append(label)

        x_genome = np.array(x_genome)
        if len(self.signals) == 0:
            x = x_genome
        else:
            x = [x_genome]
            for x_signal in x_signals:
                x.append(np.array(x_signal))
        y = np.array(y)
        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.bedgraph.shuffle()
