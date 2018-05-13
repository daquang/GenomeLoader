import numpy as np
import keras


class BedGenerator(keras.utils.Sequence):
    def __init__(self, bed, genome, batch_size=128, seq_len=1000,  shuffle=True):
        # Initialization
        self.bed = bed
        self.genome = genome
        self.batch_size = batch_size
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
        X = []
        y = []
        for interval in intervals_df.itertuples():
            chrom = interval[1]
            chromStart = interval[2]
            chromEnd = interval[3]
            label = interval[4]
            X.append(self.genome[chrom, chromStart:chromEnd])
            y.append(label)

        X = np.array(X)
        y = np.array(y)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.bedgraph.shuffle()

    def _generate_negatives(self):
        return


class BedGraphGenerator(keras.utils.Sequence):
    def __init__(self, bedgraph, genome, batch_size=128, seq_len=1000,  shuffle=True):
        # Initialization
        self.bedgraph = bedgraph
        self.genome = genome
        self.batch_size = batch_size
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
        X = []
        y = []
        for interval in intervals_df.itertuples():
            chrom = interval[1]
            chromStart = interval[2]
            chromEnd = interval[3]
            label = interval[4]
            X.append(self.genome[chrom, chromStart:chromEnd])
            y.append(label)

        X = np.array(X)
        y = np.array(y)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            self.bedgraph.shuffle()
