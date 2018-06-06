# GenomeLoader

---

## Table of Contents
* [Citation](#citation)
* [Installation](#installation)
    * [Required dependencies](#required-dependencies)
* [Examples](#examples)
    * [BedGraph](#bedgraph)
* [To-Do](#to-do)

---

## Citation

Coming soon

## Installation
Clone the repository and run:
```
python setup.py develop
```

develop is preferred over install because I will be constantly updating this repository.

The best and easiest way to install all dependencies is with [Anaconda](https://www.anaconda.com/) (5.1, Python 3.6 
version). Anaconda uses pre-built binaries for specific operating systems to allow simple installation of Python and 
non-Python software packages. macOS High Sierra or Ubuntu 18.04 is recommended.

### Required dependencies
* [pyfaidx](https://github.com/mdshw5/pyfaidx) (0.5.2). Python wrapper module for indexing, retrieval, and in-place 
modification of FASTA files using a samtools compatible index. Easily installed in Anaconda with the following command 
line:
```
pip install pyfaidx
```
* [py2bit] . Preferred over genome FASTA files, due to its faster data retrieval and smaller file size footprint.
* [pyBigWig](https://github.com/deeptools/pyBigWig) (0.3.11). A python extension for quick access to bigWig and bigBed 
files: Easily installed in Anaconda with the following command line:
```
pip install pyBigWig
```
* [pybedtools](https://daler.github.io/pybedtools/) (0.7.10). BEDTools wrapper and extension that offers feature-level 
manipulations from within Python.
* [quicksect]
* [keras]
* [tqdm]
* [pandas]
* [scikit-learn]
* [numpy]

### Optional dependencies
* [biopython](http://biopython.org/) (1.7.0). Required to read bgzipped FASTA files. Convenient for large genome files.
```
conda install -c anaconda biopython
```
---
## Examples

### BedGraph
This is an example of loading a BedGraph file for training a simple keras model. It uses the same files and follows the 
same format from the genomelake repository. You will need to first download the following files:
* [hg19.2bit](hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.2bit). The hg19 genome in 2.bit format. Although 
FASTA files are also allowed, .2bit files are preferred.
* [JUND.HepG2.chr22.101bp_intervals.tsv.gz](https://github.com/kundajelab/genomelake/raw/master/examples/JUND.HepG2.chr22.101bp_intervals.tsv.gz).
BedGraph example from the genomelake Github repository. 
```python
from genomeloader.wrapper import TwoBitWrapper, BedGraphWrapper
from genomeloader.generator import BedGraphGenerator
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense

t = TwoBitWrapper('hg19.2bit')
bg = BedGraphWrapper('JUND.HepG2.chr22.101bp_intervals.tsv.gz')
datagen = BedGraphGenerator(bg, t, seq_len=None) # The bedGraph file is already pre-sized to contain 101 bp intervals

model = Sequential()
model.add(Conv1D(15, 25, input_shape=(101, 4)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(datagen, steps_per_epoch=100)

```

Here is the the expected result:
```
100/100 [==============================] - 1s 10ms/step - loss: 0.1575 - acc: 0.9887
```

---

## To-Do
Here is a list of features I plan to add. They will be added according to demand.
* Multi-task loading
* Batch loading with bigWig data
* Interval outputs