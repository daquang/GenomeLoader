from setuptools import setup, find_packages


setup(version='0.1.0',
      name='genomeloader',
      description="fast data loading for deep learning genomics",
      long_description=open('README.md').read(),
      url='https://github.com/daquang/GenomeLoader',
      license = "MIT",
      author="Daniel Quang",
      author_email="daquang@umich.edu",
      packages=find_packages(),
      install_requires=['numpy', 'pybedtools', 'pyfaidx', 'pandas',
                        'pyBigWig', 'py2bit', 'tqdm', 'scikit-learn']
      )
