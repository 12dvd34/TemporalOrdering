# Data Synthesis

This folder contains code for synthesizing labeled data for use with machine learning
models. It is a very simple approach that only considers well formatted dates in a few 
formats.

### Synthesizing Data

Run the `synthesize_data` function of one of the two scripts.
- `Synthesize.py` provides a basic implementation
- `SynthesizeMPI.py` is an MPI-parallelized version for use on HPC systems. It requires
an MPI implementation and the mpi4py python package.

Both calculate the BERT embeddings of the sentences and saves them, so that they can 
be directly loaded when training. 