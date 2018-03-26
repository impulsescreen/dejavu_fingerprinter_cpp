# dejavu_fingerprinter_cpp
A port of the Dejavu `fingerprint.py` program to C++.  
Thanks to Will Drevo for developing the Dejavu Audio Fingerprinter, which can be found [here]( https://github.com/worldveil/dejavu).

### Use
* Initialise the DejavuFingerprinter with the appropriate variables. Some defaults can be found in `dejavu_fingerprinter.conf`. Modify to suit your needs.
* Read your audio sample into an array of floats.
* Declare a vector of vectors of `unsigned char`. This will hold the fingerprints. Each `vector<unsigned char>` is a single fingerprint, with its corresponding offset being its position in the fingerprint vector.
* Pass a pointer to the audio array, the length of the array, and the fingerprint vector to `DejavuFingerprinter.CalculateMusicHashes`. This will fill the fingerprint vector with the fingerprints for the audio sample.

### Differences from base Dejavu
* This implementation uses Kiss FFT to generate a spectrogram.
    * While we have attempted to replicate the process of matplotlib.mlab.specgram() as closely as possible, the fingerprints generated may not be compatible with systems that use different fourier transforms, e.g. FFTW or pythons matplotlib.mlab.
* Instead if declaring a WindowSize, we use WindowSizeSeconds, from which the window for the fft is calculated. This keeps the fft window dependent primarily on the sampling rate.
* Fingerprints are returned as a `vector<vector<unsigned char>>`, rather than Hash-Offset pairs. The offset associated with a particular hash is its position within the fingerprint vector. Offsets that do not yield a hash are represented by an empty `vector<unsigned char>`. This was done to make it simpler to transmit this information to a handler. 

## Dependencies
* [Kiss FFT](https://sourceforge.net/projects/kissfft/)
    * The `CMakeLists.txt` offered expects the required Kiss FFT libraries to be located at `${PROJECT_SOURCE_DIR}/kiss_fft130`. Modify this path where relevant in the `CMakeLists.txt`, if necessary 

