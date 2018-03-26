//
// Created by James Ferris on 27/07/17.
//

#include "DejavuFingerprinter.h"
#include <complex>
#include <cstring>
#include <openssl/sha.h>
#include "kiss_fftr.h"
#include <opencv2/opencv.hpp>
#include <cpprest/asyncrt_utils.h>



DejavuFingerprinter::DejavuFingerprinter
(
    int     sampleRate,
    float   windowSizeSeconds,
    float   overlapRatio,
    int     fanValue,
    float   ampMin,
    int     peakNeighbourhoodSize,
    int     minHashTimeDelta,
    int     maxHashTimeDelta,
    bool    peakSort,
    int     fingerprintSize
) :
    _sampleRate(sampleRate),
    _windowSizeSeconds(windowSizeSeconds),
    _nfft(0),
    _overlapRatio(overlapRatio),
    _fanValue(fanValue),
    _ampMin(ampMin),
    _peakNeighbourhoodSize(peakNeighbourhoodSize),
    _minHashTimeDelta(minHashTimeDelta),
    _maxHashTimeDelta(maxHashTimeDelta),
    _peakSort(peakSort),
    _fingerprintSize(fingerprintSize),
    _hannWindow(nullptr),
    _hannWindowNorm(0)
{
    //
    // Initialise HannWindow
    //

    //Calculate nfft
    _nfft = round(_windowSizeSeconds * _sampleRate);
    _nfft += _nfft % 2; //nfft must always be even - adjust if necessary

    _hannWindow = new float[_nfft];

    for (int n = 0; n < _nfft; ++n)
    {
        double HannMultiplier = 0.5 * ( 1 - cos( 2 * M_PI * n / (_nfft - 1) ) );
        _hannWindow[n] = HannMultiplier;
        _hannWindowNorm += HannMultiplier * HannMultiplier;
    }
}

DejavuFingerprinter::~DejavuFingerprinter()
{
    delete [] _hannWindow;
}

/**
 *  Master function for calculating hashes for music
 *
 * @param audioBuffer - Location of audio data
 * @param audioBufferLen - Amount of audio data stored
 * @param sampleRate - Sampling rate at which audio was obtained
 * @param hashOffsets - Vector of vector<unsigned char> to store the resulting hashes at their corresponding offsets
 * @return always true
 */

bool
DejavuFingerprinter::CalculateMusicHashes
(
    float * audioBuffer,
    int     audioBufferLen,
    std::vector<std::vector<unsigned char>> & hashOffsets
)
{
    ///Min frequency/frequency resolution is nfft/sampleRate
    ///Total number of frequency bins is nfft/2 + 1

    float *aB = audioBuffer;
    int overlap = _nfft * _overlapRatio;

    int bins = (audioBufferLen - _nfft)/overlap;
    std::vector<std::vector<float>> spectrum;

    float buffer[_nfft] = {0};

    for (int i = 0; i < bins; ++i)
    {
        std::memset(buffer, 0, sizeof(float) * (_nfft));
        std::copy(aB, aB + _nfft, buffer);

        std::vector<float> spectra;

        HannWindow(buffer);
        GetFft(buffer, spectra);

        aB += overlap;
        spectrum.push_back( spectra );
    }

    std::vector<std::shared_ptr<DejavuPeak>> peaks;
    Get2DPeaks(spectrum, peaks);

    GenerateHashes(peaks, hashOffsets);

    return true;
}


/**
 *  Calculates the fast fourier transform of real input data over a portion of audio,
 *  and transforms the result into a real power spectral density
 *
 * @param dataIn - start buffer of _nfft size holding a portion of input audio
 * @param spectra - output vector to hold the calculated power spectra for each frequency bin
 * @param windowNorm - the norm of the Hann Window, used to scale values to compensate for windowing loss
 * @return always true
 */

bool
DejavuFingerprinter::GetFft(float * dataIn, std::vector<float> & spectra)
{
    std::vector<float> x;
    x.assign(dataIn, dataIn + _nfft);

    std::vector<std::complex<float>> fx(_nfft);

    kiss_fftr_cfg fwd = kiss_fftr_alloc(_nfft, 0, NULL, NULL);

    kiss_fftr(fwd,&x[0],(kiss_fft_cpx*) &fx[0]);

    for (int k = 0; k < _nfft/2 + 1; ++k)
    {
        float val = (fx[k] * conj(fx[k])).real();    // == re^2 + im^2
        ///Python matplotlib.mlab.specgram has scale_by_freq=None,
        ///  which makes it default to true
        ///  Thus,
        ///      result /= Fs //so that density function has units dB/Hz
        ///      result /= (np.abs(windowVals)**2).sum() //scale by the norm of the window to compensate for windowing loss
//        val *= 1./_sampleRate;
//        val *= 1./_hannWindowNorm;
        ///Currently trying to adjust for frequency and windowing loss results in far too few hashes (i.e. 0) being recognized.
        ///Without, we obtain a similar number of hashes as the python version, so kiss_fft may be doing something
        ///similar internally, or the kiss_fftr is different enough that such adjustments are not necessary.

        ///Calculate power for current frequency bin
        val = 10 * (log10(val)); //not 20log(fx), as we haven't taken the square root of the square sum/conjugate.

        if (val == - INFINITY || std::isnan((val)))   //if values is -inf, or nan, set to zero
        {
            val = 0;
        }
        spectra.push_back(val);
    }

    kiss_fft_free(fwd);

    return true;
}

/**
 * Filters out the desired peaks from those detected based on _ampMin
 *
 * @param spectrum - vector holding the power density spectra for each time bin
 * @param detectedPeaks - cv::Mat holding the detected peaks
 * @param peaks - vector for the output peaks after filtering
 * @return always true
 */

bool
DejavuFingerprinter::GetFilteredPeaks
(
    std::vector<std::vector<float>>          & spectrum,
    cv::Mat                                  & detectedPeaks,
    std::vector<std::shared_ptr<DejavuPeak>> & peaks
)
{
    for (int i = 0; i < detectedPeaks.rows; ++i)
    {
        for (int j = 0; j < detectedPeaks.cols; ++j)
        {
            if (detectedPeaks.at<float>(i,j) > 0)
            {
                if (spectrum[i][j] > _ampMin)
                {
                    auto peak = std::make_shared<DejavuPeak>(i, j);
                    peaks.push_back(peak);
                }
            }
        }
    }

    return true;
}

/**
 * Masks the local maximum of peaks using the eroded background to obtain detected peaks
 *
 * @param localMax - cv::Mat of the local maximums
 * @param erodedBackground - boolean structure to act as a mask
 * @param output
 * @return always true
 */

bool
DejavuFingerprinter::GetDetectedPeaks(cv::Mat & localMax, cv::Mat & erodedBackground, cv::Mat & output)
{
    for (int i = 0; i < localMax.rows; ++i)
    {
        for (int j = 0; j < localMax.cols; ++j)
        {
            output.at<float>(i,j) = localMax.at<float>(i,j) - erodedBackground.at<float>(i,j);
        }
    }

    return true;
}

/**
 * Erodes the originally spectrum by the neighbourhood mask to obtain a background
 *
 * @param input - the originally power density spectrum of the audio data
 * @param structure - the neighbourhood mask developed
 * @param output - the eroded background
 * @return always true
 */

bool
DejavuFingerprinter::GetErodedBackground(cv::Mat & input, cv::Mat & structure, cv::Mat & output)
{
    //First get background
    for (int i = 0; i < input.rows; ++i)
    {
        for (int j = 0; j < input.cols; ++j)
        {
            if (input.at<float>(i,j) == 0)
            {
                output.at<float>(i,j) = 1;
            }
            else
            {
                output.at<float>(i,j) = 0;
            }
        }
    }
    //Now erode once, with structure anchored at its center, for a border value of 1
    cv::erode(output, output, structure, cv::Point(-1,-1), 1, cv::BORDER_CONSTANT, 1);

    return true;
}

/**
 * Locates the maximum values within a neighbourhood and returns a matrix of those values
 *
 * @param input - the original power density spectrum
 * @param structure - the binary mask developed for the neighbourhood
 * @param output - a cv::Mat of the local maximums
 * @return always true
 */

bool
DejavuFingerprinter::MaximumFilter(cv::Mat & input, cv::Mat & structure, cv::Mat & output)
{
    //Dilate input by structure, with structure anchored at its center, once, with a border value of default.
    //Store the result in output.
    cv::dilate(input, output, structure, cv::Point(-1,-1), 1);

    for (int i = 0; i < output.rows; ++i)
    {
        for (int j = 0; j < output.cols; ++j)
        {
            if (output.at<float>(i,j) != input.at<float>(i,j)) {
                output.at<float>(i,j) = 0;
            }
            else
            {
                output.at<float>(i,j) = 1;
            }
        }
    }

    return true;
}

/**
 * Converts a vector of vectors to a 2D cv::Mat
 *
 * @param input - the vector of vector to convert
 * @param output - the output cv::Mat
 * @return always true
 */

bool
DejavuFingerprinter::VectorToMat(std::vector<std::vector<float>> & input, cv::Mat & output)
{
    for (int i = 0; i < output.rows; ++i)
    {
        for (int j = 0; j < output.cols; ++j)
        {
            output.at<float>(i,j) = input.at(i).at(j);
        }
    }

    return true;
}

/**
 * Builds a binary mask to define a neighbourhood on the input data
 * Simulates the result of dilating a cross mask _peakNeighbourhoodSize times
 *
 * @param structure - the pre-initialised square cv::Mat structure of appropriate dimensions (size*size)
 * @param size - the dimensions of the structure
 * @return always true
 */
bool
DejavuFingerprinter::BuildStructure(cv::Mat & structure, int size)
{
    int midpoint = (size - 1)/2;

    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            int x = i;
            int y = j;
            if (i > midpoint)
            {
                x = (size - 1) - i;
            }
            if (j > midpoint)
            {
                y = (size - 1) - j;
            }

            if (x + y >= midpoint)
            {
                structure.at<uchar>(i,j) = 1;
            }
            else
            {
                structure.at<uchar>(i,j) = 0;
            }
        }
    }

    return true;
}

/**
 * Finds the relevant peaks within the original power spectrum density to generate hashes on
 *
 * @param spectrum - the calculated power spectrum density, as a vector of vectors
 * @param peaks - the output vector of DejavuPeaks
 * @return always true
 */

bool
DejavuFingerprinter::Get2DPeaks(std::vector<std::vector<float>> & spectrum, std::vector<std::shared_ptr<DejavuPeak>> & peaks)
{
    //Define neighbourhood structure - is 3x3 cross structure dilated by itself Pn times
    int size = 1 + 2 * _peakNeighbourhoodSize; //for Pn=20, size = 41x41

    cv::Mat neighbourhood(size, size, CV_8U, cvScalar(0.0));   //initialise to zero
    BuildStructure(neighbourhood, size);

    cv::Mat spectrumMat(spectrum.size(), spectrum.at(0).size(), CV_32FC1, 0.0);
    VectorToMat(spectrum, spectrumMat);

    cv::Mat localMaximum(spectrum.size(), spectrum.at(0).size(), CV_32FC1, 0.0);
    MaximumFilter(spectrumMat, neighbourhood, localMaximum);

    cv::Mat erodedBackground(spectrum.size(), spectrum.at(0).size(), CV_32FC1, 0.0);
    GetErodedBackground(spectrumMat, neighbourhood, erodedBackground);

    cv::Mat detectedPeaks(spectrum.size(), spectrum.at(0).size(), CV_32FC1, 0.0);
    GetDetectedPeaks(localMaximum, erodedBackground, detectedPeaks);

    GetFilteredPeaks(spectrum, detectedPeaks, peaks);

    return true;
}

/**
 * Calculates the SHA1 hashes of the calculated peaks and stores the result in a hashmap (simulating a Dictionary type)
 *
 * @param peaks - vector of DejavuPeaks
 * @param hashOffsets - Vector of vector<unsigned char> to store the resulting hashes at their corresponding offsets. Each vector is
 *      associated with a single offset - offsets with no corresponding hashes are represented by an empty vector
 * @return always true
 */

bool
DejavuFingerprinter::GenerateHashes(std::vector<std::shared_ptr<DejavuPeak>> & peaks, std::vector<std::vector<unsigned char>> & hashOffsets)
{
    if (_peakSort)
    {
        std::sort(peaks.begin(), peaks.end(),
            [] (std::shared_ptr<DejavuPeak> & lhs, std::shared_ptr<DejavuPeak> const & rhs) {
                return lhs->_timeBin < rhs->_timeBin;
            });
    }

    for (size_t i = 0; i < peaks.size(); ++i)
    {
        for (size_t j = 1; j < static_cast<size_t>(_fanValue); ++j)
        {
            if (i + j < peaks.size())
            {
                int freq1 = peaks[i]  ->_frequencyBin;
                int freq2 = peaks[i+j]->_frequencyBin;
                int time1 = peaks[i]  ->_timeBin;
                int time2 = peaks[i+j]->_timeBin;

                int tDelta = time2 - time1;

                if (tDelta >= _minHashTimeDelta && tDelta <= _maxHashTimeDelta)
                {
                    std::string hashSeed = std::to_string(freq1) + "|" + std::to_string(freq2) + "|" + std::to_string(tDelta);

                    std::vector<unsigned char> hash(SHA_DIGEST_LENGTH);
                    SHA1((const unsigned char *) hashSeed.c_str(), hashSeed.size(), hash.data());

                    //
                    //  SHA1 hash is by default 20 hex values (40 chars, when converted to a string)
                    //  At maximum we want all 20 hex values
                    //  By default _fingerprintSize is 10 hex values (20 chars)
                    //  Note that the dejavu code this was based on specifies a default fingerprint size of
                    //  20 characters (10 hex values)
                    //
                    int hashLength = (_fingerprintSize > SHA_DIGEST_LENGTH ) ? SHA_DIGEST_LENGTH : _fingerprintSize;
                    hash.resize(hashLength);

                    int offsets = hashOffsets.size();
                    if (offsets <= time1) {
                        for (int i = offsets; i < time1; ++i) {
                            hashOffsets.push_back(std::vector<unsigned char>());
                        }
                        hashOffsets.push_back(hash);
                    } else {
                        hashOffsets[time1].insert(hashOffsets[time1].end(), hash.begin(), hash.end());
                    }
                }
            }
        }
    }

    return true;
}

/**
 * Modifies the input data based on a Hann function, for improving the fft results. The Hann window multipliers and
 *  norm have been precomputed
 *
 * @param data - A section of the input audio
 * @return always true
 */

bool
DejavuFingerprinter::HannWindow(float *data)
{
    for (int n = 0; n < _nfft; ++n)
    {
        *(data + n) *= _hannWindow[n];
    }
    return true;
}


