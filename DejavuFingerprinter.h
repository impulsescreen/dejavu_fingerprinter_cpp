//
// Created by ismdev on 27/07/17.
//

#pragma once


#include <string>
#include <vector>
#include <map>
#include <memory>

namespace cv {
    class Mat;
}

class DejavuPeak
{
    public:
        int const _timeBin;
        int const _frequencyBin;

        DejavuPeak(int timeBin, int frequencyBin) : _timeBin(timeBin), _frequencyBin(frequencyBin) {};
        ~DejavuPeak(void) {};

    private:
        DejavuPeak(DejavuPeak const * rhs) = delete;
        DejavuPeak operator=(DejavuPeak const * rhs) = delete;

};

class DejavuFingerprinter
{
    public:
        int                 const           _sampleRate;
        float                               _windowSizeSeconds;          //Window_Size in dejavu
        int                                 _nfft;
        float               const           _overlapRatio;
        int                 const           _fanValue;
        float               const           _ampMin;
        int                 const           _peakNeighbourhoodSize;
        int                 const           _minHashTimeDelta;
        int                 const           _maxHashTimeDelta;
        bool                const           _peakSort;
        int                 const           _fingerprintSize; //fingerprintReduction in dejavu - here we can specify size of hash though
        float                   *           _hannWindow;
        float                               _hannWindowNorm;  //This implementation does no adjusting for windowing loss

        DejavuFingerprinter
        (
            int sampleRate,
            float windowSizeSeconds,
            float overlapRatio,
            int fanValue,
            float ampMin,
            int peakNeighbourhoodSize,
            int minHashTimeDelta,
            int maxHashTimeDelta,
            bool peakSort,
            int fingerprintSize
        );
        ~DejavuFingerprinter();

        bool CalculateMusicHashes
        (
            float * audioBuffer,
            int     audioBufferLen,
            std::vector<std::vector<unsigned char>> & hashOffsets
        );

    private:

        bool GetFilteredPeaks(std::vector<std::vector<float>> & spectrum, cv::Mat & detectedPeaks, std::vector<std::shared_ptr<DejavuPeak>> & peaks);
        bool GetDetectedPeaks(cv::Mat & localMax, cv::Mat & erodedBackground, cv::Mat & output);
        bool GetErodedBackground(cv::Mat & input, cv::Mat & structure, cv::Mat & output);
        bool BuildStructure(cv::Mat & structure, int size);
        bool MaximumFilter(cv::Mat & input, cv::Mat & structure, cv::Mat & output);
        bool VectorToMat(std::vector<std::vector<float>> & input, cv::Mat & output);
        bool Get2DPeaks(std::vector<std::vector<float>> & spectrum, std::vector<std::shared_ptr<DejavuPeak>> & peaks);
        bool GenerateHashes(std::vector<std::shared_ptr<DejavuPeak>> & peaks, std::vector<std::vector<unsigned char>> & hashOffsets);
        bool HannWindow(float * data);
        bool GetFft(float * dataIn, std::vector<float> & spectra);

};
