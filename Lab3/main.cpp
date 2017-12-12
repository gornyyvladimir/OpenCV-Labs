#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <QDebug>

void equalizeHistogram(IplImage * src, IplImage * dst) {
    int bins = 256;
    int counts[256] = {0};
    double maxValue = 0;
    double minValue = 9999;
    int x;
    int y;

    for(y = 0; y < src->height; y++)
        for(x = 0; x < src->width; x++)
        {
            double pixel = cvGetReal2D(src, y, x);

            if (pixel>maxValue) maxValue = pixel;
            if (pixel<minValue) minValue = pixel;
        }

    for(y = 0; y < src->height; y++)
        for(x = 0; x < src->width; x++)
        {
            double pixel = cvGetReal2D(src, y, x);
            //int bin = static_cast<int>(((pixel/maxValue) * range));
            int bin = static_cast<int>(pixel);
            counts[bin]++;
        }

    double pis[256];
    double imageSize = src->width * src->height;

    for (int i=0; i<bins; i++) {
        pis[i] = (counts[i] / imageSize);
        //qDebug() << "bin: " << i << " cnt: " << counts[i] << " pis: " << pis[i];
    }

    for (int i=1; i<bins; i++) {
        pis[i] = pis[i-1] + pis[i];
    }

    for(y = 0; y < src->height; y++)
        for(x = 0; x < src->width; x++)
        {
            double pixel = cvGetReal2D(src, y, x);
            //int bin = static_cast<int>(((pixel/maxValue) * range));
            int bin = static_cast<int>(pixel);
            pixel = (maxValue - minValue) * pis[bin] + minValue;
            cvSetReal2D(dst, y, x, pixel);
        }
}

//Гистограмма
void opencvHistogram(IplImage * src, IplImage * dst)
{
    CvHistogram * hist;
    int bins = 256;
    float ranges[] = {0, 255};
    float * histRanges[] = {ranges};
    float maxVal=0;
    int bin;
    double val;
    int x;
    int y;

    hist = cvCreateHist(1, &bins, CV_HIST_ARRAY, histRanges, 1);
    cvCalcHist(&src, hist, 0, NULL);
    cvGetMinMaxHistValue(hist, NULL, &maxVal, NULL, NULL);
    for(bin = 0, x = 0; bin < bins; bin++, x++)
    {
        val = cvGetReal1D(hist->bins, bin);
        y = static_cast<int>(239 - val / static_cast<double>(maxVal) * 220);

        //qDebug() << "opencv " << "bin: " << bin << "val: " << val << "y: " << y << "maxvalue: " << maxVal;

        cvLine(dst, cvPoint(x, 239), cvPoint(x, y),
               cvScalar(0, 0, 0, 0), 1, 8, 0);
    }
}

//Моя гистограмма
void histogram(IplImage * src, IplImage * dst)
{
    int bins = 256;
    int counts[256] = {0};
    double val;
    int x;
    int y;

    for(y = 0; y < src->height; y++)
        for(x = 0; x < src->width; x++)
        {
            double pixel = cvGetReal2D(src, y, x);
            int bin = static_cast<int>(pixel);
            counts[bin]++;
        }

    //Почему то гистаграмма openCV не учитывает 255 и он всегда 0, чтобы наша гистограмма была 1 в 1 делаем так
    counts[255] = 0;
    int maxVal = *std::max_element(counts, counts + bins);

    for(int bin = 0, x = 0; bin < bins; bin++, x++)
    {
        val = counts[bin];
        y = static_cast<int>(239 - val / static_cast<double>(maxVal) * 220);

        //qDebug() << "my " << "bin: " << bin << "val: " << val << "y: " << y << "maxvalue: " << maxVal;

        cvLine(dst, cvPoint(x, 239), cvPoint(x, y),
               cvScalar(0, 0, 0, 0), 1, 8, 0);
    }
}

int main(int argc, char *argv[])
{
    if (argc!=2) return 0;

    IplImage * source;
    IplImage * gray;
    IplImage * equalizeGray;
    IplImage * myEqualizeGray;
    IplImage * histImage;
    IplImage * myHistImage;
    IplImage * equalizehistImage;
    IplImage * myEqualizehistImage;

    source = cvLoadImage(argv[1]);
    gray = cvLoadImage(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    equalizeGray = cvLoadImage(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

    myEqualizeGray = cvCreateImage(cvSize(gray->width, gray->height),
    gray->depth, gray->nChannels);

    histImage = cvCreateImage(cvSize(256, 240), 8, 1);
    cvSet(histImage, cvScalar(255, 0, 0, 0), NULL);

    myHistImage = cvCreateImage(cvSize(256, 240), 8, 1);
    cvSet(myHistImage, cvScalar(255, 0, 0, 0), NULL);

    equalizehistImage = cvCreateImage(cvSize(256, 240), 8, 1);
    cvSet(equalizehistImage, cvScalar(255, 0, 0, 0), NULL);

    myEqualizehistImage = cvCreateImage(cvSize(256, 240), 8, 1);
    cvSet(myEqualizehistImage, cvScalar(255, 0, 0, 0), NULL);


    //функции
    opencvHistogram(gray, histImage);
    histogram(gray, myHistImage);

    cvEqualizeHist(gray, equalizeGray);
    opencvHistogram(equalizeGray, equalizehistImage);

    equalizeHistogram(gray, myEqualizeGray);
    histogram(myEqualizeGray, myEqualizehistImage);

    cvNamedWindow("Source Image", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Gray Image", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("OpenCV Equalize Gray Image", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Equalize Gray Image", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("OpenCV Histogram", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Histogram", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("OpenCV Equalize Histogram", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Equalize Histogram", CV_WINDOW_AUTOSIZE);

    cvShowImage("Source Image", source);
    cvShowImage("Gray Image", gray);
    cvShowImage("OpenCV Equalize Gray Image", equalizeGray);
    cvShowImage("Equalize Gray Image", myEqualizeGray);
    cvShowImage("OpenCV Histogram", histImage);
    cvShowImage("Histogram", myHistImage);
    cvShowImage("OpenCV Equalize Histogram", equalizehistImage);
    cvShowImage("Equalize Histogram", myEqualizehistImage);

    cvWaitKey(0);
    cvReleaseImage(&source);
    cvReleaseImage(&gray);
    cvReleaseImage(&equalizeGray);
    cvReleaseImage(&myEqualizeGray);
    cvReleaseImage(&histImage);
    cvReleaseImage(&myHistImage);
    cvReleaseImage(&equalizehistImage);
    cvReleaseImage(&myEqualizehistImage);
    return 0;
}
