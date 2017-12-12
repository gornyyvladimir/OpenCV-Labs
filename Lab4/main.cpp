#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <QVector>

//усредняющий фильтр
void averageBlur(IplImage *src, IplImage *dst, int n)
{
    for(int y = 0; y < src->height; y++)
         for(int x = 0; x < src->width; x++)
         {
            double sum = 0;
            int k = 0;
            //цикл чтобы идти окошоком
            for (int y0 = y; y0 < y + n && y0 < src->height; y0++) {
                for (int x0 = x; x0 < x+n && x0 < src->width; x0++) {
                    double pixel = cvGetReal2D(src, y0, x0);
                    sum+=pixel;
                    k++;
                }
            }

            double val = sum / k;
            cvSetReal2D(dst, y, x, val);
         }
}

//helpers
int min(int a1, int a2) {
    return a1 < a2 ? a1 : a2;
}

int max(int a1, int a2) {
    return a1 > a2 ? a1 : a2;
}

double distance(int x, int y, int i, int j)
{
    return double(sqrt(pow(x - i, 2) + pow(y - j, 2)));
}

//коэфициент маски гауссова
double gaussian(double x, double y, double sigma)
{
    return exp(-(pow(x, 2)+pow(y, 2))/(2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
}

double gaussian(double x, double sigma)
{
    return exp(-(pow(x, 2))/(2 * pow(sigma, 2))) / (2 * CV_PI * pow(sigma, 2));
}


double gaussiandifference(double x, double y, double sigma1, double sigma2)
{
    return ((exp(-1 * (pow(x, 2) + pow(y, 2))/(2*sigma1))/pow(sigma1, 2)) -
            (exp(-1 * (pow(x, 2) + pow(y, 2))/(2*sigma2))/pow(sigma2, 2))) /
            2 * CV_PI;
}

//Гаусов фильтр
void gaussianBlur(IplImage *src, IplImage *dst, int n)
{
    double sigma = 1;

    for(int y = 0; y < src->height; y++)
        for(int x = 0; x < src->width; x++)
        {
            double res = 0;

            //чтобы не выходить за край в начале
            int y0 = y - n >= 0 ? y - n : 0;
            for (; y0 < y + n + 1 && y0 < src->height; y0++)
            {
                //чтобы не выходить за край в начале
                int x0 = x - n + 1 >= 0 ? x - n : 0;
                for (; x0 < x + n && x0 < src->width; x0++)
                {
                    int xg  = min(src->width, max(0, x0));
                    int yg  = min(src->height, max(0, y0));
                    double val = cvGetReal2D(src, yg, xg);
                    double g = gaussian(xg - x,yg - y, sigma);
                    res = res + (val * g);
                }
            }
            cvSetReal2D(dst, y, x, res);
        }
}

//медианный фильтр
void medianBlur(IplImage *src, IplImage *dst, int diameter)
{
    for(int y = 0; y < src->height; y++)
        for(int x = 0; x < src->width; x++)
        {
            double pixel = cvGetReal2D(src, y, x);

                QVector<double> window;
                int n = 0;

                //чтобы не выйти за край изображения
                int i = y >= diameter ? -diameter : -y;
                int heightRange = src->height-1;
                int d1 = y + diameter <= heightRange ? diameter : heightRange - y;

                for (; i <= d1; i++) {
                    //чтобы не выйти за край изображения
                    int j = x >= diameter ? -diameter : -x;
                    int widthRange = src->width-1;
                    int d2 = x + diameter <= widthRange ? diameter : widthRange - x;

                    for (; j <= d2; j++) {
                        window << cvGetReal2D(src, y + i, x + j);
                        n++;
                    }
                }

                std::sort(window.begin(), window.end());
                pixel = window[n/2];
            //}
            cvSetReal2D(dst, y, x, pixel);
        }
}


//билатеральная фильтрация
//Применяется в нескольких местах, поэтому вынес в отдельную функцию
double bilateralFilter(IplImage *src, int x, int y, int diameter, double sigmaI, double sigmaS)
{
    double res = 0;
    double wP = 0;
    int neighbor_x = 0;
    int neighbor_y = 0;
    int half = diameter / 2;

    for(int i = 0; i < diameter; i++) {
        for(int j = 0; j < diameter; j++) {
            neighbor_y = y - (half - i);
            neighbor_x = x - (half - j);
            double pixel  = cvGetReal2D(src, y, x);
            double neighbor  = cvGetReal2D(src, neighbor_y, neighbor_x);
            double gi = gaussian(neighbor - pixel, sigmaI);
            double gs = gaussian(distance(x, y, neighbor_x, neighbor_y), sigmaS);
            double w = gi * gs;
            res = res + neighbor * w;
            wP = wP + w;
        }
    }
    res = res / wP;
    return res;
}


void bilateralBlur(IplImage *src, IplImage *dst, int diameter, double sigmaI, double sigmaS)
{
    int width = src->width;
    int height = src->height;

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            if (i > diameter*2 && j > diameter*2 && i < height-diameter*2 && j < width-diameter*2)
            {
                double res = bilateralFilter(src, j, i, diameter, sigmaI, sigmaS);
                cvSetReal2D(dst, i, j, res);
            } else {
                cvSetReal2D(dst, i, j, cvGetReal2D(src, i, j));
            }
        }
    }
}

//метод робертса
void roberts(IplImage *src, IplImage *dst)
{
    for(int y = 0; y < src->height; y++)
        for(int x = 0; x < src->width; x++)
        {
            double pc = cvGetReal2D(src, y, x);
            int heightRange = src->height-1;
            int widthRange = src->width-1;

            int y1 = y + 1 <= heightRange ? y + 1 : y;
            int x1 = x + 1 <= widthRange ? x + 1 : x;

            double py1 = cvGetReal2D(src, y1, x);
            double px1 = cvGetReal2D(src, y, x1);
            double pxy1 = cvGetReal2D(src, y1, x1);

            double g1 = pc - pxy1;
            double g2 = px1 - py1;
            cvSetReal2D(dst, y, x, sqrt((g1*g2+g2*g2)));
        }
}

//используется в нескольких местах
void applyMask(IplImage *src, IplImage *dst, int x, int y, QVector<QVector<int>> mask)
{
    double val = 0;
    for (int y0=y; y0<y+mask.size() && y0<src->height; y0++) {
        for (int x0=x; x0<x+mask.size() && x0<src->width; x0++) {
            int xg = min(src->width, max(0, x0));
            int yg = min(src->height, max(0, y0));

            int m = mask[yg-y][xg-x];
            double v = cvGetReal2D(src, yg, xg);
            val += m*v;
        }
    }
    cvSetReal2D(dst, y, x, val);
}

//перегрузка для double
void applyMask(IplImage *src, IplImage *dst, int x, int y, QVector<QVector<double>> mask)
{
    double val = 0;
    for (int y0=y; y0<y+mask.size() && y0<src->height; y0++) {
        for (int x0=x; x0<x+mask.size() && x0<src->width; x0++) {
            int xg = min(src->width, max(0, x0));
            int yg = min(src->height, max(0, y0));

            double m = mask[yg-y][xg-x];
            double v = cvGetReal2D(src, yg, xg);
            val += m*v;
        }
    }
    cvSetReal2D(dst, y, x, val);
}

//запуск метода собела
void sobel(IplImage *src, IplImage *dst)
{
    //это матрица такая, можно было сделать так int mask[3][3] = {{-1,0,1}, {-2,0,2}, {-1,0,1}}; но мы же модные
    QVector<QVector<int>> sobelMask;
    sobelMask.append(QVector<int>() << -1 << 0 << 1);
    sobelMask.append(QVector<int>() << -2 << 0 << 2);
    sobelMask.append(QVector<int>() << -1 << 0 << 1);

    for(int y = 0; y < src->height; y++)
        for(int x = 0; x < src->width; x++)
        {
            applyMask(src, dst, x, y, sobelMask);
        }
}

//запуск метода превитт
void previtt(IplImage *src, IplImage *dst)
{
    QVector<QVector<int>> previttMask;
    previttMask.append(QVector<int>() << -1 << 0 << 1);
    previttMask.append(QVector<int>() << -1 << 0 << 1);
    previttMask.append(QVector<int>() << -1 << 0 << 1);

    for(int y = 0; y < src->height; y++)
         for(int x = 0; x < src->width; x++)
         {
            applyMask(src, dst, x, y, previttMask);
         }
}

//запуск метода лапласиана
void laplasian(IplImage *src, IplImage *dst)
{
    QVector<QVector<int>> laplasianMask;
    laplasianMask.append(QVector<int>() << 0 << 1 << 0);
    laplasianMask.append(QVector<int>() << 1 << -4 << 1);
    laplasianMask.append(QVector<int>() << 0 << 1 << 0);

    for(int y = 0; y < src->height; y++)
         for(int x = 0; x < src->width; x++)
         {
            applyMask(src, dst, x, y, laplasianMask);
         }
}

//LoG фильтрация
void LoG(IplImage *src, IplImage *dst)
{
    QVector<QVector<int>> LoGMask;
    LoGMask.append(QVector<int>() << 0 << 2 << 2 << 2 << 0);
    LoGMask.append(QVector<int>() << 2 << 1 << -4 << 1 << 2);
    LoGMask.append(QVector<int>() << 2 << -4 << -20 << -4 << 2);
    LoGMask.append(QVector<int>() << 2 << 1 << -4 << 1 << 2);
    LoGMask.append(QVector<int>() << 0 << 2 << 2 << 2 << 0);

    int size = LoGMask.at(0).size();

    //получить сумму индексов
    int sum = ((size * (size-1))/2) * size;

    if (std::abs(sum - 1) > 0.01)
    {
        double coef = 1.0/static_cast<double>(sum);
        for (int y=0; y < size; y++)
            for (int x=0; x < size; x++) {
                LoGMask[y][x] /= coef;
            }
    }

    for(int y = 0; y < src->height; y++)
        for(int x = 0; x < src->width; x++)
        {
            applyMask(src, dst, x, y, LoGMask);
        }
}

//DoG фильтрация
void DoG(IplImage *src, IplImage *dst, double sigma1, double sigma2)
{
    QVector<QVector<double>> DoGMask;
    int size = 5;

    double sum = 0;
    for (int y=0; y<size; y++) {
        QVector<double> row;
        for (int x=0; x<size; x++) {
            double gd = gaussiandifference(x,y, sigma1, sigma2);
            row << gd;
            //DoGMask[y][x] = gaussiandifference(x,y, sigma1, sigma2);
            sum += gd;
        }
        DoGMask << row;
    }

    if (std::abs(sum - 1) > 0.01)
    {
        double coef = 1.0/static_cast<double>(sum);
        for (int y=0; y < size; y++)
            for (int x=0; x < size; x++) {
                DoGMask[y][x] /= coef;
            }
    }

    for(int y = 0; y < src->height; y++)
        for(int x = 0; x < src->width; x++)
        {
            applyMask(src, dst, x, y, DoGMask);
        }
}

//повышение резкости изображения
void sharpness(IplImage *src, IplImage *dst, int diameter, double sigmaI, double sigmaS, double a)
{
    int width = src->width;
    int height = src->height;

    for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
            double fn = cvGetReal2D(src, y, x);
            double f = cvGetReal2D(src, y, x);
            if (y>diameter*2 && x>diameter*2 && y<height-diameter*2 && x<width-diameter*2)
            {
                fn = bilateralFilter(src, x, y, diameter, sigmaI, sigmaS);
            }

            double g = a*f - fn;
            cvSetReal2D(dst, y, x, g);
        }
    }
}

int main(int argc, char ** argv)
{
    IplImage * source;
    IplImage * blur;
    IplImage * gaussian;
    IplImage * median;
    IplImage * bilateral;

    IplImage * myAverageBlur;
    IplImage * myGaussian;
    IplImage * myMedian;
    IplImage * myBilateral;
    IplImage * myRoberts;
    IplImage * mySobel;
    IplImage * myPrevitt;
    IplImage * myLaplasian;
    IplImage * mySharpness;
    IplImage * myLoG;
    IplImage * myDoG;

    source = cvLoadImage(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    blur = cvCreateImage(cvSize(source->width, source->height), IPL_DEPTH_8U, 1);
    gaussian = cvCreateImage(
                cvSize(source->width, source->height), IPL_DEPTH_8U, 1);
    median = cvCreateImage(
                cvSize(source->width, source->height), IPL_DEPTH_8U, 1);
    bilateral = cvCreateImage(
                cvSize(source->width, source->height), IPL_DEPTH_8U, 1);


    myAverageBlur = cvCreateImage(
                cvSize(source->width, source->height), IPL_DEPTH_8U, 1);
    myGaussian = cvCreateImage(
                cvSize(source->width, source->height), IPL_DEPTH_8U, 1);
    myMedian = cvCreateImage(
                cvSize(source->width, source->height), IPL_DEPTH_8U, 1);
    myBilateral = cvCreateImage(
                cvSize(source->width, source->height), IPL_DEPTH_8U, 1);
    myRoberts = cvCreateImage(
                cvSize(source->width, source->height), IPL_DEPTH_8U, 1);
    mySobel = cvCreateImage(
                cvSize(source->width, source->height), IPL_DEPTH_8U, 1);
    myPrevitt = cvCreateImage(
                cvSize(source->width, source->height), IPL_DEPTH_8U, 1);
    myLaplasian = cvCreateImage(
                cvSize(source->width, source->height), IPL_DEPTH_8U, 1);
    mySharpness = cvCreateImage(
                cvSize(source->width, source->height), IPL_DEPTH_8U, 1);
    myLoG = cvCreateImage(
                cvSize(source->width, source->height), IPL_DEPTH_8U, 1);
    myDoG = cvCreateImage(
                cvSize(source->width, source->height), IPL_DEPTH_8U, 1);

    cvSmooth(source, blur, CV_BLUR, 7);
    cvSmooth(source, gaussian, CV_GAUSSIAN, 7);
    cvSmooth(source, median, CV_MEDIAN, 7);
    cvSmooth(source, bilateral, CV_BILATERAL, 7);

    //функции
    averageBlur(source, myAverageBlur, 7);
    gaussianBlur(source, myGaussian, 7);
    medianBlur(source, myMedian, 7);
    bilateralBlur(source, myBilateral, 3, 1, 2);
    roberts(source, myRoberts);
    sobel(source, mySobel);
    previtt(source, myPrevitt);
    laplasian(source, myLaplasian);
    sharpness(source, mySharpness, 3, 1 ,2, 3);
    LoG(source, myLoG);
    DoG(source, myDoG, 1.0, 1.5);

    cvNamedWindow("Source", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("OpenCV Blur", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("OpenCV Gaussian", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("OpenCV Median", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("OpenCV Bilateral", CV_WINDOW_AUTOSIZE);

    cvNamedWindow("Average Blur", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Gaussian Blur", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Median Blur", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Bilateral Blur", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Roberts Gradient", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Sobel Mask", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Previtt Mask", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Laplasian Mask", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Increase Sharpness", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("LoG Filter", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("DoG Filter", CV_WINDOW_AUTOSIZE);

    cvShowImage("Source", source);
    cvShowImage("OpenCV Blur", blur);
    cvShowImage("OpenCV Gaussian", gaussian);
    cvShowImage("OpenCV Median", median);
    cvShowImage("OpenCV Bilateral", bilateral);

    cvShowImage("Average Blur", myAverageBlur);
    cvShowImage("Gaussian Blur", myGaussian);
    cvShowImage("Median Blur", myMedian);
    cvShowImage("Bilateral Blur", myBilateral);
    cvShowImage("Roberts Gradient", myRoberts);
    cvShowImage("Sobel Mask", mySobel);
    cvShowImage("Previtt Mask", myPrevitt);
    cvShowImage("Laplasian Mask", myLaplasian);
    cvShowImage("Increase Sharpness", mySharpness);
    cvShowImage("LoG Filter", myLoG);
    cvShowImage("DoG Filter", myDoG);

    cvWaitKey(0);
    cvReleaseImage(&source);
    cvReleaseImage(&blur);
    cvReleaseImage(&gaussian);
    cvReleaseImage(&median);
    cvReleaseImage(&bilateral);
    cvReleaseImage(&myAverageBlur);
    cvReleaseImage(&myGaussian);
    cvReleaseImage(&myMedian);
    cvReleaseImage(&myBilateral);
    cvReleaseImage(&myRoberts);
    cvReleaseImage(&mySobel);
    cvReleaseImage(&myPrevitt);
    cvReleaseImage(&myLaplasian);
    cvReleaseImage(&mySharpness);
    cvReleaseImage(&myLoG);
    cvReleaseImage(&myDoG);
    cvDestroyAllWindows();
    return 0;
}
