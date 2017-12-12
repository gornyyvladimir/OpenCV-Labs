#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <QDebug>
#include <iostream>

IplImage * to8U(IplImage *src)
{
    IplImage *dst = cvCreateImage(cvSize(src->width, src->height),IPL_DEPTH_8U, 1);
    for(int col = 0; col < src->width; col++) {
        for(int row = 0; row < src->height; row++) {
            cvSetReal2D(dst, row, col, static_cast<unsigned char>(cvGet2D(src, row, col).val[0]));
        }
    }
    return dst;
}

std::complex<double> toComplex(CvScalar scalar)
{
    return std::complex<double>(scalar.val[0], scalar.val[1]);
}

CvScalar fromComplex(std::complex<double> scalar) {
    return cvScalar(scalar.real(), scalar.imag());
}

std::vector<std::complex<double>> DCT(const std::vector<std::complex<double>> &src)
{
    std::vector<double> buf(src.size());
    std::vector<std::complex<double>> dst(src.size());
    size_t N = src.size();

    for(size_t k = 0; k < N; k++){
        buf[k] = 0;
        for(uint n = 0; n < N; n++)
            buf[k] += src[n].real() * cos(CV_PI / static_cast<double>(N) * (static_cast<double>(n) + 0.5) * static_cast<double>(k));
    }

    dst[0] = buf[0] * sqrt(1.0 / static_cast<double>(N));
    for(uint i = 1; i < N; i++)
        dst[i] = std::complex<double>(buf[i] * sqrt(2.0 / static_cast<double>(N)), 0);

    return dst;
}

std::vector<std::complex<double>> IDCT(const std::vector<std::complex<double>> &src)
{
    size_t N = src.size();
    std::vector<double> buf(src.size());
    std::vector<std::complex<double>> dst(src.size());

    for(uint k = 0; k < N; k++){
        buf[k] = sqrt(0.5) * src[0].real();
        for(uint n = 1; n < N; n++)
            buf[k] += src[n].real() * cos(CV_PI / static_cast<double>(N) * static_cast<double>(n) * (static_cast<double>(k) + 0.5));
    }

    for(uint i = 0; i < N; i++)
        dst[i] = std::complex<double>(buf[i] * sqrt(2.0 / static_cast<double>(N)), 0);

    return dst;
}

std::vector<std::complex<double>> DFT(const std::vector<std::complex<double>> &data, int dir)
{
    unsigned long i,k;
    double arg;
    double cosarg,sinarg;
    std::vector<std::complex<double>> output(data.size());
    unsigned long m = data.size();

    for (i = 0; i < m; i++) {
        output[i] = 0;
        arg = - dir * 2.0 * 3.141592654 * static_cast<double>(i) / static_cast<double>(m);
        for (k = 0; k < m; k++) {
            cosarg = cos(k * arg);
            sinarg = sin(k * arg);
            output[i] = std::complex<double> (
                        output[i].real() + (data[k].real() * cosarg - data[k].imag() * sinarg),
                        output[i].imag() + (data[k].real() * sinarg + data[k].imag() * cosarg)
                        );

        }
    }


    /* Copy the data back */
    if (dir == 1) {
        for (i=0;i<m;i++) {
            output[i] /= static_cast<double>(m);
        }
    }

    return output;
}


std::vector<std::complex<double>> FFT(const std::vector<std::complex<double>> & data, int dir)
{
    unsigned long i, i1, i2,j, k, l, l1, l2, n;
    std::complex<double> tx, t1, u, c;
    std::vector<std::complex<double>> output(data.size());
    std::copy(data.begin(), data.end(), output.begin());

    /*Calculate the number of points */
    double m = log2(data.size());
    n = 1;
    for(i = 0; i < m; i++)
        n <<= 1;

    /* Do the bit reversal */
    i2 = n >> 1;
    j = 0;

    for (i = 0; i < n-1 ; i++)
    {
        if (i < j)
            swap(output[i], output[j]);
        k = i2;

        while (k <= j)
        {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    /* Compute the FFT */
    c.real(-1.0);
    c.imag(0.0);
    l2 = 1;
    for (l = 0; l < m; l++)
    {
        l1 = l2;
        l2 <<= 1;
        u.real(1.0);
        u.imag(0.0);

        for (j = 0; j < l1; j++)
        {
            for (i = j; i < n; i += l2)
            {
                i1 = i + l1;
                t1 = u * output[i1];
                output[i1] = output[i] - t1;
                output[i] += t1;
            }

            u = u * c;
        }

        c.imag(sqrt((1.0 - c.real()) / 2.0));
        if (dir == 1)
            c.imag(-c.imag());
        c.real(sqrt((1.0 + c.real()) / 2.0));
    }

    /* Scaling for forward transform */
    if (dir == 1)
    {
        for (i = 0; i < n; i++)
            output[i] /= n;
    }
    return output;
}

IplImage * transform(IplImage *source, int direction, bool fast, bool dct)
{

    int rows = source->height;
    int cols = source->width;

    IplImage *dst;
    double val_real;

    if (source->nChannels==2 && source->depth == IPL_DEPTH_64F) {
        qDebug() << " 2 channels ";
        dst = static_cast<IplImage *>(cvClone(source));
    } else {

        rows = fast ? static_cast<int>(pow(2, ceil(log(source->height)/log(2)))) : cvGetOptimalDFTSize(source->height);
        cols = fast ? static_cast<int>(pow(2, ceil(log(source->width)/log(2)))) : cvGetOptimalDFTSize(source->width);

        dst = cvCreateImage(cvSize(cols, rows),IPL_DEPTH_64F, 2);
        for(int col = 0; col < cols; col++) {
            for(int row = 0; row < rows; row++) {
                if(row < source->height)
                {
                    if(col < source->width) {
                        val_real = cvGetReal2D(source, row, col);
                    } else {
                        val_real = 0.0;
                    }
                }
                cvSet2D(dst, row, col, cvScalar(val_real, 0.0));
            }
        }
    }

    qDebug() << " rows: " << rows << " cols: " << cols;
    IplImage *chk = cvCreateImage(cvSize(cols, rows),IPL_DEPTH_64F, 2);



    if (direction==1) {
        cvDFT(dst, chk, CV_DXT_FORWARD, source->height);
    } else {
        cvDFT(dst, chk, CV_DXT_INVERSE_SCALE, source->height);
    }
    qDebug() << "dst filled" ;

    for(int col = 0; col < cols; col++) {
        std::vector<std::complex<double>> subarray;
        for(int row = 0; row < rows; row++) {
            subarray.push_back(toComplex(cvGet2D(dst,row,col)));
        }

        std::vector<std::complex<double>> result;

        qDebug() << "started col: " << col;

        if (dct && direction==1) {
            result = DCT(subarray);
        } else if (dct) {
            result = IDCT(subarray);
        } else if (fast) {
            result = FFT(subarray, direction);
        } else {
            result = DFT(subarray, direction);
        }

        qDebug() << "completed col: ";

        for(int row = 0; row < rows; row++) {
            cvSet2D(dst, row, col,fromComplex(result[static_cast<unsigned long>(row)]));
        }

        qDebug() << "filled col: " << col;
    }

    qDebug() << "cols transformed";

    for(int row = 0; row < rows; row++) {
        std::vector<std::complex<double>> subarray;
        for(int col = 0; col < cols; col++) {
            subarray.push_back(toComplex(cvGet2D(dst,row,col)));
        }

        std::vector<std::complex<double>> result;
        if (dct && direction==1) {
            result = DCT(subarray);
        } else if (dct) {
            result = IDCT(subarray);
        } else if (fast) {
            result = FFT(subarray, direction);
        } else {
            result = DFT(subarray, direction);
        }

        for(int col = 0; col < cols; col++) {
            cvSet2D(dst, row, col,fromComplex(result[static_cast<unsigned long>(col)]));
        }
    }

    qDebug() << "rows transformed";
    return dst;
}

int main(int argc, char ** argv)
{
    IplImage * src;
    IplImage * complex;
    IplImage * dft;
    IplImage * idft;
    IplImage * idftImg;
    IplImage * dst;
    IplImage * myDft;
    IplImage * myDct;
    IplImage * myIdft;
    IplImage * myIdct;

    int rows;
    int cols;
    int row;
    int col;
    double val_real;
    CvScalar val_complex;
    src = cvLoadImage(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    rows = cvGetOptimalDFTSize(src->height);
    cols = cvGetOptimalDFTSize(src->width);

    complex = cvCreateImage(cvSize(rows, cols), IPL_DEPTH_64F, 2);

    idftImg = cvCreateImage(
                cvSize(src->width, src->height), IPL_DEPTH_8U, 1);

    for(row = 0; row < rows; row++)
        for(col = 0; col < cols; col++)
        {
            if(row < src->height)
            {
                if(col < src->width)
                    val_real = cvGetReal2D(src, row, col);
                else
                    val_real = 0.0;
            }
            else
                val_real = 0.0;
            cvSet2D(complex, row, col, cvScalar(val_real, 0.0));
        }
    dft = cvCreateImage(cvSize(cols, rows), IPL_DEPTH_64F, 2);
    idft = cvCreateImage(cvSize(cols, rows), IPL_DEPTH_64F, 2);
    dst = cvCreateImage(cvSize(src->width, src->height),
                        src->depth, 1);
    cvDFT(complex, dft, CV_DXT_FORWARD, src->height);
    cvDFT(dft, idft, CV_DXT_INVERSE_SCALE, src->height);

    for(row = 0; row < dst->height; row++)
        for(col = 0; col < dst->width; col++)
        {
            val_complex = cvGet2D(complex, row, col);
            cvSetReal2D(dst, row, col, val_complex.val[0]);
        }

    //OpenCv Idft
    idftImg = to8U(idft);

    //Фурье
    myDft = transform(src, 1, true, false);
    myIdft = transform(myDft, -1, true, false);
    myDct = transform(src, 1, false, true);
    myIdct = transform(myDct, -1, false, true);


    cvNamedWindow("Source", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("OpenCV Dst", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("OpenCV Idft", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Idft", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Idct", CV_WINDOW_AUTOSIZE);

    cvShowImage("Source", src);
    cvShowImage("OpenCV Dst", dst);
    cvShowImage("OpenCV Idft", idftImg);
    cvShowImage("Idft", to8U(myIdft));
    cvShowImage("Idct", to8U(myIdct));

    cvWaitKey(0);
    cvReleaseImage(&src);
    cvReleaseImage(&dft);
    cvReleaseImage(&idft);
    cvReleaseImage(&idftImg);
    cvReleaseImage(&complex);
    cvReleaseImage(&dst);
    cvReleaseImage(&myIdft);
    cvReleaseImage(&myIdct);
    cvReleaseImage(&myDft);
    cvReleaseImage(&myDct);
    cvDestroyAllWindows();
    return 0;
}
