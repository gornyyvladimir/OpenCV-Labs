#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <QDebug>
#include <QMap>
#include <QVector>

// Бинаризация методом скользящего среднего
// N - количество усредняемых точек
void adaptiveTreshold(IplImage * src, IplImage * dst, double N=30)
{
    double n = 1;
    double avg = cvGetReal2D(src, 0, 0);
    for(int y=0; y<src->height; y++)
    {
        n=1;
        for (int x=0; x<src->width; x++)
        {
            double a = cvGetReal2D(src, y, x);
            avg = avg + (a - avg)/(n + 1);
            if(n < N)
            {
                n++;
            }
            if(cvGetReal2D(src, y, x) > avg)
            {
                cvSetReal2D(dst, y, x, 255.0);
            }
        }
    }
}

//бинарное изображение
void threshold(IplImage * src, IplImage * dst, double delta)
{
    double pixel;

    for(int y = 0; y < src->height; y++)
         for(int x = 0; x < src->width; x++)
         {
              pixel = cvGetReal2D(src, y, x);
              //Сравниваем текущий пиксель с дельтой, если больше то красим белым, если меньше черным
              pixel = pixel >= delta ? 255.0 : 0.0;
              cvSetReal2D(dst, y, x, pixel);
         }
}

void otsuThreshold(IplImage * src, IplImage * dst)
{
    double pixel;
    double N = src->width * src->height; //общее количестов пикселей

    //количество пикселей одной яркости
    QMap<double, int> counter;

    QVector<double> pi;
    QVector<double> pk;
    QVector<double> mk;
    QVector<double> o;

    //Вектор в котором хранятся К, если есть несколько максимумов
    QVector<double> K;


    //Заполнить нулями Map
    for(double i=0.0; i<256.0; i++)
    {
        counter[i] = 0;
    }

    for(int y = 0; y < src->height; y++)
         for(int x = 0; x < src->width; x++)
         {
              pixel = cvGetReal2D(src, y, x);
              //считаем количество пикселей одной яркости
              counter[pixel]++;
         }

    //qDebug() << counter;

    //it - итератор
    QMapIterator<double, int> counterIt(counter);
    while(counterIt.hasNext())
    {
        counterIt.next();
        //где n i – число пикселов с яркостью i. pi=ni/N;
        pi << counterIt.value()/N;
    }

    //Вычислить накопленные суммы P1(k)
    double piSum = 0.0;
    //Вычислить накопленные суммы m(k)
    double mkSum = 0.0;
    //Вычислить глобальную среднюю яркость mG
    double mG = 0.0;

    //тут можнро прописать цикл for от 0 до 255, просто в векторе pi, 255 значений по нему и бежим
    int i = 0;
    QVectorIterator<double> piIt(pi);
    while(piIt.hasNext())
    {
        double Pi = piIt.next();
        //суммировать pi
        piSum += Pi;
        //положить в pk
        pk << piSum;

        //суммировать i*pi
        mkSum += i*Pi;
        //положить в mk;
        mk << mkSum;

        //увеличить i
        i++;
    }

    //Вычислить глобальную среднюю яркость mG, в конце итераций mkSum это сумма всех pi
    mG = mkSum;
    qDebug() << "mG: " << mG;

    //Вычислить межклассовую дисперсию
    for(int k = 0; k<pk.length(); k++)
    {
        //Вычисляем дисперсию

        double a = pow((mG * pk.at(k) - mk.at(k)),2.0);
        double b = (pk.at(k)*(1-pk.at(k))) <= 0.0 ? 0.0 : (pk.at(k)*(1-pk.at(k)));

        //oK = pow((mG * pk.at(k) - mk.at(k)),2.0)/(pk.at(k)*(1-pk.at(k)));

        double oK = 0.0;

        if(b != 0.0)
            oK = a/b;

        //кладем в вектор
        o << oK;
    }

    qDebug() << o;

    //Определить пороговое значение Δ как такое значение k, при котором
    //величина межклассовой дисперсии максимальна
    //можно обойтись и нахождением максимума, код ниже мало когда нужен
    double max = *std::max_element(o.begin(), o.end());

    int count = 0;
    int kSum = 0;
    int index = 0;

    do
    {
        index = o.indexOf(max);
        kSum += index;
        count++;

        //делаем шаг вперед чтобы посмотреть вдруг еще есть такие же максимумы
        index++;
    }
    while(o.indexOf(max, index) > 0);

    //Если максимум неоднозначен, то выбрать в качестве Δ среднее значение k по всем найденным максимумам.
    double delta = kSum/count;
    qDebug() << "kSum: " << kSum;
    qDebug() << "delta: " << delta;

    //вызвать функцию бинаризации с нашей дельтой
    threshold(src, dst, delta);
}

//последовательное уточнение
void graduallyThreshold(IplImage * src, IplImage * dst)
{
    //128 - начальная дельта
    double delta = 128.0, tmpdelta;

    do {

        tmpdelta = delta;

        //сумма значений пикселов
        double m1sum = 0;
        //количество пикселов
        int m1cnt = 0;

        double m2sum = 0;
        int m2cnt = 0;

        for(int y = 0; y < src->height; y++)
            for(int x = 0; x < src->width; x++)
            {
                double pixel = cvGetReal2D(src, y, x);
                if (pixel>delta) {
                    m1sum = m1sum+pixel;
                    m1cnt++;
                } else {
                    m2sum = m2sum+pixel;
                    m2cnt++;
                }
            }

        double a, b;
        //mNsum/mNcnt получить средннее значение пикселов по области, делим их сумму на количество
        //если знаменатель = 0, то выражение = 0
        a = m1cnt == 0 ? 0 : m1sum/m1cnt;
        b = m2cnt == 0 ? 0 : m2sum/m2cnt;

        delta = (a + b) * 0.5;
        //qDebug() << delta;
    }
    //10 это погрешность
    while (delta - tmpdelta > 10);

    //qDebug() << delta;

    //вызвать функцию бинаризации с нашей дельтой
    threshold(src, dst, delta);
}

int main(int argc, char *argv[])
{
    if (argc!=2) return 0;

    IplImage * source;
    IplImage * gray;
    IplImage * binary;
    IplImage * binaryOtsu;
    IplImage * adaptive;
    IplImage * myBinary;
    IplImage * myGraduallyBinary;
    IplImage * myOtsuBinary;
    IplImage * myAdaptiveBinary;

    source = cvLoadImage(argv[1]);
    gray = cvLoadImage(argv[1], CV_LOAD_IMAGE_GRAYSCALE);


    binary = cvCreateImage(cvSize(gray->width, gray->height),
    gray->depth, gray->nChannels);
    binaryOtsu = cvCreateImage(cvSize(gray->width, gray->height),
    gray->depth, gray->nChannels);
    adaptive = cvCreateImage(cvSize(gray->width, gray->height),
    gray->depth, gray->nChannels);

    myBinary = cvCreateImage(cvSize(gray->width, gray->height),
    gray->depth, gray->nChannels);
    myGraduallyBinary = cvCreateImage(cvSize(gray->width, gray->height),
    gray->depth, gray->nChannels);
    myOtsuBinary = cvCreateImage(cvSize(gray->width, gray->height),
    gray->depth, gray->nChannels);
    myAdaptiveBinary = cvCreateImage(cvSize(gray->width, gray->height),
    gray->depth, gray->nChannels);

    //Функции OpenCV
    cvThreshold(gray, binaryOtsu, 0, 255, CV_THRESH_BINARY |
    CV_THRESH_OTSU);
    cvThreshold(gray, binary, 128.0, 255, CV_THRESH_BINARY);
    cvAdaptiveThreshold(gray, adaptive, 255,
    CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 3, 5);

    //Мои функции
    threshold(gray, myBinary, 128.0);
    graduallyThreshold(gray, myGraduallyBinary);
    otsuThreshold(gray, myOtsuBinary);
    adaptiveTreshold(gray, myAdaptiveBinary, 3);

    cvNamedWindow("Source Image", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Gray Image", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Binary Image", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Binary Otsu Image", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Adaptive Binary Image", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("My Binary Image", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("My Gradually Binary Image", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("My Otsu Binary Image", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("My Adaptive Binary Image", CV_WINDOW_AUTOSIZE);


    cvShowImage("Source Image", source);
    cvShowImage("Gray Image", gray);
    cvShowImage("Binary Image", binary);
    cvShowImage("Binary Otsu Image", binaryOtsu);
    cvShowImage("Adaptive Binary Image", adaptive);
    cvShowImage("My Binary Image", myBinary);
    cvShowImage("My Gradually Binary Image", myGraduallyBinary);
    cvShowImage("My Otsu Binary Image", myOtsuBinary);
    cvShowImage("My Adaptive Binary Image", myAdaptiveBinary);

    cvWaitKey(0);
    cvReleaseImage(&source);
    cvReleaseImage(&gray);
    cvReleaseImage(&binary);
    cvReleaseImage(&binaryOtsu);
    cvReleaseImage(&adaptive);
    cvReleaseImage(&myBinary);
    cvReleaseImage(&myGraduallyBinary);
    cvReleaseImage(&myOtsuBinary);
    cvReleaseImage(&myAdaptiveBinary);
    cvDestroyAllWindows();
    return 0;
}
