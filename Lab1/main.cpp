#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <QList>

struct PartStruct {
    double min;
    double max;
    double c;
    double b;
};

struct SawPart {
    double min;
    double max;
};

//линейное контрастирование
void linear(IplImage * gray, double c, double max)
{
    int x;
    int y;
    double pixel;
    IplImage * image;

    image = cvCreateImage(cvSize(gray->width, gray->height),
                                gray->depth, gray->nChannels);

    //бежим по изображению
    for(y = 0; y < image->height; y++)
         for(x = 0; x < image->width; x++)
         {
              pixel = cvGetReal2D(gray, y, x);

              pixel *= c;
              if(pixel > max) pixel = max;
              cvSetReal2D(image, y, x, pixel);
         }
    cvNamedWindow("Linear Image", CV_WINDOW_AUTOSIZE);
    cvShowImage("Linear Image", image);
}

//негатив
void negative(IplImage * gray)
{
    int x;
    int y;
    double pixel;
    IplImage * image;

    image = cvCreateImage(cvSize(gray->width, gray->height),
                                gray->depth, gray->nChannels);


    for(y = 0; y < image->height; y++)
         for(x = 0; x < image->width; x++)
         {
              pixel = cvGetReal2D(gray, y, x);
              //вычитаем яркость пикселя на котором стоим
              pixel = 255.0 - pixel;
              cvSetReal2D(image, y, x, pixel);
         }
    cvNamedWindow("Negative Image", CV_WINDOW_AUTOSIZE);
    cvShowImage("Negative Image", image);
}

//логарифмическое преобразование
void logimage(IplImage * gray, double c) {
    int x;
    int y;
    double pixel;
    IplImage * image;

    image = cvCreateImage(cvSize(gray->width, gray->height),
                                gray->depth, gray->nChannels);


    for(y = 0; y < image->height; y++)
         for(x = 0; x < image->width; x++)
         {
              pixel = cvGetReal2D(gray, y, x);
              pixel = c * log(1+pixel);
              cvSetReal2D(image, y, x, pixel);
         }
    cvNamedWindow("Log Image", CV_WINDOW_AUTOSIZE);
    cvShowImage("Log Image", image);
}

//степенное преобразование
void powimage(IplImage * gray, double c, double yi) {
    int x;
    int y;
    double pixel;
    IplImage * image;

    image = cvCreateImage(cvSize(gray->width, gray->height),
                                gray->depth, gray->nChannels);


    for(y = 0; y < image->height; y++)
         for(x = 0; x < image->width; x++)
         {
              pixel = cvGetReal2D(gray, y, x);
              pixel = c * pow(pixel, yi);
              cvSetReal2D(image, y, x, pixel);
         }
    cvNamedWindow("Pow Image", CV_WINDOW_AUTOSIZE);
    cvShowImage("Pow Image", image);
}

//кусочно-линейное преобразование
void partlinear(IplImage * gray, QList<PartStruct> parts) {
    int x;
    int y;
    double pixel;
    IplImage * image;

    image = cvCreateImage(cvSize(gray->width, gray->height),
                                gray->depth, gray->nChannels);


    for(y = 0; y < image->height; y++)
         for(x = 0; x < image->width; x++)
         {
              pixel = cvGetReal2D(gray, y, x);
              for (int i=0; i<parts.length(); i++) {
                  PartStruct part = parts[i];
                  if (pixel>=part.min && pixel<part.max)
                  {
                      pixel = part.c * pixel + part.b * pixel;
                      break;
                  }
              }

              cvSetReal2D(image, y, x, pixel);
         }
    cvNamedWindow("Part Linear Image", CV_WINDOW_AUTOSIZE);
    cvShowImage("Part Linear Image", image);
}

//пороговая обработка
void borderimage(IplImage * gray, double min, double max, const char *name) {
    int x;
    int y;
    double pixel;
    IplImage * image;

    image = cvCreateImage(cvSize(gray->width, gray->height),
                                gray->depth, gray->nChannels);




    for(y = 0; y < image->height; y++)
         for(x = 0; x < image->width; x++)
         {
              pixel = cvGetReal2D(gray, y, x);
              if (pixel >= min && pixel <max) {
                  pixel = 255;
              } else {
                  pixel = 0;
              }
              cvSetReal2D(image, y, x, pixel);
         }
    cvNamedWindow(name, CV_WINDOW_AUTOSIZE);
    cvShowImage(name, image);
}

//контрастное масштабирование
void contrastNormalize(IplImage * gray, double min, double max) {
    int x;
    int y;
    double pixel;
    IplImage * image;

    image = cvCreateImage(cvSize(gray->width, gray->height),
                                gray->depth, gray->nChannels);


    double range = (max-min);

    for(y = 0; y < image->height; y++)
         for(x = 0; x < image->width; x++)
         {
             pixel = cvGetReal2D(gray, y, x);
             if (pixel >= min && pixel <=max)
             {
                 pixel = (((pixel - min) / range) * 299); //че за 299 может max?
             } else {
                 pixel = 0;
             }
             cvSetReal2D(image, y, x, pixel);
         }
    cvNamedWindow("Contrast normalization", CV_WINDOW_AUTOSIZE);
    cvShowImage("Contrast normalization", image);
}

//пилообразное контрастное масштабирование
void contrastSawNormalize(IplImage * gray, QList<SawPart> parts) {
    int x;
    int y;
    double pixel;
    IplImage * image;

    image = cvCreateImage(cvSize(gray->width, gray->height),
                                gray->depth, gray->nChannels);




    for(y = 0; y < image->height; y++)
         for(x = 0; x < image->width; x++)
         {
              bool setted = false;
              pixel = cvGetReal2D(gray, y, x);
              for (int i=0; i<parts.length(); i++) {
                  if (pixel >= parts[i].min && pixel < parts[i].max) {
                      double range = parts[i].max - parts[i].min;
                      setted = true;
                      pixel = (((pixel - parts[i].min) / range) * 299);; //че за 299?
                      break;
                  }
              }
              if (!setted) {
                    pixel = 0;
              }

              cvSetReal2D(image, y, x, pixel);
         }
    cvNamedWindow("Contrast saw normalization", CV_WINDOW_AUTOSIZE);
    cvShowImage("Contrast saw normalization", image);
}

//Гистограмма
void histogram(IplImage * sourceImage)
{
    IplImage * histImage;
    CvHistogram * hist;
    int bins = 256;
    float ranges[] = {0, 255};
    float * histRanges[] = {ranges};
    float maxVal=0;
    int bin;
    double val;
    int x;
    int y;
    histImage = cvCreateImage(cvSize(256, 240), 8, 1);
    cvSet(histImage, cvScalar(255, 0, 0, 0), NULL);
    hist = cvCreateHist(1, &bins, CV_HIST_ARRAY, histRanges, 1);
    cvCalcHist(&sourceImage, hist, 0, NULL);
    cvGetMinMaxHistValue(hist, NULL, &maxVal, NULL, NULL);
    for(bin = 0, x = 0; bin < bins; bin++, x++)
    {
        val = cvGetReal1D(hist->bins, bin);
        y = static_cast<int>(239 - val / static_cast<double>(maxVal) * 220);
        cvLine(histImage, cvPoint(x, 239), cvPoint(x, y),
               cvScalar(0, 0, 0, 0), 1, 8, 0);
    }
    cvNamedWindow("Histogram", CV_WINDOW_AUTOSIZE);
    cvShowImage("Histogram", histImage);
}

int main(int argc, char* argv[])
{
    if (argc!=2) return 0;
    IplImage * source;
    IplImage * gray;
    source = cvLoadImage(argv[1]);
    gray = cvLoadImage(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

    cvNamedWindow("Source Image", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Gray Image", CV_WINDOW_AUTOSIZE);
    cvShowImage("Source Image", source);
    cvShowImage("Gray Image", gray);

    QList<PartStruct> parts({{10.0, 20.0,0.0,128.0},{2.0, 3.0,128.0,255.0}});
    QList<SawPart> sawParts({{20.0, 50.0}, {100.0, 150.0}});

    linear(gray, 3.5, 255.0);
    negative(gray);
    logimage(gray, 50.0);
    partlinear(gray, parts);
    borderimage(gray, 120, 255, "Border Image");
    borderimage(gray, 120, 140, "Border bright part Image");
    contrastNormalize(gray, 120, 299); //че за 299 может 255?
    contrastSawNormalize(gray, sawParts);

    histogram(gray);

    cvWaitKey(0);
    return 0;
}
