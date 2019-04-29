//
// Created by lqx on 19-3-1.
//
#ifndef DETECTPARKINGSLOT_DETECTSLOT_H
#define DETECTPARKINGSLOT_DETECTSLOT_H

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
struct SurfHF
{
    int p0,p1,p2,p3;

    float w;

    SurfHF() : p0(0),p1(0),p2(0),p3(0),w(0) { }
};
struct ValueIndex
{
    float value;
    int index;
};
struct SlotPoint
{
    int x;
    int y;
};
bool comp(const ValueIndex &a,const ValueIndex &b);

class DetectParkingSlot
{
public:
    DetectParkingSlot();
    ~DetectParkingSlot();
    int detecSlot(Mat &srcImage,Rect &srcRoi,vector<SlotPoint> &parkingSlotPoint);

private:
    void rgbGray(Mat &srcImg,Mat &grayImg);  //图像灰度化

    int max_gray=0;

    int  HistFromroi(cv::Mat img);       //直方图均衡化

    cv::Mat gray_binnary(cv::Mat img,int gray_value);      //二值化

    int  ProjectYdirect(cv::Mat img) ;  //向Y方向投影
    //用于加速计算Frangi滤波，参考opencv中surf角点检测hessian矩阵的计算
    void windowSumCol(Mat &srcImage,Point &startLoction,int step,int windowWidth,
                      int windowHeight,vector<Point> &linePoint);//竖直方向滑窗
    int windowSumRow(Mat &srcImage,Point &startLoction,int step,int windowWidth,
                     int windowHeight,vector<Point> &linePoint);//水平方向滑窗
    void findPeaks(Mat &matData,float minPeakDistance,float minPeakHeight,vector<ValueIndex> &peaks);//寻找波峰

    Point2f getCrossPoint(Vec4f &lineA, Vec4f &lineB); //求两条直线的交点
};

#endif //DETECTPARKINGSLOT_DETECTSLOT_H
