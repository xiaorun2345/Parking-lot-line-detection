//
// Created by lqx on 19-3-4.
//

#include <iostream>
#include "detectslot.h"
/*
 * 用于sort()函数中降序排列
 */
bool comp(const ValueIndex &a,const ValueIndex &b)
{
    return a.value>b.value;
}
bool comp1(const Point2f &a,const Point2f &b)
{
    return a.y<b.y;
}
DetectParkingSlot::DetectParkingSlot() {

}
DetectParkingSlot::~DetectParkingSlot() {

}
//4、直方图统计
int  DetectParkingSlot::HistFromroi(cv::Mat img)
{
    Mat hist_eq(img.rows,img.cols,CV_8UC1);
    int sum_pix[256]={0};
    int gray_sum=0;
    int temp_gray[256];
    int all_pix=img.rows*img.cols;

    float n_per[256]={0.0};

    for(int i=0;i<=img.rows;i++)
    {
        for (int j = 0; j <=img.cols ; ++j)
        {
            sum_pix[img.at<uchar>(i,j)]++;

        }

    }


    for (int k = 0; k <256 ; ++k)
    {
        gray_sum+=sum_pix[k];

        temp_gray[k]=gray_sum;
    }

    //for (int l = 1; l <256 ; ++l)
    // {
    //    n_per[l]=n_per[l]+n_per[l-1];
    //}
    int maxnum=0;
    //int max_gray=0;

    //20步滤波,获取最大差分值,找到边缘最大值位置的灰度值
    for (int a = 20; a <256 ; ++a)
    {
        if(maxnum<temp_gray[a]-temp_gray[a-20])
        {
            maxnum=temp_gray[a]-temp_gray[a-20];
            max_gray=a-10;
        }

    }

    //imshow("eqhist",hist_eq);
    return max_gray;

}
cv::Mat DetectParkingSlot::gray_binnary(cv::Mat img,int gray_value)
{
    int tempgray=0;
    Mat dst_img(img.rows,img.cols,CV_8UC1);
    for (int i = 0; i <img.rows ; ++i)
    {
        for (int j = 0; j <img.cols ; ++j)
        {
            tempgray=img.at<uchar> (i,j);
            if (max_gray>tempgray)
            {
                dst_img.at<uchar>(i,j)=0;
            }
            else
                dst_img.at<uchar>(i,j)=tempgray-max_gray;
            //dst_img.at<uchar>(i,j)=255;
        }

    }
    imshow("binnary",dst_img);
    return dst_img;
}

int  DetectParkingSlot::ProjectYdirect(cv::Mat img)
{
    int temp[img.cols];
    int maxpos;
    //向Y轴投影,获取最大位置
    for (int i = 0; i <img.cols; ++i)
    {   int sumY=0;
        for (int j = 0; j <img.rows ; ++j)
        {
            sumY+=img.at<uchar>(j,i);

        }
        //最大位置像素列平均
        temp[i]=sumY/img.rows;

    }
    int tempgray=0;
    int temp_sum=0;
    int temp_sum_right=0;
    //int tempgray_right=0;
    int maxgray=0;
    int maxgray_right=0;
    int max_index=0;
    int max_index_right=0;

    //左半边图像获取最大位置与像素平均值
    for (int k = 5; k <img.cols-1; ++k)
    {
        tempgray=(temp[k-1]+temp[k]+temp[k+1])/3;
        if (tempgray>maxgray)
        {
            maxgray = tempgray;
            max_index = k;

        }
    }

    if(maxgray<20)
    {
        //cout<<"no deteced"<<endl;

        maxgray=0;

        return maxgray;

    }

    maxpos=max_index;
    return maxpos;

}


/*
 *寻找波峰
 * matData：输入的一列数据
 * minPeakDistance：峰值最小值
 * minPeakHeight：峰值之间的最小距离
 * peaks：峰值的位置和大小
 */

void DetectParkingSlot::findPeaks(Mat &matData,float minPeakDistance,float minPeakHeight,vector<ValueIndex> &peaks)
{
    int row = matData.rows;
    int col = matData.cols;
    vector<int> Sign;
    float diff;
    for (int i = 1; i < row; i++)
    {
        /*相邻值做差：
        *小于0，赋-1
        *大于0，赋1
        *等于0，赋0
        */
        diff = matData.at<float>(i,0)-matData.at<float>(i-1,0);
        if (diff > 0)
        {
            Sign.push_back(1);
        }
        else if (diff < 0)
        {
            Sign.push_back(-1);
        }
        else
        {
            Sign.push_back(0);
        }
    }

    //再对Sign相邻位做差
    //保存极大值
    ValueIndex temp;
    for (int j = 1; j < Sign.size(); j++)
    {
        int diff = Sign[j] - Sign[j - 1];
        if (diff < 0)
        {
            if (matData.at<float>(j,0)>minPeakHeight)
            {
                //根据峰值最小高度进行筛选
                temp.index=j;
                temp.value=matData.at<float>(j,0);
                peaks.push_back(temp);
            }
        }
    }

    if(minPeakDistance>0)
    {
        int i = 1;
        while(i<peaks.size())
        {
            int sub = peaks[i].index - peaks[i-1].index;
            if(sub < minPeakDistance)
            {
                peaks.erase(peaks.begin()+i);
            }
            else i ++;
        }

    }
}

/*
 * 图像灰度化函数
 */
void  DetectParkingSlot::rgbGray(Mat &srcImg,Mat &grayImg)
{
    //Mat gray_img(img.rows,img.cols,CV_8UC1);
    //cvtColor(img, img,COLOR_BGR2GRAY);
    if(srcImg.empty())
    {
        cout<<"can not load image";
    }

    if (srcImg.channels()==1)
    {
        cout<<"please load a RGB image"<<endl;
    }
   for (int i = 0; i < srcImg.rows; i++)
    {
        for (int j = 0; j < srcImg.cols; j++)
        {
            if (srcImg.channels()==3)
            {
                //uchar b = img.at<Vec3b>(i, j)[0];

                uchar g = srcImg.at<Vec3b>(i, j)[1];

                uchar r = srcImg.at<Vec3b>(i, j)[2];

                grayImg.at<uchar>(i, j) =0.5 * g + 0.5 * r;


            }

            if (srcImg.channels()==1)
            {

                grayImg.at<uchar>(i,j)=grayImg.at<uchar>(i,j);

            }
        }
    }
    imwrite("grayImg.jpg",grayImg);
}
/*
 * srcImage:经预处理后的灰度图像，数据类型为32F
 * startLoction：为全局的按列求和后，强度最大值的位置
 * step 滑窗步长
 * windowWidth 滑窗的宽
 * windowHeight 滑窗的高
 * linePoint：按强度最大值，检测出垂直车位线上的点
 */

void DetectParkingSlot::windowSumCol(Mat &srcImage,Point &startLoction,int step,int windowWidth,
                                     int windowHeight, vector<Point> &linePoint)
{
    int halfWidth=windowWidth/2;
    int halfHeight=windowHeight/2;
    //类似卷积后求图像大小
    int windowsnum=(srcImage.rows-windowHeight)/step+1;
    Mat windowimage;
    Mat sumwindowcol=Mat::zeros(1,windowWidth,CV_32F);
    double maxValue = 0;
    Point maxloction;
    linePoint.push_back(startLoction);
    for (int i=0;i<windowsnum;i++)
    {
        if(linePoint.back().x < halfWidth)
        {
            Rect roi(0, step * i, windowWidth, windowHeight);
            windowimage = srcImage(roi);
            reduce(windowimage, sumwindowcol, 0, CV_REDUCE_SUM, CV_32F);
            minMaxLoc(sumwindowcol, 0, &maxValue, 0, &maxloction);
            if (maxValue > 5)
            {
                linePoint.push_back(Point(maxloction.x, step * i + halfHeight));
            }
        }
            else if(linePoint.back().x+halfWidth>srcImage.cols)
        {
            Rect roi(srcImage.cols-windowWidth,step*i,windowWidth,windowHeight);
            windowimage=srcImage(roi);
            reduce(windowimage,sumwindowcol,0,CV_REDUCE_SUM,CV_32F);
            if (maxValue > 5)
            {
                linePoint.push_back(Point(srcImage.cols-windowWidth + maxloction.x, step * i + halfHeight));
            }
        }
        else
        {
            Rect roi(linePoint.back().x - halfWidth, step * i, windowWidth, windowHeight);
            windowimage = srcImage(roi);
            reduce(windowimage, sumwindowcol, 0, CV_REDUCE_SUM, CV_32F);
            minMaxLoc(sumwindowcol, 0, &maxValue, 0, &maxloction);
            if (maxValue > 5) {
                linePoint.push_back(Point(linePoint.back().x - halfWidth + maxloction.x, step * i + halfHeight));
            }
        }
    }
    linePoint.erase(linePoint.begin());

}

/*
 * srcImage:经预处理后的灰度图像，数据类型为32F
 * startLoction:全局的按列求和和按行求和后最大值的交点位置
 * linePoint：按强度最大值，检测出水平车位线上的点
 */
int DetectParkingSlot::windowSumRow(Mat &srcImage,Point &startLoction, int step,int windowWidth,
                                    int windowHeight,vector<Point> &linePoint)
{
    int halfWidth=windowWidth/2;
    int halfHeight=windowHeight/2;
    //类似卷积后求图像大小
    int windowsnum=(srcImage.cols-startLoction.x-windowWidth)/step+1;
    if(windowsnum<2)
    {
        return 0;
    }
    Mat windowimage;
    Mat sumwindowrow=Mat::zeros(windowHeight,1,CV_32F);
    double maxValue = 0;
    Point maxloction;
    linePoint.push_back(startLoction);
    //暂时没有考虑垂直线极其靠近边缘的情况，即startloction.y-100<0
    for(int i=0;i<windowsnum;i++)
    {
        if(linePoint.back().y-halfHeight<0)
        {
            Rect roi(startLoction.x+step*i,0,windowWidth,windowHeight);
            windowimage=srcImage(roi);
            reduce(windowimage, sumwindowrow, 1, CV_REDUCE_SUM,CV_32F);
            minMaxLoc(sumwindowrow,0,&maxValue, 0, &maxloction);
            if (maxValue>20)
            {
                linePoint.push_back(Point(startLoction.x+step*i+halfWidth,maxloction.y));
            }
        }
        else if(linePoint.back().y+halfHeight>srcImage.rows)
        {
            Rect roi(startLoction.x+step*i,srcImage.rows-windowHeight,windowWidth,windowHeight);
            windowimage=srcImage(roi);
            reduce(windowimage, sumwindowrow, 1, CV_REDUCE_SUM,CV_32F);
            minMaxLoc(sumwindowrow,0,&maxValue, 0, &maxloction);
            if (maxValue>20)
            {
                linePoint.push_back(Point(startLoction.x+step*i+halfWidth,srcImage.rows-windowHeight+maxloction.y));
            }
        }
        else
        {
            Rect roi(startLoction.x+step*i,linePoint.back().y-halfHeight,windowWidth,windowHeight);
            windowimage=srcImage(roi);
            reduce(windowimage, sumwindowrow, 1, CV_REDUCE_SUM,CV_32F);
            minMaxLoc(sumwindowrow,0,&maxValue, 0, &maxloction);
            if (maxValue>0)
            {
                linePoint.push_back(Point(startLoction.x+step*i+halfWidth,linePoint.back().y-halfHeight+maxloction.y));
            }
        }
    }
    linePoint.erase(linePoint.begin());
    return 1;
}
/*
 * lineA 拟合出的垂直车位线的参数
 * lineB 拟合出的水平车位线的参数
 */
Point2f DetectParkingSlot::getCrossPoint(Vec4f &lineA, Vec4f &lineB)
{
    Point2f crossPoint;
    if(lineA[1]==1)
    {
        double kb = lineB[1]/lineB[0];
        crossPoint.x=lineA[2];
        crossPoint.y=kb*(crossPoint.x-lineB[2])+lineB[3];
    }
    else{
        double ka,kb;
        ka = lineA[1]/lineA[0]; //求出LineA斜率
        kb = lineB[1]/lineB[0]; //求出LineB斜率
        crossPoint.x = (ka*lineA[2] - lineA[3] - kb*lineB[2] + lineB[3]) / (ka - kb);
        crossPoint.y = (ka*kb*(lineA[2] - lineB[2]) + ka*lineB[3] - kb*lineA[3]) / (ka - kb);
    }
    return crossPoint;
}

/*
 * 检测车位线函数
 * srcImage：原始图像
 * srcRoi：在原图像上截取的感兴趣区域
 * frangiRoi：frangi滤波后去掉四周的白色区域
 * parkingSpacePoint：检测出的车位线交点的位置，顺序为按照y轴从小到大。
 */
int DetectParkingSlot::detecSlot(Mat &srcImage,Rect &srcRoi,vector<SlotPoint> &parkingSlotPoint)
{
    Mat roi_image0=srcImage(srcRoi);
    Mat gray(roi_image0.rows,roi_image0.cols,CV_8UC1);
    rgbGray(roi_image0,gray);

    //去掉frangi滤波后图像中四条白边
    //相对于原图roi的位置
    Point roistart(srcRoi.x,srcRoi.y);
    Size roisize(srcRoi.width,srcRoi.height);
    //分别按列累加和按行累加
//获取最大合适值
    int maxvalue=HistFromroi(gray);
    Mat grad_y,abs_grad_y;
    Mat gray_image=gray_binnary(gray,maxvalue);
    imshow("gray",gray_image);
    Sobel(gray_image, grad_y,CV_16S,0, 1,3, 1, 1, BORDER_DEFAULT);
    convertScaleAbs(grad_y,abs_grad_y);
    imshow("y向soble", abs_grad_y);
// 按行投影到Ｙ轴
    Mat sumcol=Mat::zeros(1,roisize.width,CV_32F);
    Mat sumrow=Mat::zeros(roisize.height,1,CV_32F);
    reduce(gray_image, sumcol, 0, CV_REDUCE_SUM,CV_32F);
    reduce(abs_grad_y, sumrow, 1, CV_REDUCE_SUM,CV_32F);
    //垂直车位线粗略检测，累加强度最大值的位置
    double maxValue_x = 0;
    Point maxloction_x;
    minMaxLoc(sumcol,0,&maxValue_x, 0, &maxloction_x);
    //垂直车位线精确检测，滑窗检测
    vector<Point>linepointcol;
    windowSumCol(gray_image,maxloction_x,50,50,50,linepointcol);
    if(linepointcol.size()<2)
    {
        //cout<<"未检测到车位线"<<endl;
        namedWindow("image",WINDOW_NORMAL);
        imshow("image",srcImage);
        return 0;
    }
    //将检测出的点转换到原图上
    for(int i=0;i<linepointcol.size();i++)
    {
        linepointcol[i]=linepointcol[i]+roistart;
    }
    Vec4f line_paracol;
    fitLine(linepointcol, line_paracol, cv::DIST_L2, 0, 1e-2, 1e-2);
    //画出检测出的点和拟合出的垂直线
    if(line_paracol[1]==1)
    {
        cv::line(srcImage, Point(line_paracol[2],0), Point(line_paracol[2],srcImage.rows), cv::Scalar(0, 255, 0), 2,CV_AA);
        for (int i = 0; i < linepointcol.size(); i++)
        {
            circle(srcImage, linepointcol[i], 3, cv::Scalar(0, 0, 255), 2, 8, 0);
        }
    }
    else{
        Point point1;
        point1.x = line_paracol[2];
        point1.y = line_paracol[3];
        double k1 = line_paracol[1] / line_paracol[0];
        //计算直线的端点(y = k(x - x0) + y0)
        Point point11, point12;
        point11.x = 0;
        point11.y = k1 * ( point11.x - point1.x) + point1.y;
        point12.x = srcImage.cols-1;
        point12.y = k1 * (point12.x  - point1.x) + point1.y;
        cv::line(srcImage, point11, point12, cv::Scalar(0, 255, 0), 2,CV_AA);
        for (int i = 0; i < linepointcol.size(); i++)
        {
            circle(srcImage, linepointcol[i], 3, cv::Scalar(0, 0, 255), 2, 8, 0);
        }
    }

    //通过求sumrow前两个最大的波峰，粗略找到水平车位线的位置
    //将灰度累加强度值归一化到0～255
    double maxValue_y=0;
    double minValue_y=0;
    minMaxLoc(sumrow,&minValue_y, &maxValue_y, 0, 0);
    sumrow=sumrow*255/(maxValue_y-minValue_y);
    vector<ValueIndex>peaks;
    findPeaks(sumrow,100,100,peaks); //设置峰值的最小阈值为100
    sort(peaks.begin(), peaks.end(),comp);//将峰值降序排列
    Point startloction; //水平滑窗开始的位置
    vector<Point2f>parkingSpacePoint;
    if(peaks.size()>1)
    {

        for(int i=0;i<2;i++)
        {
            startloction=Point(maxloction_x.x,peaks[i].index);
            vector<Point>linepointrow0;
            //cout<<startloction<<endl;
            windowSumRow(abs_grad_y,startloction,20,40,30,linepointrow0);
            Point2f crossPoint0;
            Vec4f line_pararow0;
            if (linepointrow0.size()>1)
            {
                //将所有点转换到原图坐标中
                for(int i=0;i<linepointrow0.size();i++)
                {
                    linepointrow0[i]=linepointrow0[i]+roistart;
                }
                //将检测出的点画到原图上
                for (int i = 0; i < linepointrow0.size(); i++)
                {
                    circle(srcImage, linepointrow0[i], 3, cv::Scalar(0, 0, 255), 2, 8, 0);
                }
                fitLine(linepointrow0, line_pararow0, cv::DIST_L2, 0, 1e-2, 1e-2);
                //求交点
                crossPoint0=getCrossPoint(line_paracol,line_pararow0);


                //画出拟合的水平线
                Point point2;
                point2.x = line_pararow0[2];
                point2.y = line_pararow0[3];
                double k2 = line_pararow0[1] / line_pararow0[0];
                //计算直线的终点(y = k(x - x0) + y0),起点为交点
                Point point21;
                point21.x = srcImage.cols-1;
                point21.y = k2 * (point21.x  - point2.x) + point2.y;
                cv::line(srcImage, crossPoint0, point21, cv::Scalar(0, 255, 0), 2,CV_AA);

            }
            else
            {

                if(line_paracol[1]==1)
                {
                    crossPoint0.y=peaks[i].index+roistart.y; //转换到原图坐标
                    crossPoint0.x=line_paracol[2];
                    //画出垂直于垂直车位线的水平线
                    line(srcImage, crossPoint0, Point(srcImage.cols-1,crossPoint0.y), cv::Scalar(0, 255, 0), 2,CV_AA);
                }
                else{
                    double k=line_paracol[1]/line_paracol[0];
                    double b=line_paracol[3]-k*line_paracol[2];
                    crossPoint0.y=peaks[i].index+roistart.y; //转换到原图坐标
                    crossPoint0.x=(crossPoint0.y-b)/k;
                    //画出垂直于垂直车位线的水平线
                    double k1 = 1/k;
                    Point point21;
                    point21.x = srcImage.cols-1;
                    point21.y = k1 * (point21.x  - crossPoint0.x) + crossPoint0.y;
                    cv::line(srcImage, crossPoint0, point21, cv::Scalar(0, 255, 0), 2,CV_AA);
                }

            }
            parkingSpacePoint.push_back(crossPoint0);
        }
    }
        //只有一个波峰的时候，暂时考虑寻找水平车位线
    else if(peaks.size()==1)
    {
        startloction=Point(maxloction_x.x,peaks[0].index);
        vector<Point>linepointrow0;
        windowSumRow(gray_image,startloction,50,50,50,linepointrow0);
        Point2f crossPoint0;
        Vec4f line_pararow0;
        if (linepointrow0.size()>1)
        {
            //将所有点转换到原图坐标中
            for(int i=0;i<linepointrow0.size();i++)
            {
                linepointrow0[i]=linepointrow0[i]+roistart;
            }
            //将检测出的点画到原图上
            for (int i = 0; i < linepointrow0.size(); i++)
            {
                circle(srcImage, linepointrow0[i], 5, cv::Scalar(0, 0, 255), 2, 8, 0);
            }
            fitLine(linepointrow0, line_pararow0, cv::DIST_L2, 0, 1e-2, 1e-2);
            //求交点
            crossPoint0=getCrossPoint(line_paracol,line_pararow0);

            //画出拟合的水平线
            Point point2;
            point2.x = line_pararow0[2];
            point2.y = line_pararow0[3];
            double k2 = line_pararow0[1] / line_pararow0[0];
            //计算直线的终点(y = k(x - x0) + y0),起点为交点
            Point point21;
            point21.x = srcImage.cols-1;
            point21.y = k2 * (point21.x  - point2.x) + point2.y;
            cv::line(srcImage, crossPoint0, point21, cv::Scalar(0, 255, 0), 2,CV_AA);
        }
        else
        {
            double k=line_paracol[1]/line_paracol[0];
            double b=line_paracol[3]-k*line_paracol[2];
            crossPoint0.y=peaks[0].index+roistart.y;
            crossPoint0.x=(crossPoint0.y-b)/k;
            //画出垂直于垂直车位线的水平线
            double k1 = 1/k;
            Point point21;
            point21.x = srcImage.cols-1;
            point21.y = k1 * (point21.x  - crossPoint0.x) + crossPoint0.y;
            cv::line(srcImage, crossPoint0, point21, cv::Scalar(0, 255, 0), 2,CV_AA);
        }
        parkingSpacePoint.push_back(crossPoint0);

    }
    else
    {
        //cout<<"未检测到车位线"<<endl;
        namedWindow("image",WINDOW_NORMAL);
        imshow("image",srcImage);
        return 0;
    }

    if(parkingSpacePoint.size()>0)
    {
        SlotPoint temp;
        sort(parkingSpacePoint.begin(),parkingSpacePoint.end(),comp1);
        for (int i = 0; i < parkingSpacePoint.size(); i++)
        {
            temp.x=int(parkingSpacePoint[i].x);
            temp.y=int(parkingSpacePoint[i].y);
            parkingSlotPoint.push_back(temp);
            circle(srcImage, parkingSpacePoint[i], 2, cv::Scalar(0, 255, 255), 2, 8, 0);
        }

    }

    namedWindow("image",WINDOW_NORMAL);
    imshow("image",srcImage);
    return 1;
}



