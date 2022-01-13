package com.example.android.ocvelectrocodeapp;

import static org.opencv.core.CvType.CV_8UC1;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.Window;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.photo.Photo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class CameraActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2{
    private static final String TAG="MainActivity";

    private Mat mRgba;
    private Mat mGray;

    private static final int MAX_HEIGHT = 500;

    private CameraBridgeViewBase mOpenCvCameraView;
    private BaseLoaderCallback mLoaderCallback =new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case LoaderCallbackInterface
                        .SUCCESS:{
                    Log.i(TAG,"OpenCv Is loaded");
                    mOpenCvCameraView.enableView();
                }
                default:
                {
                    super.onManagerConnected(status);

                }
                break;
            }
        }
    };

    public CameraActivity(){
        Log.i(TAG,"Instantiated new "+this.getClass());
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        int MY_PERMISSIONS_REQUEST_CAMERA=0;
        
        if (ContextCompat.checkSelfPermission(CameraActivity.this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(CameraActivity.this, new String[] {Manifest.permission.CAMERA}, MY_PERMISSIONS_REQUEST_CAMERA);
        }

        setContentView(R.layout.activity_camera);

        mOpenCvCameraView=(CameraBridgeViewBase) findViewById(R.id.frame_surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()){
            
            Log.d(TAG,"Opencv initialization is done");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        else{
            
            Log.d(TAG,"Opencv is not loaded. try again");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0,this,mLoaderCallback);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView !=null){
            mOpenCvCameraView.disableView();
        }
    }

    public void onDestroy(){
        super.onDestroy();
        if(mOpenCvCameraView !=null){
            mOpenCvCameraView.disableView();
        }

    }

    public void onCameraViewStarted(int width ,int height){
        mRgba=new Mat(height,width, CvType.CV_8UC4);
        mGray =new Mat(height,width, CV_8UC1);
    }


    public void onCameraViewStopped(){
        mRgba.release();
    }


    public MatOfPoint2f rectanglePoints(Mat mRgba){

        Mat tempMat = new Mat();

        Mat hierarchy = new Mat();

        List<MatOfPoint> contourList = new ArrayList<MatOfPoint>();

        Imgproc.cvtColor(mRgba, tempMat, Imgproc.COLOR_BGR2GRAY);

        Imgproc.GaussianBlur(tempMat, tempMat, new Size(5,5), 0);

        Imgproc.adaptiveThreshold(tempMat, tempMat, 255,1,1,11,2);

        Imgproc.erode(tempMat, tempMat, new Mat(), new Point(-1, -1), 1);

        //Imgproc.Canny(tempMat, tempMat, 250, 150);

        Imgproc.findContours(tempMat, contourList, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        MatOfPoint2f biggest = new MatOfPoint2f();
        double max_area = 0;
        for (MatOfPoint i : contourList) {
            double area = Imgproc.contourArea(i);
            if (area > 100) {
                MatOfPoint2f m = new MatOfPoint2f(i.toArray());
                double peri = Imgproc.arcLength(m, true);
                MatOfPoint2f approx = new MatOfPoint2f();
                Imgproc.approxPolyDP(m, approx, 0.02 * peri, true);
                if (area > max_area && approx.total() == 4) {
                    biggest = approx;
                    max_area = area;
                }
            }
        }

        return biggest;

    }

    public void displayRectanglePoints(Mat mRgba, MatOfPoint2f points){

        Point[] point = points.toArray();

        Imgproc.circle(mRgba, new Point(point[0].x, point[0].y), 0, new Scalar(255, 255, 0), 10);
        Imgproc.circle(mRgba, new Point(point[1].x, point[1].y), 0, new Scalar(255, 255, 0), 10);
        Imgproc.circle(mRgba, new Point(point[2].x, point[2].y), 0, new Scalar(255, 255, 0), 10);
        Imgproc.circle(mRgba, new Point(point[3].x, point[3].y), 0, new Scalar(255, 255, 0), 10);

    }

    public Mat hough_Lines(Mat mRgba){

        Mat tempMat = new Mat();
        Mat lines = new Mat();

        Imgproc.cvtColor(mRgba, tempMat, Imgproc.COLOR_BGR2GRAY);

        Imgproc.adaptiveThreshold(tempMat, tempMat, 255,1,1,11,2);

        Imgproc.erode(tempMat, tempMat, new Mat(), new Point(-1, -1), 1);

        Imgproc.HoughLines(tempMat, lines, 1.0, Math.PI / 180.0, 140, 20, 20);

        return lines;
    }



    public void displayHoughLines(Mat mRgba, Mat lines){

        Point p1 = new Point();
        Point p2 = new Point();

        double a, b;
        double x0, y0;

        for (int i = 0; i < lines.rows(); i++){

            double[] vec = lines.get(i, 0);
            double rho = vec[0];
            double theta = vec[1];

            a = Math.cos(theta);
            b = Math.sin(theta);

            x0 = a * rho;
            y0 = b * rho;

            p1.x = Math.round(x0 + 1000 * (-b));
            p1.y = Math.round(y0 + 1000 * a);
            p2.x = Math.round(x0 - 1000 * (-b));
            p2.y = Math.round(y0 - 1000 * a);

            Imgproc.line(mRgba, p1, p2, new Scalar(255, 0, 255), 2, 1);
        }

    }

    private Mat applyThreshold(Mat src) {
        Mat m = new Mat();

        Imgproc.cvtColor(src, m, Imgproc.COLOR_BGR2GRAY);

//        Imgproc.adaptiveThreshold(m, m, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 15, 15);
//        Imgproc.threshold(m, m, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);

        Imgproc.GaussianBlur(m, m, new Size(5, 5), 0);
        Imgproc.adaptiveThreshold(m, m, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2);

        return m;

    }

    private Point[] sortPoints(Point[] src) {
        ArrayList<Point> srcPoints = new ArrayList<>(Arrays.asList(src));
        Point[] result = {null, null, null, null};

        Comparator<Point> sumComparator = new Comparator<Point>() {
            @Override
            public int compare(Point lhs, Point rhs) {
                return Double.valueOf(lhs.y + lhs.x).compareTo(rhs.y + rhs.x);
            }
        };
        Comparator<Point> differenceComparator = new Comparator<Point>() {
            @Override
            public int compare(Point lhs, Point rhs) {
                return Double.valueOf(lhs.y - lhs.x).compareTo(rhs.y - rhs.x);
            }
        };

        result[0] = Collections.min(srcPoints, sumComparator);        
        result[2] = Collections.max(srcPoints, sumComparator);        
        result[1] = Collections.min(srcPoints, differenceComparator); 
        result[3] = Collections.max(srcPoints, differenceComparator); 

        return result;
    }

    private Mat fourPointTransform(Mat src, Point[] pts) {
        double ratio = src.size().height / (double) MAX_HEIGHT;

        Point ul = pts[0];
        Point ur = pts[1];
        Point lr = pts[2];
        Point ll = pts[3];

        double widthA = Math.sqrt(Math.pow(lr.x - ll.x, 2) + Math.pow(lr.y - ll.y, 2));
        double widthB = Math.sqrt(Math.pow(ur.x - ul.x, 2) + Math.pow(ur.y - ul.y, 2));
        double maxWidth = Math.max(widthA, widthB) * ratio;

        double heightA = Math.sqrt(Math.pow(ur.x - lr.x, 2) + Math.pow(ur.y - lr.y, 2));
        double heightB = Math.sqrt(Math.pow(ul.x - ll.x, 2) + Math.pow(ul.y - ll.y, 2));
        double maxHeight = Math.max(heightA, heightB) * ratio;

        Mat resultMat = new Mat(Double.valueOf(maxHeight).intValue(), Double.valueOf(maxWidth).intValue(), CvType.CV_8UC4);

        Mat srcMat = new Mat(4, 1, CvType.CV_32FC2);
        Mat dstMat = new Mat(4, 1, CvType.CV_32FC2);
        srcMat.put(0, 0, ul.x * ratio, ul.y * ratio, ur.x * ratio, ur.y * ratio, lr.x * ratio, lr.y * ratio, ll.x * ratio, ll.y * ratio);
        dstMat.put(0, 0, 0.0, 0.0, maxWidth, 0.0, maxWidth, maxHeight, 0.0, maxHeight);

        Mat M = Imgproc.getPerspectiveTransform(srcMat, dstMat);
        Imgproc.warpPerspective(src, resultMat, M, resultMat.size());

        srcMat.release();
        dstMat.release();
        M.release();

        return resultMat;
    }


    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame){

        mRgba = inputFrame.rgba();

        Mat srcGray = new Mat();
        Mat cannyOutput = new Mat();
        Mat hierarchy = new Mat();

        int threshold = 100;

        Random rng = new Random(12345);

        try{

            Imgproc.cvtColor(mRgba, srcGray, Imgproc.COLOR_BGR2GRAY);

            //Imgproc.GaussianBlur(srcGray, srcGray, new Size(5,5), 0);

            //Imgproc.adaptiveThreshold(srcGray, cannyOutput, 255,1,1,11,2);

            Imgproc.Canny(srcGray, cannyOutput, threshold, threshold * 2);

            Mat drawing = Mat.zeros(cannyOutput.size(), CvType.CV_8UC3);

            List<MatOfPoint> contours = new ArrayList<>();

            Imgproc.findContours(cannyOutput, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

            MatOfPoint2f biggest = new MatOfPoint2f();
            double max_area = 0;
            int index = 0;
            for (MatOfPoint i : contours) {
                double area = Imgproc.contourArea(i);
                if (area > 100) {
                    MatOfPoint2f m = new MatOfPoint2f(i.toArray());
                    double peri = Imgproc.arcLength(m, true);
                    MatOfPoint2f approx = new MatOfPoint2f();
                    Imgproc.approxPolyDP(m, approx, 0.02 * peri, true);
                    if (area > max_area && approx.total() == 4) {
                        biggest = approx;
                        max_area = area;
                        index = contours.indexOf(i);
                    }
                }
            }

            Scalar color = new Scalar(rng.nextInt(256), rng.nextInt(256), rng.nextInt(256));
            Imgproc.drawContours(drawing, contours, index, color, 2, Core.LINE_8, hierarchy, 0, new Point());

            Point[] point = sortPoints(biggest.toArray());


            Imgproc.circle(drawing, new Point(point[0].x, point[0].y), 0, new Scalar(255, 0, 0), 10);
            Imgproc.circle(drawing, new Point(point[1].x, point[1].y), 0, new Scalar(0, 255, 0), 10);
            Imgproc.circle(drawing, new Point(point[2].x, point[2].y), 0, new Scalar(0, 0, 255), 10);
            Imgproc.circle(drawing, new Point(point[3].x, point[3].y), 0, new Scalar(255, 255, 0), 10);

            Imgproc.putText(drawing, "P1", new Point(point[0].x, point[0].y), Core.FONT_HERSHEY_COMPLEX, 1, new Scalar(255, 255, 255), 2);
            Imgproc.putText(drawing, "P2", new Point(point[1].x, point[1].y), Core.FONT_HERSHEY_COMPLEX, 1, new Scalar(255, 255, 255), 2);
            Imgproc.putText(drawing, "P3", new Point(point[2].x, point[2].y), Core.FONT_HERSHEY_COMPLEX, 1, new Scalar(255, 255, 255), 2);
            Imgproc.putText(drawing, "P4", new Point(point[3].x, point[3].y), Core.FONT_HERSHEY_COMPLEX, 1, new Scalar(255, 255, 255), 2);

            double mid1X = (point[0].x + point[1].x)/2;
            double mid1Y = (point[0].y + point[1].y)/2;

            double mid2X = (point[1].x + point[2].x)/2;
            double mid2Y = (point[1].y + point[2].y)/2;

            double mid3X = (point[2].x + point[3].x)/2;
            double mid3Y = (point[2].y + point[3].y)/2;

            double mid4X = (point[3].x + point[0].x)/2;
            double mid4Y = (point[3].y + point[0].y)/2;

            double centre1X = (point[0].x + point[2].x)/2;
            double centre1Y = (point[0].y + point[2].y)/2;

            double centre2X = (point[1].x + point[3].x)/2;
            double centre2Y = (point[1].y + point[3].y)/2;

            double centreX = (point[0].x + point[1].x + point[2].x + point[3].x + mid1X + mid2X + mid3X + mid4X + centre1X + centre2X)/10;
            double centreY = (point[0].y + point[1].y + point[2].y + point[3].y + mid1Y + mid2Y + mid3Y + mid4Y + centre1Y + centre2Y)/10;

            Imgproc.circle(drawing, new Point(centreX, centreY), 0, new Scalar(255, 255, 255), 10);

            Imgproc.circle(drawing, new Point(mid1X, mid1Y), 0, new Scalar(255,   0,   0), 10);
            Imgproc.circle(drawing, new Point(mid2X, mid2Y), 0, new Scalar(  0, 255,   0), 10);
            Imgproc.circle(drawing, new Point(mid3X, mid3Y), 0, new Scalar(  0,   0, 255), 10);
            Imgproc.circle(drawing, new Point(mid4X, mid4Y), 0, new Scalar(  0, 255, 255), 10);

            int d = 15;

            double distanceX = centreX - mid3X;
            double distanceY = centreY - mid3Y;

            double stepX = distanceX / d;
            double stepY = distanceY / d;

            double xx = 0;
            double yy = 0;

            double r = 0;
            double g = 0;
            double b = 0;

            //double g = 0;

            double[] colors;

            for (int i = 1; i < (d + 1); i++){
                xx = centreX + (i * stepX);
                yy = centreY + (i * stepY);

                colors = srcGray.get(Double.valueOf(xx).intValue(), Double.valueOf(yy).intValue());

                System.out.println("colors step (" + i + ") : " + colors[0]);

                Imgproc.circle(drawing, new Point(xx, yy), 0, new Scalar(colors[0], colors[0], colors[0]), 10);

            }


            distanceX = centreX - mid1X;
            distanceY = centreY - mid1Y;

            stepX = distanceX / d;
            stepY = distanceY / d;

            for (int i = 1; i < (d + 1); i++){
                xx = centreX + (i * stepX);
                yy = centreY + (i * stepY);

                colors = srcGray.get(Double.valueOf(xx).intValue(), Double.valueOf(yy).intValue());

                System.out.println("colors step (" + i + ") : " + colors[0]);

                Imgproc.circle(drawing, new Point(xx, yy), 0, new Scalar(colors[0], colors[0], colors[0]), 10);

            }

            distanceX = centreX - mid2X;
            distanceY = centreY - mid2Y;

            stepX = distanceX / d;
            stepY = distanceY / d;

            for (int i = 1; i < (d + 1); i++){
                xx = centreX + (i * stepX);
                yy = centreY + (i * stepY);

                colors = srcGray.get(Double.valueOf(xx).intValue(), Double.valueOf(yy).intValue());

                System.out.println("colors step (" + i + ") : " + colors[0]);

                Imgproc.circle(drawing, new Point(xx, yy), 0, new Scalar(colors[0], colors[0], colors[0]), 10);

            }

            distanceX = centreX - mid4X;
            distanceY = centreY - mid4Y;

            stepX = distanceX / d;
            stepY = distanceY / d;

            for (int i = 1; i < (d + 1); i++){
                xx = centreX + (i * stepX);
                yy = centreY + (i * stepY);

                colors = srcGray.get(Double.valueOf(xx).intValue(), Double.valueOf(yy).intValue());

                System.out.println("colors step (" + i + ") : " + colors[0]);

                Imgproc.circle(drawing, new Point(xx, yy), 0, new Scalar(colors[0], colors[0], colors[0]), 10);

            }






            //MatOfPoint2f matPoints = rectanglePoints(mRgba);

            //Point[] points = matPoints.toArray();



            //int t = 3;
            //Rect R = new Rect(new Point(points[0].x, points[0].y), new Point(points[2].x, points[2].y));

            //Mat cropped = fourPointTransform(mRgba, sortPoints(points));

            //Imgproc.threshold(cropped, cropped, 200, 255, 3);



            /*
            Bitmap bm = Bitmap.createBitmap(cropped.width(), cropped.height(), Bitmap.Config.ARGB_8888);
            org.opencv.android.Utils.matToBitmap(cropped, bm);

             */

            //Mat lines = hough_Lines(mRgba);

            //displayHoughLines(mRgba, lines);

            //displayRectanglePoints(mRgba, matPoints);

            return drawing;

        }catch (Exception e){



            return mRgba;
        }

    }

}
