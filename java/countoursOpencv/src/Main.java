import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.Image;
import java.util.*;
import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
class FindContours {
    private Mat srcGray = new Mat();
    private Mat src;
    private JFrame frame;
    private JLabel imgSrcLabel;
    private JLabel imgContoursLabel;
    private static final int MAX_THRESHOLD = 255;
    private int threshold = 100;
    private Random rng = new Random(12345);
    public FindContours(String[] args) {
        String filename = args.length > 0 ? args[0] : "D:/Users/admin/IdeaProjects/countoursOpencv/src/test4.jpg";

        // test4 area : Area of points : 167983.0 / 577 ---> 291.131715771230
        // test3 area : Area of points : 172164.0 / 600 ---> 286.94
        // test  area : Area of points : 564001.0 / 815 ---> 692.0257668711




        src = Imgcodecs.imread(filename);
        if (src.empty()) {
            System.err.println("Cannot read image: " + filename);
            System.exit(0);
        }
        Imgproc.cvtColor(src, srcGray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.blur(srcGray, srcGray, new Size(3, 3));
        // Create and set up the window.
        frame = new JFrame("Finding contours in your image demo");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        // Set up the content pane.
        Image img = HighGui.toBufferedImage(src);
        addComponentsToPane(frame.getContentPane(), img);
        // Use the content pane's default BorderLayout. No need for
        // "setLayout(new BorderLayout());
        // Display the window.
        frame.pack();
        frame.setVisible(true);
        update();
    }
    private void addComponentsToPane(Container pane, Image img) {
        if (!(pane.getLayout() instanceof BorderLayout)) {
            pane.add(new JLabel("Container doesn't use BorderLayout!"));
            return;
        }
        JPanel sliderPanel = new JPanel();
        sliderPanel.setLayout(new BoxLayout(sliderPanel, BoxLayout.PAGE_AXIS));
        sliderPanel.add(new JLabel("Canny threshold: "));
        JSlider slider = new JSlider(0, MAX_THRESHOLD, threshold);
        slider.setMajorTickSpacing(20);
        slider.setMinorTickSpacing(10);
        slider.setPaintTicks(true);
        slider.setPaintLabels(true);
        slider.addChangeListener(new ChangeListener() {
            @Override
            public void stateChanged(ChangeEvent e) {
                JSlider source = (JSlider) e.getSource();
                threshold = source.getValue();
                update();
            }
        });
        sliderPanel.add(slider);
        pane.add(sliderPanel, BorderLayout.PAGE_START);
        JPanel imgPanel = new JPanel();
        imgSrcLabel = new JLabel(new ImageIcon(img));
        imgPanel.add(imgSrcLabel);
        Mat blackImg = Mat.zeros(srcGray.size(), CvType.CV_8U);
        imgContoursLabel = new JLabel(new ImageIcon(HighGui.toBufferedImage(blackImg)));
        imgPanel.add(imgContoursLabel);
        pane.add(imgPanel, BorderLayout.CENTER);
    }

    public Mat hough_Lines(Mat mRgba){

        Mat tempMat = new Mat();
        Mat lines = new Mat();

        Imgproc.cvtColor(mRgba, tempMat, Imgproc.COLOR_BGR2GRAY);

        Imgproc.adaptiveThreshold(tempMat, tempMat, 255,1,1,11,2);

        Imgproc.erode(tempMat, tempMat, new Mat(), new Point(-1, -1), 1);

        Imgproc.HoughLinesP(tempMat, lines, 1.0, Math.PI / 180.0, 140, 20, 20);

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

        // Some other approaches
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

        result[0] = Collections.min(srcPoints, sumComparator);        // Upper left has the minimal sum
        result[2] = Collections.max(srcPoints, sumComparator);        // Lower right has the maximal sum
        result[1] = Collections.min(srcPoints, differenceComparator); // Upper right has the minimal difference
        result[3] = Collections.max(srcPoints, differenceComparator); // Lower left has the maximal difference

        return result;
    }

    public double getRealAspectRatio(int imageWidth, int imageHeight, Point[] pts) {

        double u0 = imageWidth/2;
        double v0 = imageHeight/2;

        System.out.println("u0 : " + u0);
        System.out.println("v0 : " + v0);

        double m1x = pts[0].x + u0;
        double m1y = pts[0].y + v0;
        double m2x = pts[1].x + u0;
        double m2y = pts[1].y + v0;
        double m3x = pts[2].x + u0;
        double m3y = pts[2].y + v0;
        double m4x = pts[3].x + u0;
        double m4y = pts[3].y + v0;

        /*
        double m1x = pts[0].x;
        double m1y = pts[0].y;
        double m2x = pts[1].x;
        double m2y = pts[1].y;
        double m3x = pts[2].x;
        double m3y = pts[2].y;
        double m4x = pts[3].x;
        double m4y = pts[3].y;


        double m1x = pts[0].x - u0;
        double m1y = pts[0].y - v0;
        double m2x = pts[1].x - u0;
        double m2y = pts[1].y - v0;
        double m3x = pts[2].x - u0;
        double m3y = pts[2].y - v0;
        double m4x = pts[3].x - u0;
        double m4y = pts[3].y - v0;
         */

        System.out.println("m1x : " + m1x);
        System.out.println("m1y : " + m1y);
        System.out.println("m2x : " + m2x);
        System.out.println("m2y : " + m2y);
        System.out.println("m3x : " + m3x);
        System.out.println("m3y : " + m3y);
        System.out.println("m4x : " + m4x);
        System.out.println("m4y : " + m4y);


        double k2 = ((m1y - m4y)*m3x - (m1x - m4x)*m3y + m1x*m4y - m1y*m4x) /
                ((m2y - m4y)*m3x - (m2x - m4x)*m3y + m2x*m4y - m2y*m4x) ;

        System.out.println("k2 : " + k2);

        double k3 = ((m1y - m4y)*m2x - (m1x - m4x)*m2y + m1x*m4y - m1y*m4x) /
                ((m3y - m4y)*m2x - (m3x - m4x)*m2y + m3x*m4y - m3y*m4x) ;

        System.out.println("k3 : " + k3);

        double f_squared =
                -((k3*m3y - m1y)*(k2*m2y - m1y) + (k3*m3x - m1x)*(k2*m2x - m1x)) /
                        ((k3 - 1)*(k2 - 1)) ;

        System.out.println("f_squared : " + f_squared);

        double whRatio = Math.sqrt(
                (Math.pow((k2 - 1),2) + Math.pow((k2*m2y - m1y),2)/f_squared + Math.pow((k2*m2x - m1x),2)/f_squared) /
                        (Math.pow((k3 - 1),2) + Math.pow((k3*m3y - m1y),2)/f_squared + Math.pow((k3*m3x - m1x),2)/f_squared)
        ) ;

        System.out.println("whRatio1 : " + whRatio);

        if (k2==1 && k3==1 ) {
            whRatio = Math.sqrt(
                    (Math.pow((m2y-m1y),2) + Math.pow((m2x-m1x),2)) /
                            (Math.pow((m3y-m1y),2) + Math.pow((m3x-m1x),2)));
        }

        System.out.println("whRatio2 : " + whRatio);

        return (double)(whRatio);
    }

    private Mat fourPointTransform(Mat src, Point[] pts) {


        Point ul = pts[0];
        Point ur = pts[1];
        Point lr = pts[2];
        Point ll = pts[3];

        double widthA = Math.sqrt(Math.pow(lr.x - ll.x, 2) + Math.pow(lr.y - ll.y, 2));
        double widthB = Math.sqrt(Math.pow(ur.x - ul.x, 2) + Math.pow(ur.y - ul.y, 2));

        double heightA = Math.sqrt(Math.pow(ur.x - lr.x, 2) + Math.pow(ur.y - lr.y, 2));
        double heightB = Math.sqrt(Math.pow(ul.x - ll.x, 2) + Math.pow(ul.y - ll.y, 2));

        double ratio = getRealAspectRatio(src.width(), src.height(), pts);
        System.out.println("ratio : " + ratio);
        // double ratio = src.size().height / (double) 600;

        // FIND A PATTERN FROM THESE NUMBERS:
        // test4 area : Area of points : 167983.0 and 291.131715771230  --->  577
        // test3 area : Area of points : 172164.0 and 286.94            --->  600
        // test  area : Area of points : 564001.0 and 692.0257668711    --->  815

        double maxWidth = Math.max(widthA, widthB) * ratio;

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

    private void update() {
        Mat cannyOutput = new Mat();

        Imgproc.Canny(srcGray, cannyOutput, threshold, threshold * 2);
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(cannyOutput, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        Mat drawing = Mat.zeros(cannyOutput.size(), CvType.CV_8UC3);


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

        //Point[] point = biggest.toArray();
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

        //double quarter011X = (point[0].x + point[1].x)/4;
        //double quarter011Y = (point[0].y + point[1].y)/4;

        double quarter011X = (point[0].x + mid1X)/2;
        double quarter011Y = (point[0].y + mid1Y)/2;

        double quarter121X = (point[1].x + mid2X)/2;
        double quarter121Y = (point[1].y + mid2Y)/2;

        double quarter231X = (point[2].x + mid3X)/2;
        double quarter231Y = (point[2].y + mid3Y)/2;

        double quarter301X = (point[3].x + mid4X)/2;
        double quarter301Y = (point[3].y + mid4Y)/2;

        double quarter012X = (mid1X + point[1].x)/2;
        double quarter012Y = (mid1Y + point[1].y)/2;

        double quarter122X = (mid2X + point[2].x)/2;
        double quarter122Y = (mid2Y + point[2].y)/2;

        double quarter232X = (mid3X + point[3].x)/2;
        double quarter232Y = (mid3Y + point[3].y)/2;

        double quarter302X = (mid4X + point[0].x)/2;
        double quarter302Y = (mid4Y + point[0].y)/2;

        /*
        double centre1X = (point[0].x + point[2].x)/2;
        double centre1Y = (point[0].y + point[2].y)/2;

        double centre2X = (point[1].x + point[3].x)/2;
        double centre2Y = (point[1].y + point[3].y)/2;

        Imgproc.circle(drawing, new Point(centre1X, centre1Y), 0, new Scalar(255, 255, 255), 10);
        Imgproc.circle(drawing, new Point(centre2X, centre2Y), 0, new Scalar(255, 0, 255), 10);
        */

        Imgproc.circle(drawing, new Point(centreX, centreY), 0, new Scalar(255, 255, 255), 10);

        Imgproc.circle(drawing, new Point(mid1X, mid1Y), 0, new Scalar(255,   0,   0), 10);
        Imgproc.circle(drawing, new Point(mid2X, mid2Y), 0, new Scalar(  0, 255,   0), 10);
        Imgproc.circle(drawing, new Point(mid3X, mid3Y), 0, new Scalar(  0,   0, 255), 10);
        Imgproc.circle(drawing, new Point(mid4X, mid4Y), 0, new Scalar(  0, 255, 255), 10);

        Imgproc.circle(drawing, new Point(quarter011X, quarter011Y), 0, new Scalar(255,   0,   255), 10);

        //double distanceX = Math.abs(centreX - mid3X);
        //double distanceY = Math.abs(centreY - mid3Y);

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

        double prec = 0.0;

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
        /*

        distanceX = centreX - quarter011X;
        distanceY = centreY - quarter011Y;

        stepX = distanceX / d;
        stepY = distanceY / d;

        for (int i = 1; i < (d + 1); i++){
            xx = quarter011X + (i * stepX);
            yy = quarter011Y + (i * stepY);

            colors = src.get(Double.valueOf(xx).intValue(), Double.valueOf(yy).intValue());

            System.out.println("colors step (" + i + ") : " + colors[0] + " " + colors[1] + " " + colors[2]);

            Imgproc.circle(drawing, new Point(xx, yy), 0, new Scalar(colors[0], colors[1], colors[2]), 10);

        }

        distanceX = centreX - quarter121X;
        distanceY = centreY - quarter121Y;

        stepX = distanceX / d;
        stepY = distanceY / d;

        for (int i = 1; i < (d + 1); i++){
            xx = quarter121X + (i * stepX);
            yy = quarter121Y + (i * stepY);

            colors = src.get(Double.valueOf(xx).intValue(), Double.valueOf(yy).intValue());

            System.out.println("colors step (" + i + ") : " + colors[0] + " " + colors[1] + " " + colors[2]);

            Imgproc.circle(drawing, new Point(xx, yy), 0, new Scalar(colors[0], colors[1], colors[2]), 10);

        }

        distanceX = centreX - quarter231X;
        distanceY = centreY - quarter231Y;

        stepX = distanceX / d;
        stepY = distanceY / d;

        for (int i = 1; i < (d + 1); i++){
            xx = quarter231X + (i * stepX);
            yy = quarter231Y + (i * stepY);

            colors = src.get(Double.valueOf(xx).intValue(), Double.valueOf(yy).intValue());

            System.out.println("colors step (" + i + ") : " + colors[0] + " " + colors[1] + " " + colors[2]);

            Imgproc.circle(drawing, new Point(xx, yy), 0, new Scalar(colors[0], colors[1], colors[2]), 10);

        }

        distanceX = centreX - quarter301X;
        distanceY = centreY - quarter301Y;

        stepX = distanceX / d;
        stepY = distanceY / d;

        for (int i = 1; i < (d + 1); i++){
            xx = quarter301X + (i * stepX);
            yy = quarter301Y + (i * stepY);

            colors = src.get(Double.valueOf(xx).intValue(), Double.valueOf(yy).intValue());

            System.out.println("colors step (" + i + ") : " + colors[0] + " " + colors[1] + " " + colors[2]);

            Imgproc.circle(drawing, new Point(xx, yy), 0, new Scalar(colors[0], colors[1], colors[2]), 10);

        }

        distanceX = centreX - quarter012X;
        distanceY = centreY - quarter012Y;

        stepX = distanceX / d;
        stepY = distanceY / d;

        for (int i = 1; i < (d + 1); i++){
            xx = quarter012X + (i * stepX);
            yy = quarter012Y + (i * stepY);

            colors = src.get(Double.valueOf(xx).intValue(), Double.valueOf(yy).intValue());

            System.out.println("colors step (" + i + ") : " + colors[0] + " " + colors[1] + " " + colors[2]);

            Imgproc.circle(drawing, new Point(xx, yy), 0, new Scalar(colors[0], colors[1], colors[2]), 10);

        }

        distanceX = centreX - quarter122X;
        distanceY = centreY - quarter122Y;

        stepX = distanceX / d;
        stepY = distanceY / d;

        for (int i = 1; i < (d + 1); i++){
            xx = quarter122X + (i * stepX);
            yy = quarter122Y + (i * stepY);

            colors = src.get(Double.valueOf(xx).intValue(), Double.valueOf(yy).intValue());

            System.out.println("colors step (" + i + ") : " + colors[0] + " " + colors[1] + " " + colors[2]);

            Imgproc.circle(drawing, new Point(xx, yy), 0, new Scalar(colors[0], colors[1], colors[2]), 10);

        }

        distanceX = centreX - quarter232X;
        distanceY = centreY - quarter232Y;

        stepX = distanceX / d;
        stepY = distanceY / d;

        for (int i = 1; i < (d + 1); i++){
            xx = quarter232X + (i * stepX);
            yy = quarter232Y + (i * stepY);

            colors = src.get(Double.valueOf(xx).intValue(), Double.valueOf(yy).intValue());

            System.out.println("colors step (" + i + ") : " + colors[0] + " " + colors[1] + " " + colors[2]);

            Imgproc.circle(drawing, new Point(xx, yy), 0, new Scalar(colors[0], colors[1], colors[2]), 10);

        }

        distanceX = centreX - quarter302X;
        distanceY = centreY - quarter302Y;

        stepX = distanceX / d;
        stepY = distanceY / d;

        for (int i = 1; i < (d + 1); i++){
            xx = quarter302X + (i * stepX);
            yy = quarter302Y + (i * stepY);

            colors = src.get(Double.valueOf(xx).intValue(), Double.valueOf(yy).intValue());

            System.out.println("colors step (" + i + ") : " + colors[0] + " " + colors[1] + " " + colors[2]);

            Imgproc.circle(drawing, new Point(xx, yy), 0, new Scalar(colors[0], colors[1], colors[2]), 10);

        }


        distanceX = centreX - point[0].x;
        distanceY = centreY - point[0].y;

        stepX = distanceX / d;
        stepY = distanceY / d;

        for (int i = 1; i < (d + 1); i++){
            xx = point[0].x + (i * stepX);
            yy = point[0].y + (i * stepY);

            colors = src.get(Double.valueOf(xx).intValue(), Double.valueOf(yy).intValue());

            System.out.println("colors step (" + i + ") : " + colors[0] + " " + colors[1] + " " + colors[2]);

            Imgproc.circle(drawing, new Point(xx, yy), 0, new Scalar(colors[0], colors[1], colors[2]), 10);

        }

        distanceX = centreX - point[1].x;
        distanceY = centreY - point[1].y;

        stepX = distanceX / d;
        stepY = distanceY / d;

        for (int i = 1; i < (d + 1); i++){
            xx = point[1].x + (i * stepX);
            yy = point[1].y + (i * stepY);

            System.out.println("Double.valueOf(xx).intValue() : " + Double.valueOf(xx).intValue());
            System.out.println("Double.valueOf(yy).intValue() : " + Double.valueOf(yy).intValue());

            colors = src.get(Double.valueOf(xx).intValue(), Double.valueOf(yy).intValue());

            System.out.println("colors step (" + i + ") : " + colors[0] + " " + colors[1] + " " + colors[2]);

            Imgproc.circle(drawing, new Point(xx, yy), 0, new Scalar(colors[0], colors[1], colors[2]), 10);

        }

        distanceX = centreX - point[2].x;
        distanceY = centreY - point[2].y;

        stepX = distanceX / d;
        stepY = distanceY / d;

        for (int i = 1; i < (d + 1); i++){
            xx = point[2].x + (i * stepX);
            yy = point[2].y + (i * stepY);

            System.out.println("Double.valueOf(xx).intValue() : " + Double.valueOf(xx).intValue());
            System.out.println("Double.valueOf(yy).intValue() : " + Double.valueOf(yy).intValue());

            colors = src.get(Double.valueOf(xx).intValue(), Double.valueOf(yy).intValue());

            System.out.println("colors step (" + i + ") : " + colors[0] + " " + colors[1] + " " + colors[2]);

            Imgproc.circle(drawing, new Point(xx, yy), 0, new Scalar(colors[0], colors[1], colors[2]), 10);

        }


        System.out.println("point3.x " + point[3].x);
        System.out.println("point3.y " + point[3].y);

        System.out.println("centreX : " + centreX);
        System.out.println("centreY : " + centreY);

        distanceX = (centreX - point[3].x);
        distanceY = (centreY - point[3].y);

        System.out.println("distanceX centreX to point3X " + distanceX);
        System.out.println("distanceY centreY to point3Y " + distanceY);

        stepX = distanceX / d;
        stepY = distanceY / d;

        System.out.println("stepX " + stepX);
        System.out.println("stepY " + stepY);

        for (int i = 1; i < (d + 1); i++){
            xx = centreX + (i * stepX);
            yy = centreY + (i * stepY);

            System.out.println("Double.valueOf(xx).intValue() : " + Double.valueOf(xx).intValue());
            System.out.println("Double.valueOf(yy).intValue() : " + Double.valueOf(yy).intValue());

            System.out.println("src width and height : " + src.width() + " __ " + src .height());

            //colors = srcGray.get(Double.valueOf(xx).intValue(), Double.valueOf(yy).intValue());
            //Imgproc.circle(drawing, new Point(xx, yy),  0, new Scalar(colors[0], colors[1], colors[2]), 10);

            //System.out.println("colors step (" + i + ") : " + colors[0] + " " + colors[1] + " " + colors[2]);


        }

         */



        //drawing = fourPointTransform(srcGray, sortPoints(point));

        //System.out.println("Area of points : " + Imgproc.contourArea(new MatOfPoint(sortPoints(point))));

        //displayHoughLines(drawing, hough_Lines(drawing));

        /*
        for (int i = 0; i < contours.size(); i++) {
            Scalar color = new Scalar(rng.nextInt(256), rng.nextInt(256), rng.nextInt(256));
            Imgproc.drawContours(drawing, contours, i, color, 2, Core.LINE_8, hierarchy, 0, new Point());
        }

         */
        imgContoursLabel.setIcon(new ImageIcon(HighGui.toBufferedImage(drawing)));
        frame.repaint();
    }
}
public class Main {
    public static void main(String[] args) {
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        // Schedule a job for the event dispatch thread:
        // creating and showing this application's GUI.
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                new FindContours(args);
            }
        });
    }
}