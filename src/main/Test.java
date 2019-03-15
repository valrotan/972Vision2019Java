package main;

import java.util.ArrayList;
import java.util.List;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

public class Test {

	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		setup();
	}

	static VideoCapture capture;

	static List<Point3> objPointsList;

	static Scalar HSV_LOW = new Scalar(70, 62, 240);
	static Scalar HSV_HIGH = new Scalar(94, 200, 255);

	static double MIN_AREA = 1000;

	static MatOfPoint3f objPoints; // real life 3d coordinates of object being tracked
	static MatOfPoint2f imgPoints;
	static Mat frame;

	static MatOfDouble distCoeffs;
	static List<Double> distCoeffsList;
	static Mat mtx;
	static Mat rvec;
	static Mat tvec;

	public static Mat process(Mat in) {
		return in;
	}
	
	public static void setup() {
		// setting object points
		objPointsList = new ArrayList<Point3>();
		objPointsList.add(new Point3(-5.936, -.501, 0));
		objPointsList.add(new Point3(-4, 0, 0));
		objPointsList.add(new Point3(-5.377, 5.325, 0));
		objPointsList.add(new Point3(-7.313, 4.824, 0));

		objPointsList.add(new Point3(5.936, -.501, 0));
		objPointsList.add(new Point3(4, 0, 0));
		objPointsList.add(new Point3(5.377, 5.325, 0));
		objPointsList.add(new Point3(7.313, 4.824, 0));

		objPoints = new MatOfPoint3f();
		objPoints.fromList(objPointsList);

		capture = new VideoCapture(0);

		distCoeffsList = new ArrayList<>();
		distCoeffsList.add(0.09231099);
		distCoeffsList.add(-0.5412314);
		distCoeffsList.add(0.00438005);
		distCoeffsList.add(0.00368862);
		distCoeffsList.add(1.95331636);

		distCoeffs = new MatOfDouble();
		distCoeffs.fromList(distCoeffsList);

		/*
		 * [[[473.80026 160.63934]]
		 * 
		 * [[475.76413 204.87401]]
		 * 
		 * [[528.3594 205.90977]]
		 * 
		 * [[525.2771 161.96121]]
		 * 
		 * [[478.3829 126.92225]]
		 * 
		 * [[480.5602 173.02005]]
		 * 
		 * [[536.2645 174.35051]]
		 * 
		 * [[532.8351 128.56291]]]
		 */
		mtx = new Mat(3, 3, CvType.CV_32FC1);
		float[] buff = {
				(float) 166.30436665602548, 
				(float) 0.0, 
				(float) 153.34803805846408, 
				(float) 0.0, 
				(float) 171.91094733236676, 
				(float) 132.4151799881648, 
				(float) 0.0, 
				(float) 0.0, 
				(float) 1.0
				};
		mtx.put(0, 0, buff);
	}

	public static void main(String[] args) {

		Imgcodecs imageCodecs = new Imgcodecs();

		frame = new Mat();
//		capture.read(frame);
		frame = imageCodecs.imread("test/front.JPG");

		// filter for tape thing & detect points on it
		Mat mask = filter(frame);
		imgPoints = detectPoints(frame, mask);

//		imageCodecs.imwrite("out/mask.jpg", frame);
//		imageCodecs.imwrite(file2, ); 

		rvec = new Mat(3, 1, CvType.CV_64FC1);
		tvec = new Mat(3, 1, CvType.CV_64FC1);

//		System.out.println(imgPoints);
//		System.out.println(objPoints);
//		System.out.println(distCoeffs);
//		System.out.println(rvec);
//		System.out.println(tvec);

		Calib3d.solvePnP(objPoints, imgPoints, mtx, distCoeffs, rvec, tvec);

		System.out.println(tvec.dump());
		System.out.println(rvec.dump());

	}

	public static Mat filter(Mat frame) {
		Mat frameHSV = new Mat();
		Imgproc.cvtColor(frame, frameHSV, Imgproc.COLOR_BGR2HSV); // create hsv encoded frame

		Mat thresh = new Mat(); // thresholded/filtered image
		Core.inRange(frameHSV, HSV_LOW, HSV_HIGH, thresh);
		return thresh;
	}

	public static MatOfPoint2f detectPoints(Mat frame, Mat mask) {
		List<MatOfPoint> contours = new ArrayList<>();
		Imgproc.findContours(mask, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

		ArrayList<Point> pntSet = new ArrayList<>();

		for (MatOfPoint contour : contours) {
			double area = Imgproc.contourArea(contour);
			if (area < MIN_AREA)
				continue;

			double epsilon = 0.1 * Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true);
			MatOfPoint2f approx = new MatOfPoint2f();
			Imgproc.approxPolyDP(new MatOfPoint2f(contour.toArray()), approx, epsilon, true);

			long sides = approx.total();
			if (sides != 4)
				continue;

			for (Point pnt : approx.toArray()) {
				pntSet.add(pnt);
				Imgproc.circle(frame, pnt, 10, new Scalar(0, 255, 255), 15);
			}
		}
		MatOfPoint2f pnts = new MatOfPoint2f();
		pnts.fromList(pntSet);
		return pnts;
	}

}
