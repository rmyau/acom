package lab4;

import org.bytedeco.opencv.opencv_core.*;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_highgui.*;

public class findBorders {

    public static int[][] convolution(Mat img, int[][] kernel) {
        int kernelSize = kernel.length;
        int xStart = kernelSize / 2;
        int yStart = kernelSize / 2;
        int[][] matr = new int[img.rows()][img.rows()];


        for (int i = 0; i < img.rows(); i++) {
            for (int j = 0; j < img.cols(); j++) {
                matr[i][j] = img.ptr(i, j).get();
            }
        }
        for (int i = xStart; i < img.rows() - xStart; i++) {
            for (int j = yStart; j < img.cols() - yStart; j++) {
                int val = 0;

                for (int k = -(kernelSize / 2); k <= kernelSize / 2; k++) {
                    for (int l = -(kernelSize / 2); l <= kernelSize / 2; l++) {
                        int imgValue = img.ptr(i + k, j + l).get();
                        if (imgValue < 0) {
                            imgValue = 128 + (128 % imgValue);
                        }
                        int res = imgValue * kernel[k + (kernelSize / 2)][l + (kernelSize / 2)];
                        val += res;
                    }
                }

                matr[i][j] = val;
            }
        }
        return matr;
    }

    public static int getAngleNumber(double x, double y) {
        double tg = (x != 0) ? y / x : 999;
        if (x < 0) {
            if (y < 0) {
                if (tg > 2.414) return 0;
                else if (tg < 0.414) return 6;
                else return 7;
            } else {
                if (tg < -2.414) return 4;
                else if (tg < -0.414) return 5;
                else return 6;
            }
        } else {
            if (y < 0) {
                if (tg < -2.414) return 0;
                else if (tg < -0.414) return 1;
                else return 2;
            } else {
                if (tg < 0.414) return 2;
                else if (tg < 2.414) return 3;
                else return 4;
            }
        }
    }

    public static int[] getOffset(int angle) {
        int xShift = 0;
        int yShift = 0;

        if (angle == 0 || angle == 4) {
            xShift = 0;
        } else if (angle > 0 && angle < 4) {
            xShift = 1;
        } else {
            xShift = -1;
        }

        if (angle == 2 || angle == 6) {
            yShift = 0;
        } else if (angle > 2 && angle < 6) {
            yShift = -1;
        } else {
            yShift = 1;
        }

        return new int[]{xShift, yShift};
    }

    public static void main(String[] args) {
        double standardDeviation = 20;
        int kernelSize = 3;
        int boundPath = 6;

        // Load the image
        Mat img = imread("images/test_512.jpg", IMREAD_GRAYSCALE);
        Mat imgBlurCV2 = new Mat();
        GaussianBlur(img, imgBlurCV2, new Size(kernelSize, kernelSize), standardDeviation);
        imshow("blur", imgBlurCV2);

        // Create the kernel
        int[][] Gx = {
                {-1, 0, 1},
                {-2, 0, 2},
                {-1, 0, 1}
        };
        int[][] Gy = {
                {-1, -2, -1},
                {0, 0, 0},
                {1, 2, 1}
        };

        // Perform convolution
        int[][] imgGx = convolution(img, Gx);
        int[][] imgGy = convolution(img, Gy);

        // Calculate gradient length matrix
        Mat matrGradient = new Mat(img.size(), CV_64F);
        for (int i = 0; i < img.rows(); i++) {
            for (int j = 0; j < img.cols(); j++) {
                int gx = imgGx[i][j];
                int gy = imgGy[i][j];
                double magnitude = Math.sqrt(gx * gx + gy * gy);
                matrGradient.ptr(i, j).putDouble(magnitude);
            }
        }
        //нормирование
        double maxGradient = 0;
        for (int i = 0; i < img.rows(); i++) {
            for (int j = 0; j < img.cols(); j++) {
                if (matrGradient.ptr(i, j).getDouble() > maxGradient) {
                    maxGradient = matrGradient.ptr(i, j).getDouble();
                }
            }
        }
        for (int i = 0; i < img.rows(); i++) {
            for (int j = 0; j < img.cols(); j++) {
                double newVal = matrGradient.ptr(i, j).getDouble()/maxGradient;
                matrGradient.ptr(i, j).putDouble(newVal);
            }
        }


        // Calculate gradient angle matrix
        Mat imgAngles = new Mat(img.size(), img.type());
        for (int i = 0; i < img.rows(); i++) {
            for (int j = 0; j < img.cols(); j++) {
                int angle = getAngleNumber(imgGx[i][j], imgGy[i][j]);
                imgAngles.ptr(i, j).put((byte) angle);
            }
        }


        Mat imgBorder = new Mat(img.size(), CV_32S);
        for (int i = 0; i < img.rows(); i++) {
            for (int j = 0; j < img.cols(); j++) {
                if (i == 0 || i == img.rows() - 1 || j == 0 || j == img.cols() - 1) {
                    imgBorder.ptr(i, j).putInt(0);
                } else {
                    int[] offset = getOffset(imgAngles.ptr(i, j).get());
                    double gradient = matrGradient.ptr(i, j).getDouble();
                    boolean isMax = gradient >= matrGradient.ptr(i + offset[1], j + offset[0]).getDouble() &&
                            gradient >= matrGradient.ptr(i - offset[1], j - offset[0]).getDouble();

                    imgBorder.ptr(i, j).putInt((byte) (isMax ? 255 : 0));
                }
            }
        }

        //Double thresholding
        Mat doubleThresholded = new Mat(img.size(), img.type());
        maxGradient = 0;
        for (int i = 0; i < img.rows(); i++) {
            for (int j = 0; j < img.cols(); j++) {
                if (matrGradient.ptr(i, j).getDouble() > maxGradient) {
                    maxGradient = matrGradient.ptr(i, j).getDouble();
                }
            }
        }
        double lowerBound = maxGradient / boundPath;
        double upperBound = maxGradient - (maxGradient / boundPath);

        for (int i = 0; i < img.rows(); i++) {
            for (int j = 0; j < img.cols(); j++) {
                double gradient = matrGradient.ptr(i, j).getDouble();
                if (imgBorder.ptr(i, j).get() == (byte) 255) {
                    if (gradient >= lowerBound && gradient <= upperBound) {
                        boolean hasStrongNeighbor = false;
                        for (int k = -1; k <= 1; k++) {
                            for (int l = -1; l <= 1; l++) {
                                if (matrGradient.ptr(i + k, j + l).getDouble() >= lowerBound) {
                                    hasStrongNeighbor = true;
                                    break;
                                }
                            }
                            if (hasStrongNeighbor) {
                                break;
                            }
                        }
                        if (hasStrongNeighbor) {
                            doubleThresholded.ptr(i, j).put((byte) 255);
                        }
                    } else if (gradient > upperBound) {
                        doubleThresholded.ptr(i, j).put((byte) 255);
                    }
                } else {
                    doubleThresholded.ptr(i, j).put((byte) 0);
                }
            }
        }

        imshow("result", doubleThresholded);
        waitKey(0);
    }
}
