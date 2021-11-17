#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <bits-stdc++.h>

#define OPENPOSE_FLAGS_DISABLE_POSE
#define OPENPOSE_FLAGS_DISABLE_PRODUCER
#define OPENPOSE_FLAGS_DISABLE_DISPLAY
#include <openpose/flags.hpp>
#include <openpose/headers.hpp>
#include<string>

DEFINE_string(image_path, "",
	"Process an image. Read all standard formats (jpg, png, bmp, etc.).");
// Display
DEFINE_bool(no_display, false,
	"Enable to disable the visual display.");

#include <windows.h>
#include <mmsystem.h>

struct DoublePoint
{
public:
	double x, y;
};

DoublePoint CoordinateOfViolence[20];

int NumPerS;
int vio = 0, non = 0;


void display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr,
	std::string RRS,int ArrOfViolence[],int driverVio,float *score)
{
	cv::Point pp;
	int k=0;
	try
	{

		if (datumsPtr != nullptr && !datumsPtr->empty())
		{
			// Display image
			const cv::Mat cvMat = OP_OP2CVCONSTMAT(datumsPtr->at(0)->cvOutputData);
			if (!cvMat.empty())
			{
				if (RRS == "violence")
				{
					while (k < driverVio)
					{
						if (CoordinateOfViolence[ArrOfViolence[k]].x != 0)
						{
							cv::Rect R = cv::Rect(int (CoordinateOfViolence[ArrOfViolence[k]].x),int ( CoordinateOfViolence[ArrOfViolence[k]].y), 50, 100);
							cv::rectangle(cvMat, R, cv::Scalar(255, 225, 225), 5, 8, 0);
							pp.x = int (CoordinateOfViolence[ArrOfViolence[k]].x - 10);
							pp.y = int ( CoordinateOfViolence[ArrOfViolence[k]].y - 10);
						}
						else
						{
							op::opLog("There Is Violence But cannot detect the person ", op::Priority::High);
						}
						k++;
					}
					vio++;
				}
				else { non++; }

				for (int i = 0; i < NumPerS; i++)
				{
					
					pp.x = int(CoordinateOfViolence[i].x - 10);
					pp.y = int(CoordinateOfViolence[i].y - 10);
					cv::putText(cvMat, std::to_string(score[i])
						+ "%", pp, cv::HersheyFonts::FONT_ITALIC, 1, cv::Scalar(225, 225, 225), 3, 8);
				}

				cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", cvMat);
				cv::waitKey(1);
			}
			else
				op::opLog("Empty cv::Mat as output.", op::Priority::High, __LINE__, __FUNCTION__, __FILE__);
		}
		else
			op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	}
}

struct Point {

public:
	double a1=0;
	double a2=0;
	double a3=0, a4=0, a5=0, a6=0, a7=0, a8=0;

	double distance;
	int val;
};

double calculateAngle(double P1X, double P1Y, double P2X, double P2Y, double P3X, double P3Y) {

	double numerator = P2Y * (P1X - P3X) + P1Y * (P3X - P2X) + P3Y * (P2X - P1X);
	double denominator = (P2X - P1X) * (P1X - P3X) + (P2Y - P1Y) * (P1Y - P3Y);
	double ratio = numerator / denominator;

	double angleRad = atan(ratio);
	double angleDeg = (angleRad * 180) / 3.141;

	return angleDeg;
}

void StoreKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr
	, bool Is_training, Point arr[], int& KnnDriver, Point PointToTest[], int& Data)
{
	try
	{
		
		if (datumsPtr != nullptr && !datumsPtr->empty())
		{

			Point P[3];
			Point Joints[25]{};
			std::string valueToPrint;
			double result;
			const auto& poseKeypoints = datumsPtr->at(0)->poseKeypoints;

			if (Is_training)
			{
				for (auto person = 0; person < poseKeypoints.getSize(0); person++)
				{
					for (auto bodyPart = 0; bodyPart < poseKeypoints.getSize(1); bodyPart++)
					{
						for (auto xyscore = 0; xyscore < poseKeypoints.getSize(2); xyscore++)
						{
							valueToPrint = std::to_string(poseKeypoints[{person, bodyPart, xyscore}]) + " ";
							if (xyscore == 0)
							{
								Joints[bodyPart].a1 = std::stod(valueToPrint);
							}
							if (xyscore == 1)
							{
								Joints[bodyPart].a2 = std::stod(valueToPrint);
							}
						}
					}
				
					op::opLog(std::to_string(KnnDriver) + " , " +std::to_string(Data), op::Priority::High);

					
					
					// Left Sholder
					P[0].a1 = Joints[1].a1;
					P[0].a2 = Joints[1].a2;
					P[1].a1 = Joints[2].a1;
					P[1].a2 = Joints[2].a2;
					P[2].a1 = Joints[3].a1;
					P[2].a2 = Joints[3].a2;
					result = calculateAngle(P[1].a1, P[1].a2, P[0].a1, P[0].a2, P[2].a1, P[2].a2); //Elbow -> Sholder -> Rest.
					arr[KnnDriver].a1 = result;
					
					if (isnan(result))
					{
						if (isnan(result))
						{
							if (Data == 1)
							{
								arr[KnnDriver].a1 = 1.5902;
							}
							else if (Data == 0)
							{
								arr[KnnDriver].a1 = -2.18849;
							}
						}
					}

					//if (Joints[1].a1 == 0 || Joints[1].a2 == 0 ||
					//	Joints[2].a1 == 0 || Joints[2].a2 == 0 ||
					//	Joints[3].a1 == 0 || Joints[3].a2 == 0)
					//{
					//	//rep[KnnDriver].
					//	//if (Data == 1)
					//	//{
					//	//	arr[KnnDriver].a1 = 74.6651;
					//	//}
					//	//else if (Data == 0)
					//	//{
					//	//	arr[KnnDriver].a1 = 85.6669;
					//	//}
					//}


					// Right Sholder
					P[0].a1 = Joints[1].a1;
					P[0].a2 = Joints[1].a2;
					P[1].a1 = Joints[5].a1;
					P[1].a2 = Joints[5].a2;
					P[2].a1 = Joints[6].a1;
					P[2].a2 = Joints[6].a2;
					result = calculateAngle(P[1].a1, P[1].a2, P[0].a1, P[0].a2, P[2].a1, P[2].a2); //Elbow -> Sholder -> Rest.
					arr[KnnDriver].a2 = result;

					if (isnan(result))
					{
						if (isnan(result))
						{
							if (Data == 1)
							{
								arr[KnnDriver].a2 = -15.1393;
							}
							else if (Data == 0)
							{
								arr[KnnDriver].a2 = -2.14098;
							}
						}
					}

					//if (Joints[1].a1 == 0 || Joints[1].a2 == 0 ||
					//	Joints[5].a1 == 0 || Joints[5].a2 == 0 ||
					//	Joints[6].a1 == 0 || Joints[6].a2 == 0)
					//{
					//	if (Data == 1)
					//	{
					//		arr[KnnDriver].a2 = 106.029;
					//	}
					//	else if (Data == 0)
					//	{
					//		arr[KnnDriver].a2 = 99.5143;
					//	}
					//}

					// Left elbow
					P[0].a1 = Joints[2].a1;
					P[0].a2 = Joints[2].a2;
					P[1].a1 = Joints[3].a1;
					P[1].a2 = Joints[3].a2;
					P[2].a1 = Joints[4].a1;
					P[2].a2 = Joints[4].a2;
					result = calculateAngle(P[1].a1, P[1].a2, P[0].a1, P[0].a2, P[2].a1, P[2].a2); //Elbow -> Sholder -> Rest.
					arr[KnnDriver].a3 = result;
					
					if (isnan(result))
					{
						if (isnan(result))
						{
							if (Data == 1)
							{
								arr[KnnDriver].a3 = 3.08789;
							}
							else if (Data == 0)
							{
								arr[KnnDriver].a3 = 2.1058;
							}
						}
					}

					//if (Joints[2].a1 == 0 || Joints[2].a2 == 0 ||
					//	Joints[3].a1 == 0 || Joints[3].a2 == 0 ||
					//	Joints[4].a1 == 0 || Joints[4].a2 == 0)
					//{
					//	if (Data == 1)
					//	{
					//		arr[KnnDriver].a3 = 98.9564;
					//	}
					//	else if (Data == 0)
					//	{
					//		arr[KnnDriver].a3 = 85.0415;
					//	}
					//}

					// Right elbow
					P[0].a1 = Joints[5].a1;
					P[0].a2 = Joints[5].a2;
					P[1].a1 = Joints[6].a1;
					P[1].a2 = Joints[6].a2;
					P[2].a1 = Joints[7].a1;
					P[2].a2 = Joints[7].a2;
					result = calculateAngle(P[1].a1, P[1].a2, P[0].a1, P[0].a2, P[2].a1, P[2].a2); //Elbow -> Sholder -> Rest.
					arr[KnnDriver].a4 = result;

					if (isnan(result))
					{
						if (isnan(result))
						{
							if (Data == 1)
							{
								arr[KnnDriver].a4 = 6.98473;
							}
							else if (Data == 0)
							{
								arr[KnnDriver].a4 = 1.46292;
							}
						}
					}

					//if (Joints[5].a1 == 0 || Joints[5].a2 == 0 ||
					//	Joints[6].a1 == 0 || Joints[6].a2 == 0 ||
					//	Joints[7].a1 == 0 || Joints[7].a2 == 0)
					//{
					//	if (Data == 1)
					//	{
					//		arr[KnnDriver].a4 = 84.1423;
					//	}
					//	else if (Data == 0)
					//	{
					//		arr[KnnDriver].a4 = 94.5322;
					//	}
					//}

					// Left socket
					P[0].a1 = Joints[8].a1;
					P[0].a2 = Joints[8].a2;
					P[1].a1 = Joints[9].a1;
					P[1].a2 = Joints[9].a2;
					P[2].a1 = Joints[10].a1;
					P[2].a2 = Joints[10].a2;
					result = calculateAngle(P[1].a1, P[1].a2, P[0].a1, P[0].a2, P[2].a1, P[2].a2); //Elbow -> Sholder -> Rest.
					arr[KnnDriver].a5 = result;

					if (isnan(result))
					{
						if (Data == 1)
						{
							arr[KnnDriver].a5 = 10.3206;
						}
						else if (Data == 0)
						{
							arr[KnnDriver].a5 = 6.75702;
						}
					}

					//if (Joints[8].a1 == 0 || Joints[8].a2 == 0 ||
					//	Joints[9].a1 == 0 || Joints[9].a2 == 0 ||
					//	Joints[10].a1== 0 || Joints[10].a2 == 0)
					//{
					//	if (Data == 1)
					//	{
					//		arr[KnnDriver].a5 = 79.1958;
					//	}
					//	else if (Data == 0)
					//	{
					//		arr[KnnDriver].a5 = 89.5162;
					//	}
					//}

					// Right socket
					P[0].a1 = Joints[8].a1;
					P[0].a2 = Joints[8].a2;
					P[1].a1 = Joints[12].a1;
					P[1].a2 = Joints[12].a2;
					P[2].a1 = Joints[13].a1;
					P[2].a2 = Joints[13].a2;
					result = calculateAngle(P[1].a1, P[1].a2, P[0].a1, P[0].a2, P[2].a1, P[2].a2); //Elbow -> Sholder -> Rest.
					arr[KnnDriver].a6 = result;

					if (isnan(result))
					{
						if (Data == 1)
						{
							arr[KnnDriver].a6 = -15.8119;
						}
						else if (Data == 0)
						{
							arr[KnnDriver].a6 = -6.03688;
						}
					}

					//if (Joints[8].a1 == 0 || Joints[8].a2 == 0 ||
					//	Joints[12].a1 == 0 || Joints[12].a2 == 0 ||
					//	Joints[13].a1 == 0 || Joints[13].a2 == 0)
					//{
					//	if (Data == 1)
					//	{
					//		arr[KnnDriver].a6 = 105.1;
					//	}
					//	else if (Data == 0)
					//	{
					//		arr[KnnDriver].a6 = 88.8663;
					//	}
					//}

					// Left knee
					P[0].a1 = Joints[9].a1;
					P[0].a2 = Joints[9].a2;
					P[1].a1 = Joints[10].a1;
					P[1].a2 = Joints[10].a2;
					P[2].a1 = Joints[11].a1;
					P[2].a2 = Joints[11].a2;
					result = calculateAngle(P[1].a1, P[1].a2, P[0].a1, P[0].a2, P[2].a1, P[2].a2); //Elbow -> Sholder -> Rest.
					arr[KnnDriver].a7 = result;

					if (isnan(result))
					{
						if (Data == 1)
						{
							arr[KnnDriver].a7 = 6.77537;
						}
						else if (Data == 0)
						{
							arr[KnnDriver].a7 = -1.01978;
						}
					}

					//if (Joints[9].a1 == 0 || Joints[9].a2 == 0 ||
					//	Joints[10].a1 == 0 || Joints[10].a2 == 0 ||
					//	Joints[11].a1 == 0 || Joints[11].a2 == 0)
					//{
					//	if (Data == 1)
					//	{
					//		arr[KnnDriver].a7 = 82.6912;
					//	}
					//	else if (Data == 0)
					//	{
					//		arr[KnnDriver].a7 = 107.62;
					//	}
					//}

					// Right Knee
					P[0].a1 = Joints[12].a1;
					P[0].a2 = Joints[12].a2;
					P[1].a1 = Joints[13].a1;
					P[1].a2 = Joints[13].a2;
					P[2].a1 = Joints[14].a1;
					P[2].a2 = Joints[14].a2;
					result = calculateAngle(P[1].a1, P[1].a2, P[0].a1, P[0].a2, P[2].a1, P[2].a2); //Elbow -> Sholder -> Rest.
					arr[KnnDriver].a8 = result;
					arr[KnnDriver].val = Data;

					if (isnan(result))
					{
						if (Data == 1)
						{
							arr[KnnDriver].a8 = 1.84573;
						}
						else if (Data == 0)
						{
							arr[KnnDriver].a8 = 0.587264;
						}
					}

					//if (Joints[12].a1 == 0 || Joints[12].a2 == 0 ||
					//	Joints[13].a1 == 0 || Joints[13].a2 == 0 ||
					//	Joints[14].a1 == 0 || Joints[14].a2 == 0)
					//{
					//	if (Data == 1)
					//	{
					//		arr[KnnDriver].a8 = 93.3151;
					//	}
					//	else if (Data == 0)
					//	{
					//		arr[KnnDriver].a8 = 90.7355;
					//	}
					//}
					

					


					std::fstream myfile;
					myfile.open("",std::ios::app); //Use The Path of the dataset TXT. file

					myfile << std::to_string(arr[KnnDriver].a1) + "\n";
					myfile << std::to_string(arr[KnnDriver].a2) + "\n";
					myfile << std::to_string(arr[KnnDriver].a3) + "\n";
					myfile << std::to_string(arr[KnnDriver].a4) + "\n";
					myfile << std::to_string(arr[KnnDriver].a5) + "\n";
					myfile << std::to_string(arr[KnnDriver].a6) + "\n";
					myfile << std::to_string(arr[KnnDriver].a7) + "\n";
					myfile << std::to_string(arr[KnnDriver].a8) + "\n";
					myfile << std::to_string(arr[KnnDriver].val) + "\n";
					
					myfile.close();

					KnnDriver++;
					
					NumPerS = poseKeypoints.getSize(0);
				}
				op::opLog(" ", op::Priority::High);
			}
			else
			{
				for (auto person = 0; person < poseKeypoints.getSize(0); person++)
				{
					
					for (auto bodyPart = 0; bodyPart < poseKeypoints.getSize(1); bodyPart++)
					{
						for (auto xyscore = 0; xyscore < poseKeypoints.getSize(2); xyscore++)
						{
							valueToPrint = std::to_string(poseKeypoints[{person, bodyPart, xyscore}]) + " ";
							if (xyscore == 0)
							{
								Joints[bodyPart].a1 = std::stod(valueToPrint);
							}
							if (xyscore == 1)
							{
								Joints[bodyPart].a2 = std::stod(valueToPrint);
							}
						}
					}

					if (Joints[0].a1 != 0)
					{
						CoordinateOfViolence[person].x = Joints[0].a1;
						CoordinateOfViolence[person].y = Joints[0].a2;
					}
					else if (Joints[2].a1 != 0)
					{
						CoordinateOfViolence[person].x = Joints[2].a1;
						CoordinateOfViolence[person].y = Joints[2].a2;
					}
					else if (Joints[5].a1 != 0)
					{
						CoordinateOfViolence[person].x = Joints[5].a1;
						CoordinateOfViolence[person].y = Joints[5].a2;
					}

					// Left Sholder
					P[0].a1 = Joints[1].a1;
					P[0].a2 = Joints[1].a2;
					P[1].a1 = Joints[2].a1;
					P[1].a2 = Joints[2].a2;
					P[2].a1 = Joints[3].a1;
					P[2].a2 = Joints[3].a2;
					result = calculateAngle(P[1].a1, P[1].a2, P[0].a1, P[0].a2, P[2].a1, P[2].a2); //Elbow -> Sholder -> Rest.
					PointToTest[person].a1 = result;

					// Right Sholder
					P[0].a1 = Joints[1].a1;
					P[0].a2 = Joints[1].a2;
					P[1].a1 = Joints[5].a1;
					P[1].a2 = Joints[5].a2;
					P[2].a1 = Joints[6].a1;
					P[2].a2 = Joints[6].a2;
					result = calculateAngle(P[1].a1, P[1].a2, P[0].a1, P[0].a2, P[2].a1, P[2].a2); //Elbow -> Sholder -> Rest.
					PointToTest[person].a2 = result;



					// Left elbow
					P[0].a1 = Joints[2].a1;
					P[0].a2 = Joints[2].a2;
					P[1].a1 = Joints[3].a1;
					P[1].a2 = Joints[3].a2;
					P[2].a1 = Joints[4].a1;
					P[2].a2 = Joints[4].a2;
					result = calculateAngle(P[1].a1, P[1].a2, P[0].a1, P[0].a2, P[2].a1, P[2].a2); //Elbow -> Sholder -> Rest.
					PointToTest[person].a3 = result;

					// Right elbow
					P[0].a1 = Joints[5].a1;
					P[0].a2 = Joints[5].a2;
					P[1].a1 = Joints[6].a1;
					P[1].a2 = Joints[6].a2;
					P[2].a1 = Joints[7].a1;
					P[2].a2 = Joints[7].a2;
					result = calculateAngle(P[1].a1, P[1].a2, P[0].a1, P[0].a2, P[2].a1, P[2].a2); //Elbow -> Sholder -> Rest.
					PointToTest[person].a4 = result;

					// Left socket
					P[0].a1 = Joints[8].a1;
					P[0].a2 = Joints[8].a2;
					P[1].a1 = Joints[9].a1;
					P[1].a2 = Joints[9].a2;
					P[2].a1 = Joints[10].a1;
					P[2].a2 = Joints[10].a2;
					result = calculateAngle(P[1].a1, P[1].a2, P[0].a1, P[0].a2, P[2].a1, P[2].a2); //Elbow -> Sholder -> Rest.
					PointToTest[person].a5 = result;

					// Right socket
					P[0].a1 = Joints[8].a1;
					P[0].a2 = Joints[8].a2;
					P[1].a1 = Joints[12].a1;
					P[1].a2 = Joints[12].a2;
					P[2].a1 = Joints[13].a1;
					P[2].a2 = Joints[13].a2;
					result = calculateAngle(P[1].a1, P[1].a2, P[0].a1, P[0].a2, P[2].a1, P[2].a2); //Elbow -> Sholder -> Rest.
					PointToTest[person].a6 = result;

					// Left knee
					P[0].a1 = Joints[9].a1;
					P[0].a2 = Joints[9].a2;
					P[1].a1 = Joints[10].a1;
					P[1].a2 = Joints[10].a2;
					P[2].a1 = Joints[11].a1;
					P[2].a2 = Joints[11].a2;
					result = calculateAngle(P[1].a1, P[1].a2, P[0].a1, P[0].a2, P[2].a1, P[2].a2); //Elbow -> Sholder -> Rest.
					PointToTest[person].a7 = result;

					// Right Knee
					P[0].a1 = Joints[12].a1;
					P[0].a2 = Joints[12].a2;
					P[1].a1 = Joints[13].a1;
					P[1].a2 = Joints[13].a2;
					P[2].a1 = Joints[14].a1;
					P[2].a2 = Joints[14].a2;
					result = calculateAngle(P[1].a1, P[1].a2, P[0].a1, P[0].a2, P[2].a1, P[2].a2); //Elbow -> Sholder -> Rest.
					PointToTest[person].a8 = result;


					

					NumPerS = poseKeypoints.getSize(0);
					
				}
				op::opLog(" ", op::Priority::High);
			}
		}
		else
			op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);
	}
	catch (const std::exception& e)
	{
		op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
	}
}

bool comparison(Point a, Point b)	
{
	return (a.distance < b.distance);
}

float classifyAPoint(Point arr[], int n, int k, Point p)
{
	// Fill distances of all points from p
	for (int i = 0; i < n; i++)
		arr[i].distance =
		sqrt((arr[i].a1 - p.a1) * (arr[i].a1 - p.a1) // 8 angels
			+ (arr[i].a2 - p.a2) * (arr[i].a2 - p.a2)
			+ (arr[i].a3 - p.a3) * (arr[i].a3 - p.a3)
			+ (arr[i].a4 - p.a4) * (arr[i].a4 - p.a4)
			+ (arr[i].a5 - p.a5) * (arr[i].a5 - p.a5)
			+ (arr[i].a6 - p.a6) * (arr[i].a6 - p.a6)
			+ (arr[i].a7 - p.a7) * (arr[i].a7 - p.a7)
			+ (arr[i].a8 - p.a8) * (arr[i].a8 - p.a8)
		);

	// Sort the Points by distance from p
	std::sort(arr, arr + n, comparison);
	// Now consider the first k elements and only
	// two groups

	float freq1 = 0;     // Frequency of group 0
	float freq2 = 0;     // Frequency of group 1
	for (int i = 0; i < k; i++)
	{
		if (arr[i].val == 0)
			freq1++;
		else if (arr[i].val == 1)
			freq2++;
	}

	return freq2 / (freq1  + freq2);
}

int Start(int driver, bool& training, Point arr[], const int n, Point PointToTest[], int& KnnDriver, int K, int Data)
{
	try
	{
		op::opLog("Starting OpenPose demo...", op::Priority::High);
		const auto opTimer = op::getTimerInit();

		std::string RRS = "x";
		int HowManyV = 0;
		int ArrOfViolene[20]{99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99};
		int ArrOfVioleneDriver = 0;
		bool IsNear = false;
		

		// Configuring OpenPose
		op::opLog("Configuring OpenPose...", op::Priority::High);
		op::Wrapper opWrapper{ op::ThreadManagerMode::Asynchronous };
		// Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
		if (FLAGS_disable_multi_thread)
			opWrapper.disableMultiThreading();

		
		opWrapper.start();

		if (!training)
		{
			

			for (int i = 100; i < driver; i++)
			{
				const cv::Mat cvImageToProcess = cv::imread(FLAGS_image_path + std::to_string(i) + ".jpg");
				const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(cvImageToProcess);
				auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
				ArrOfVioleneDriver = 0;

				StoreKeypoints(datumProcessed, training, arr, KnnDriver, PointToTest, Data);
				
				float* Score = new float[NumPerS];

				for (int k = 0; k < NumPerS; k++)
				{
					float TheFinal = classifyAPoint(arr, n, K, PointToTest[k]);
					
					std::cout << "suspicion percentage of person " << k << ": "<< TheFinal << "\n";

					if (TheFinal < 0.5)
					{
						RRS = "Non-Violence";

					}
					else if (TheFinal >= 0.5 )
					{
						RRS = "violence";
						ArrOfViolene[ArrOfVioleneDriver] = k;
						ArrOfVioleneDriver++;
						HowManyV++;
						
					}

					Score[k] = TheFinal;
				}

				
				if (HowManyV > 0)
				{
					RRS = "violence";
				}
				else
				{
					RRS = "Non-Violence";
				}

				op::opLog("The Frame Is:" + RRS + "," + std::to_string(HowManyV) + " suspects", op::Priority::High);
				if (!FLAGS_no_display)
					display(datumProcessed,RRS,ArrOfViolene,ArrOfVioleneDriver, Score);

				HowManyV = 0;
				IsNear = false;
			}
		}
		else
		{
			for (int i = 1; i <= driver; i++)
			{
				const cv::Mat cvImageToProcess = cv::imread(FLAGS_image_path + "1 (" + std::to_string(i) + ")" + ".jpg");
				const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(cvImageToProcess);
				auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);

				StoreKeypoints(datumProcessed, training, arr, KnnDriver, PointToTest, Data);
			}
		}

		
		op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);

		
		return 0;
	}
	catch (const std::exception&)
	{
		return -1;
	}
}

int ImportTheData(Point arr[])
{
	std::ifstream myfile;
	myfile.open("", std::ios::app); //Use the path of the dataset .TXT File.
	std::string Line;
	int k=0;

	while (getline(myfile, Line))
	{
		
		arr[k].a1 = std::stod(Line);
		getline(myfile, Line);
		arr[k].a2 = std::stod(Line);
		getline(myfile, Line);
		arr[k].a3 = std::stod(Line);
		getline(myfile, Line);
		arr[k].a4 = std::stod(Line);
		getline(myfile, Line);
		arr[k].a5 = std::stod(Line);
		getline(myfile, Line);
		arr[k].a6 = std::stod(Line);
		getline(myfile, Line);
		arr[k].a7 = std::stod(Line);
		getline(myfile, Line);
		arr[k].a8 = std::stod(Line);
		getline(myfile, Line);
		arr[k].val = std::stoi(Line);
		std::cout << k <<std::endl<< arr[k].val << std::endl;
		k++;

	}
	return 1;
}

int main(int argc, char* argv[])
{

	const int SizeOfArr = 476;
	int KnnDriver = 0;
	Point arr[SizeOfArr];
	Point PointToTest[50]{};
	bool training = true;
	int NumOfFrames;
	int k = 5;
	int Input = -1;


	gflags::ParseCommandLineFlags(&argc, &argv, true);

	std::cout << "Enter 1 for Re-Train the data, 0 to use the trained data" "\n";
	std::cin >> Input;
	
	if (Input == 1)
	{
		
		FLAGS_image_path = ""; // Use the path of the Violence DataSet
		NumOfFrames = 110;

		Start(NumOfFrames, training, arr, SizeOfArr, PointToTest, KnnDriver, k, 1);

		
		FLAGS_image_path = ""; // Use the path of the NON-Violence DataSet
		NumOfFrames = 77;

		Start(NumOfFrames, training, arr, SizeOfArr, PointToTest, KnnDriver, k, 0);
	}
	if (Input == 0)
	{
		ImportTheData(arr);
	}

	
	
	FLAGS_image_path = ""; // Use the path of the input video frames
	training = false;
	NumOfFrames = 600;

	
	Start(NumOfFrames, training, arr, SizeOfArr, PointToTest, KnnDriver, k, 3);

	std::cout << vio << ", " << non;

}