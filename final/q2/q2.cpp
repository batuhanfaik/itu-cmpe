/***********************************************************
STUDENT INFORMATION
Full Name : Batuhan Faik Derinbay
Student ID:  150180705
**********************************************************/

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>

using namespace std;

struct Point {
  int xCoordinate;
  int yCoordinate;
};

class PointSpace {
  int numberOfPoints;
  vector<Point> points;

 public:
  void setNumberOfPoints(int n) {
    numberOfPoints = n;
  }

  int getNumberOfPoints() {
    return numberOfPoints;
  }

  void addPoint(int x, int y) {
    Point p = Point();
    p.xCoordinate = x;
    p.yCoordinate = y;

    points.push_back(p);
  }

  void printNumberOfPoints() {
    cout << "Total number of points: " << getNumberOfPoints() << endl;
  }

  void printAllPoints() {
    cout << "Points coordinates (x y): " << endl;

    for (std::vector<Point>::const_iterator i = points.begin(); i != points.end(); ++i) {
      cout << i->xCoordinate << "\t" << i->yCoordinate << endl;
    }
  }

  static bool compareX(const Point &p1, const Point &p2) {
    return (p1.xCoordinate < p2.xCoordinate);
  }

  static bool compareY(const Point &p1, const Point &p2) {
    return (p1.yCoordinate < p2.yCoordinate);
  }

  static double distance(const Point &p1, const Point &p2) {
    return sqrt(pow((p1.xCoordinate - p2.xCoordinate), 2) + pow((p1.yCoordinate - p2.yCoordinate), 2));
  }

  static double distance(const pair<Point, Point> p) {
    Point p1 = p.first;
    Point p2 = p.second;
    return sqrt(pow((p1.xCoordinate - p2.xCoordinate), 2) + pow((p1.yCoordinate - p2.yCoordinate), 2));
  }

  static double distanceX(const Point &p1, const Point &p2) {
    return (abs(p1.xCoordinate - p2.xCoordinate));
  }

  // Get the pair with minimum distance
  static pair<Point, Point> pairwiseDistance(const vector<Point> &p) {
    double min = distance(p[0], p[1]);
    pair<Point, Point> minPair = make_pair(p[0], p[1]);
    for (int i = 0; i < p.size(); i++) {
      for (int j = i+1; j < p.size(); j++) {
        if (distance(p[i], p[j]) < min)
          minPair = make_pair(p[i], p[j]);
      }
    }
    return minPair;
  }

  pair<Point, Point> closestPairRec(const vector<Point> &px, const vector<Point> &py) {
    int len = int(px.size());
    if (len <= 3)
      return pairwiseDistance(px);

    // Split the list in the middle
    // First part gets the remainder :)
    int mid = int(ceil(float(len) / 2)) - 1;

    // Construct Qx, Qy, Rx, Ry arrays
    vector<Point> qx, qy, rx, ry;
    for (int i = 0; i < len; i++) {
      if (i < mid) {
        qx.push_back(px[i]);
        qy.push_back(py[i]);
      } else {
        rx.push_back(px[i]);
        ry.push_back(py[i]);
      }
    }

    pair<Point, Point> qPair = closestPairRec(qx, qy);
    pair<Point, Point> rPair = closestPairRec(rx, ry);

    double delta = min(distance(qPair), distance(rPair));
    // L is actually a set of points where x = xStar but we don't need the line here
    // We only need the x coord of xStar
    Point xStar = qx.back();
    // Construct Sy with points closer to mid line than delta
    vector<Point> sy;
    for (int i = 0; i < len; i++) {
      if (distanceX(px[i], xStar) < delta) {
        sy.push_back(px[i]);
      }
    }
    // Sort sy in ascending order of y coords
    sort(sy.begin(), sy.end(), compareY);
    pair<Point, Point> sPair = pairwiseDistance(sy);

    if (distance(sPair) < delta)
      return sPair;
    else if (distance(qPair) < distance(rPair))
      return qPair;
    else
      return rPair;
  }

  double FindClosestPairDistance() {
    // Create Px, Py from P
    vector<Point> px = points;
    vector<Point> py = points;
    sort(px.begin(), px.end(), compareX);
    sort(py.begin(), py.end(), compareY);

    // Get the closest pair
    auto closestPair = closestPairRec(px, py);

    return distance(closestPair);
  }
};

int main(int argc, char *argv[]) {
  //define a point space
  PointSpace pointSpace;

  //get file name
  string inputFileName = argv[1];

  string line;
  ifstream infile(inputFileName);

  //read the number of total points (first line)
  getline(infile, line);
  pointSpace.setNumberOfPoints(stoi(line));

  //read points' coordinates and assign each point to the space (second to last line)
  int x, y;
  while (infile >> x >> y) {
    pointSpace.addPoint(x, y);
  }

  //print details of point space (not necessary for assignment evaluation: calico will not check this part)
  pointSpace.printAllPoints();
  pointSpace.printNumberOfPoints();

  //find and print the distance between closest pair of points (calico will check this part)
  double closestDistance = pointSpace.FindClosestPairDistance();
  cout << "Distance between the closest points: " << closestDistance << endl;

  return 0;
}



