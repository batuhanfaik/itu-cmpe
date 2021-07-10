#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>

using namespace std;

struct Point{
    int xCoordinate;
    int yCoordinate;
};

class PointSpace{
    int numberOfPoints;
    vector<Point> points;

public:
    void setNumberOfPoints(int n){
        numberOfPoints = n;
    }
    
    int getNumberOfPoints(){
        return numberOfPoints;
    }
    
    void addPoint(int x, int y){
        Point p = Point();
        p.xCoordinate = x;
        p.yCoordinate = y;
        
        points.push_back(p);
    }
    
    
    void printNumberOfPoints(){
        cout << "Total number of points: " << getNumberOfPoints() << endl;
    }
    
    void printAllPoints(){
        cout << "Points coordinates (x y): " << endl;
        
        for (std::vector<Point>::const_iterator i = points.begin(); i != points.end(); ++i){
            cout << i->xCoordinate << "\t" << i->yCoordinate << endl;
        }
    }
    
      
   
    
    double FindClosestPairDistance(){
        // FILL HERE
        
        // You may define new functions if you need
    }
       
    
};



int main(int argc, char* argv[]){
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
    while (infile >> x >> y){
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



