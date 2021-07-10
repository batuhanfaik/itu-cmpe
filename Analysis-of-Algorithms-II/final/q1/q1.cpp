/***********************************************************
2021 Spring - BLG 336E-Analysis of Algorithms II
Final Project
Question on Greedy Algorithms
Modified Dijkstra Algorithms for Maximum Capacity Path
Submitted: 15.06.2021 
**********************************************************/

/***********************************************************
STUDENT INFORMATION
Full Name : Batuhan Faik Derinbay
Student ID:  150180705
**********************************************************/

// Some of the libraries you may need have already been included.
// If you need additional libraries, feel free to add them
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>

// Do not change this definition
#define INT_MAX 1000

using namespace std;

class Graph {
 public:
  int node_count;
  int edge_count;
  int **adjacency_matrix;

  Graph() {};
  void read_file(char *filename);
  void print_adjacency_matrix(); // in case you need
  int weight(int i, int j) { return this->adjacency_matrix[i][j]; }
  ~Graph();
};

Graph::~Graph() {
  for (int i = 0; i < node_count; i++) {
    delete[] adjacency_matrix[i];
  }
  delete[] adjacency_matrix;
}

void Graph::print_adjacency_matrix() {
  // Prints the adjacency matrix
  for (int i = 0; i < this->node_count; i++) {
    for (int j = 0; j < this->node_count; j++) {
      cout << this->adjacency_matrix[i][j] << ", ";
    }
    cout << endl;
  }
}

void Graph::read_file(char *filename) {
  /*********************************************/
  /****** CODE HERE TO READ THE TEXT FILE ******/
  /*********************************************/

  ifstream file;
  file.open(filename);
  if (!file) {
    cerr << "File cannot be opened!";
    exit(1);
  }
  string line;
  int v1, v2, w;

  // Read the number of vertices
  file >> this->node_count;
  getline(file, line);
  // Construct the adjacency matrix
  this->adjacency_matrix = new int *[this->node_count];
  for (int i = 0; i < this->node_count; i++)
    this->adjacency_matrix[i] = new int[this->node_count];
  // Read first N lines
  while (!file.eof()) {
    // Read values of the Sale object
    file >> v1; file >> v2; file >> w;
    getline(file, line, '\n'); //this is for reading the \n character into dummy variable.

    this->adjacency_matrix[v1][v2] = w;
    this->adjacency_matrix[v2][v1] = w;
  }
}

void Modified_Dijkstra(Graph *graph) {

  /*********************************************/
  /****** CODE HERE TO FOR THE ALGORITHM *******/
  /*********************************************/
  int nc = graph->node_count;
  int s = 0;
  int status[nc], wt[nc], dad[nc];

  // Fill arrays
  for (int v = 0; v < nc; v++) {
    status[v] = 0; wt[v] = -1; dad[v] = -1;
  }
  status[s] = 2; wt[s] = INT_MAX;
  // Update fringes
  for (int w = 1; w < nc; w++) {
    status[w] = 1; wt[w] = graph->weight(s, w); dad[w] = s;
  }

  bool fringe_exists = true;
  while (fringe_exists) {
    int v = 0;
    int max_fringe = 0;
    // Check if fringes exist and update v
    fringe_exists = false;
    for (int i = 0; i < nc; i++) {
      if (status[i] == 1 && wt[i] > max_fringe) {
        max_fringe = wt[i]; v = i;
      }
      if (status[i] == 1)
        fringe_exists = true;
    }

    status[v] = 2;
    for (int w = 0; w < nc; w++) {
      if (!status[w]) {
        status[w] = 1;
        wt[w] = min(wt[v], graph->weight(v, w));
        dad[w] = v;
      } else if (status[w] == 1 && wt[w] < min(wt[v], graph->weight(v, w))) {
        wt[w] = min(wt[v], graph->weight(v, w));
        dad[w] = v;
      }
    }
  }

  /*********************************************/
  /***** DO NOT CHANGE THE FOLLOWING LINES *****/
  /**** THEY PRINT OUT THE EXPECTED RESULTS ****/
  /*********************************************/

  // The following line prints wt array (or vector).
  // Do not change anything in the following lines.
  cout << "###########RESULTS###########" << endl;
  cout << endl;

  cout << "1. WT ARRAY" << endl;
  cout << "------------------------" << endl;
  cout << "  ";
  for (int i = 0; i < graph->node_count - 1; i++) {
    cout << wt[i] << ", ";
  }
  cout << wt[graph->node_count - 1] << endl;

  // The following lines print the final path.
  // Do not change anything in the following lines.
  int iterator = graph->node_count - 1;
  vector<int> path_info;
  path_info.push_back(iterator);
  while (iterator != 0) {
    path_info.push_back(dad[iterator]);
    iterator = dad[iterator];
  }
  cout << endl;
  cout << "2. MAXIMUM CAPACITY PATH" << endl;
  cout << "------------------------" << endl;
  cout << "  ";
  vector<int>::iterator it;
  for (it = path_info.end() - 1; it > path_info.begin(); it--)
    cout << *it << " -> ";
  cout << *path_info.begin() << endl;

  cout << endl;
  cout << "3. MAXIMUM CAPACITY" << endl;
  cout << "------------------------" << endl;
  cout << "  ";
  cout << wt[graph->node_count - 1] << endl;
  cout << "#############################" << endl;
}

int main(int argc, char **argv) {
  Graph *graph = new Graph();
  graph->read_file(argv[1]);
  graph->print_adjacency_matrix();
  Modified_Dijkstra(graph);

  return 0;
}
