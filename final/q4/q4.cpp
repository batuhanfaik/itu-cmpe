/* * * * * * * * * * * * * * * * * * * * * * * * * *
 * BLG 336E Analysis of Algorithms II
 * Spring 2021
 * Student Name: NESE GUNES
 * Student ID: 504192555
 * * * * * * * * * * * * * * * * * * * * * * * * * */
/***********************************************************
STUDENT INFORMATION
Full Name : Batuhan Faik Derinbay
Student ID:  150180705
**********************************************************/
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;

class Solution {
  // NONE = 0, BLACK = 1, WHITE = 2
  enum Color { NONE, BLACK, WHITE };

  void colorDFS(vector<vector<int>>& graph, vector<int>& colored, int idx, Color a, Color b, bool& isBipartite) {
    // Stop as soon as no longer bipartite property is satisfied
    if (!isBipartite)
      return;
    // DFS
    int vSize = int(graph[idx].size());
    for (int i = 0; i < vSize; i++) {
      // If neighbours are the same color, graph is not bipartite
      if (colored[graph[idx][i]] == a) {
        isBipartite = false;
        return;
      }
      // If a node is not colored, color it opposite to its neighbour and recurse
      if (colored[graph[idx][i]] == NONE) {
        colored[graph[idx][i]] = b;
        colorDFS(graph, colored, graph[idx][i], b, a, isBipartite);
      }
    }
  }
 public:
  bool isBipartite(vector<vector<int>> &graph) {
    bool isBipartite = true;
    int N = int(graph.size());
    // Check graph size
    if (N < 3)
      return true;

    // Create coloring vector
    vector<int> colored(N, NONE);
    for (int i = 0; i < N; i++) {
      if (colored[i] == NONE) {
        colored[i] = BLACK;
        colorDFS(graph, colored, i, BLACK, WHITE, isBipartite);
      }
    }

    return isBipartite;
  }
};

// This function is provided to check if your graph data is stored well or not
void printv(vector<vector<int> > v) {
  for (int i = 0; i < v.size(); i++) {
    for (int j = 0; j < v[i].size(); j++)
      cout << v[i][j] << " ";
    cout << endl;
  }
}

int main() {
  // Solution class for bipartite-ness problem
  Solution s;

  // Read data filename from std input stream and open it with file handler/pointer
  string fname;
  cin >> fname;
  ifstream graphFileHandler(fname);

  // An array of integers to store neighbours of a vertex
  vector<int> adj;
  // A 2D array of integers to store neighbours of each vertex in a graph
  vector<vector<int> > graph;

  string line;
  // Read from the file until there is no new line left
  while (graphFileHandler >> line) {
    // Save line as string stream object
    stringstream ss(line);
    // Get a string variable
    string substr;

    // Until the end of the line, read the line as substings separated by commas
    while (ss.good()) {
      getline(ss, substr, ',');
      // Push the substring in adjacency list, we got a neighbour here
      adj.push_back(stoi(substr));
    }

    // When line ends, push all the neighbours of the vertex into the graph
    graph.push_back(adj);

    // Clear the array, before the next iteration begins
    // Next iteration, we will read the subsequent line from the file and it will contain neighbours of some other guy
    adj.clear();
  }

  // Check if data file is read in the way it is supposed to be
  // D O   N O T   P R I N T   T H E   G R A P H   I N   Y O U R   S O L U T I O N
  // The expected output only includes a True or a False not the graph data itself
  // Do not uncomment this line
  printv(graph);

  // Save the return value of the function
  bool answer = s.isBipartite(graph);

  // If answer is True, print True otherwise False
  if (answer == 1)
    cout << "True" << endl;
  else
    cout << "False" << endl;

  return 0;
}
