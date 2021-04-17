/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * /
* @ Filename: q1
* @ Date: 16-Apr-2021
* @ AUTHOR: batuhanfaik
* @ Copyright (C) 2021 Batuhan Faik Derinbay
* @ Project: hw2
* @ Description: Kruskal's Algorithm for HW2
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <iostream>
#include <fstream>
#include <map>
#include <utility>
#include <vector>
#include <algorithm>
#include <iterator>
#include <bits/stdc++.h>

#define INF 0x7FFFFFFF

using namespace std;

typedef pair<string, int> spot;
typedef pair<spot, int> weighted_spot;
typedef pair<int, int> index_pair;

class Edge {
 public:
  spot spot1, spot2;
  int weight;
  Edge(const spot &, const spot &, int);
};

Edge::Edge(const spot &s1, const spot &s2, int wt) {
  spot1 = s1;
  spot2 = s2;
  weight = wt;
}

class Graph {
  int num_nodes;
  list<weighted_spot>* adjacency;
 public:
  Graph(int);
  void add_edge(Edge&);
  void dijkstras_sp(spot);
//  ~Graph();
};

Graph::Graph(int n) {
  num_nodes = n;
  adjacency = new list<weighted_spot>[n];
}

void Graph::add_edge(Edge& edge) {
  weighted_spot s1_w = make_pair(edge.spot1, edge.weight);
  weighted_spot s2_w = make_pair(edge.spot2, edge.weight);
  adjacency[edge.spot1.second].push_back(s2_w);
  adjacency[edge.spot2.second].push_back(s1_w);
}

void Graph::dijkstras_sp(spot s_v) {
  // s_v = starting vertex
  priority_queue<index_pair, vector<index_pair>, greater<>> queue;
  vector<weighted_spot> distance(num_nodes, make_pair(s_v, INF));
  queue.push(make_pair(0, s_v.second));
  distance[s_v.second] = make_pair(s_v, 0);

  while (!queue.empty()) {
    int u = queue.top().second;
    queue.pop();

    list<weighted_spot>::iterator it;
    for (it = adjacency[u].begin(); it != adjacency[u].end(); it++){
      int v = (*it).first.second;
      int weight = (*it).second;

      if (distance[v].second > distance[u].second + weight){
        distance[v].second = distance[u].second + weight;
        queue.push(make_pair(distance[v].second, v));
      }
    }
  }
  // Print shortest distances stored in dist[]
  printf("Vertex   Distance from Source\n");
  for (int i = 0; i < num_nodes; ++i)
    printf("%d \t\t %d\n", i, distance[i].second);
}

int main() {
  // Open the file
  string filename;
//  cin >> filename;
  filename = "path_info_1.txt";
  ifstream q2_file(filename);
  if (!q2_file) {
    cerr << "File cannot be opened!";
    exit(1);
  }

  // Declare variables
  map<string, int> g_v;
  string v1, v2, linebreak;
  int weight, n_vertices;
  n_vertices = 0;
  vector<Edge> edges;

  // Read the file
  while (!q2_file.eof()) {
    getline(q2_file, v1, ',');
    getline(q2_file, v2, ',');
    q2_file >> weight;
    getline(q2_file, linebreak, '\n'); //this is for reading the \n character into dummy variable

    // if vertices are not in the map, add them
    if (g_v.find(v1) == g_v.end()) {
      g_v.insert(pair<string, int>(v1, n_vertices++));
    }
    if (g_v.find(v2) == g_v.end()) {
      g_v.insert(pair<string, int>(v2, n_vertices++));
    }

    Edge e = Edge(pair<string, int>(v1, g_v[v1]), pair<string, int>(v2, g_v[v2]), weight);
    // Don't add edges that don't satisfy the constraints
//    if (!e.connects_hipp_bas() && !e.connects_hp_hp())
    edges.push_back(e);
  }

  Graph conquerors_path = Graph(g_v.size());
  for (auto& e : edges) {
    conquerors_path.add_edge(e);
  }
  conquerors_path.dijkstras_sp(edges[0].spot1);

  q2_file.close();
  return 0;
}
