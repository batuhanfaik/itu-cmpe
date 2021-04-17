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

// Vertex name and index
typedef pair<string, int> spot;
// Vertex (spot) and weight
typedef pair<spot, int> weighted_spot;

class Edge {
 public:
  spot spot1, spot2;
  int weight;
  Edge(const spot &, const spot &, int);
  bool close_to_enemy();
};

Edge::Edge(const spot &s1, const spot &s2, int wt) {
  spot1 = s1;
  spot2 = s2;
  weight = wt;
}

bool Edge::close_to_enemy(){
  if ((spot1.first.find("E") != string::npos) || (spot2.first.find("E") != string::npos))
    return weight < 5;
  else
    return false;
}

class Graph {
  int num_nodes;
  vector<vector<weighted_spot>> adjacency;
  vector<int> distances;
 public:
  Graph(int);
  void add_edge(Edge &);
  void dijkstras_sp(spot);
  void print_shortest_path(spot);
};

Graph::Graph(int n): num_nodes(n), adjacency(n, vector<weighted_spot>(n)), distances(n, INF){}

void Graph::add_edge(Edge& edge) {
  adjacency[edge.spot1.second].push_back(make_pair(edge.spot2, edge.weight));
  adjacency[edge.spot2.second].push_back(make_pair(edge.spot1, edge.weight));
}

void Graph::dijkstras_sp(spot s_v) {
  // s_v = starting vertex (spot) - Ma
  vector<spot> visited;
  priority_queue<weighted_spot, vector<weighted_spot>, greater<>> p_queue;

  p_queue.push(make_pair(s_v, 0));
  distances[s_v.second] = 0;

  while (!p_queue.empty()) {
    spot u = p_queue.top().first;
    p_queue.pop();
    bool enemy_spot = (u.first.find("E") != string::npos);
    bool close_to_enemy = (u.second == -1);
    if (!enemy_spot && !close_to_enemy){
      visited.push_back(u);

      for (int i = 0; i < adjacency[u.second].size(); i++) {
        spot v = adjacency[u.second][i].first;
        int w = adjacency[u.second][i].second;

        // if the distance to v is shorter by going through u
        if (distances[v.second] > distances[u.second] + w) {
          if (v.second == num_nodes - 2){
            cout << u.first << " " << endl;
          }
          distances[v.second] = distances[u.second] + w;
          p_queue.push(make_pair(v, distances[v.second]));
        }
      }
    }
  }

  for (int i = 0; i < visited.size(); i++) {
//    cout << i << " " << distances[i] << endl;
    cout << i << " " << visited[i].first << "\t" << visited[i].second << "\t" << distances[visited[i].second] << endl;
  }
}

void Graph::print_shortest_path(spot d_v) {
  // d_v = destination vertex (spot) - Mo
  cout << distances[d_v.second] << endl;
}

int main() {
  // Open the file
  string filename;
//  cin >> filename;
  filename = "path_info_2.txt";
  ifstream q2_file(filename);
  if (!q2_file) {
    cerr << "File cannot be opened!";
    exit(1);
  }

  // Declare variables
  map<string, int> g_v;
  map<int, string> g_vi;
  string v1, v2, linebreak;
  int weight, n_vertices;
  n_vertices = 0;
  vector<Edge> edges;
  vector<int> blacklist;

  // Read the file
  while (!q2_file.eof()) {
    getline(q2_file, v1, ',');
    getline(q2_file, v2, ',');
    q2_file >> weight;
    getline(q2_file, linebreak, '\n'); //this is for reading the \n character into dummy variable

    // if vertices are not in the map, add them
    if (g_v.find(v1) == g_v.end()){
      g_v.insert(pair<string, int>(v1, n_vertices));
      g_vi.insert(pair<int, string>(n_vertices++, v1));
    }
    if (g_v.find(v2) == g_v.end()){
      g_v.insert(pair<string, int>(v2, n_vertices));
      g_vi.insert(pair<int, string>(n_vertices++, v2));
    }

    Edge e = Edge(pair<string, int>(v1, g_v[v1]), pair<string, int>(v2, g_v[v2]), weight);
    // Don't add edges that don't satisfy the constraints
    if (e.close_to_enemy()) {
      blacklist.push_back(g_v[v1]);
      blacklist.push_back(g_v[v2]);
    } else {
      if (!count(blacklist.begin(), blacklist.end(), g_v[v1]) && !count(blacklist.begin(), blacklist.end(), g_v[v2]))
        edges.push_back(e);
    }
  }

  int mankara = g_v["Ma"];
  int monstantinople = g_v["Mo"];

  Graph conquerors_path = Graph(g_vi.size());
  for (auto &e : edges) {
    conquerors_path.add_edge(e);
  }
  conquerors_path.dijkstras_sp(make_pair("Ma", mankara));
  conquerors_path.print_shortest_path(make_pair("Mo", monstantinople));

  q2_file.close();
  return 0;
}
