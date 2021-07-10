/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * /
* @ Filename: q2
* @ Date: 16-Apr-2021
* @ AUTHOR: batuhanfaik
* @ Copyright (C) 2021 Batuhan Faik Derinbay
* @ Project: hw2
* @ Description: Dijkstra's Algorithm for HW2
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <iostream>
#include <map>
#include <utility>
#include <vector>
#include <algorithm>
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
  bool close_to_enemy() const;
};

Edge::Edge(const spot &s1, const spot &s2, int wt) {
  spot1 = s1;
  spot2 = s2;
  weight = wt;
}

bool Edge::close_to_enemy() const{
  if ((spot1.first.find("E") != string::npos) || (spot2.first.find("E") != string::npos))
    return weight < 5;
  else
    return false;
}

class Graph {
  int num_nodes;
  vector<vector<weighted_spot>> adjacency;
  vector<int> distances;
  vector<int> visited;
  map<string, int> g_v;
  map<int, string> g_vi;
 public:
  Graph(int, map<string, int>, map<int, string>);
  void add_edge(Edge &);
  void dijkstras_sp(spot);
  void print_path(int);
  void print_shortest_path(int);
};

Graph::Graph(int n, map<string, int> g_v, map<int, string> g_vi): num_nodes(n),
  adjacency(n, vector<weighted_spot>(n)), distances(n, INF), visited(n, -1), g_v(g_v), g_vi(g_vi){}

void Graph::add_edge(Edge& edge) {
  adjacency[edge.spot1.second].push_back(make_pair(edge.spot2, edge.weight));
  adjacency[edge.spot2.second].push_back(make_pair(edge.spot1, edge.weight));
}

void Graph::dijkstras_sp(spot s_v) {
  // s_v = starting vertex (spot) - Ma
  priority_queue<weighted_spot, vector<weighted_spot>, greater<weighted_spot>> p_queue;

  p_queue.push(make_pair(s_v, 0));
  distances[s_v.second] = 0;

  while (!p_queue.empty()) {
    spot u = p_queue.top().first;
    p_queue.pop();

    bool enemy_spot = (u.first.find("E") != string::npos);
    bool close_to_enemy = (u.second == -1);
    if (!enemy_spot && !close_to_enemy){
      int adj_size = adjacency[u.second].size();
      for (int i = 0; i < adj_size; i++) {
        spot v = adjacency[u.second][i].first;
        int w = adjacency[u.second][i].second;

        // if the distance to v is shorter by going through u
        if (distances[v.second] > distances[u.second] + w) {
          visited[v.second] = u.second;
          distances[v.second] = distances[u.second] + w;
          p_queue.push(make_pair(v, distances[v.second]));
        }
      }
    }
  }
}

void Graph::print_path(int in) {
  if (visited[in] == -1) {
    return;
  }
  print_path(visited[in]);
  cout << g_vi[in] << " ";
}

void Graph::print_shortest_path(int end_vertex_index) {
  cout << "Ma ";
  print_path(end_vertex_index);
  cout << distances[end_vertex_index] << endl;
}

int main() {
  // Open the file
  string filename;
  cin >> filename;
//  filename = "path_info_2.txt";
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

  Graph conquerors_path = Graph(g_v.size(), g_v, g_vi);
  for (auto &e : edges) {
    conquerors_path.add_edge(e);
  }
  conquerors_path.dijkstras_sp(make_pair("Ma", mankara));
  conquerors_path.print_shortest_path(monstantinople);

  q2_file.close();
  return 0;
}
