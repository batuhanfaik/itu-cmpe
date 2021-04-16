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
#include<algorithm>
#include <iterator>

using namespace std;

class Edge {
 public:
  pair<string, int> start_node, end_node;
  int weight;
  Edge(const pair<string, int> &, const pair<string, int> &, int);
};

Edge::Edge(const pair<string, int> &sn, const pair<string, int> &en, int wt) {
  start_node = sn;
  end_node = en;
  weight = wt;
}

class Graph {
  int num_nodes;
  vector<Edge> edges;
  vector<pair<string, int>> parent;
  vector<int> rank;
  vector<Edge> mst;
 public:
  Graph(int);
  void set_edges(vector<Edge>);
  pair<string, int> get_parent(pair<string, int>) const;
  static bool compare_weight(const Edge &, const Edge &);
  void print_mst();
  void kruskals_mst();
};

Graph::Graph(int n) {
  num_nodes = n;
  parent.resize(n);
  rank.resize(n);
}

void Graph::set_edges(vector<Edge> e) {
  edges = std::move(e);
}

pair<string, int> Graph::get_parent(pair<string, int> node) const {
  if (parent[node.second] == node) {
    return node;
  } else {
    return get_parent(parent[node.second]);
  }
}

bool Graph::compare_weight(const Edge &a, const Edge &b) {
  return a.weight < b.weight;
}

void Graph :: print_mst() {
  int cost = 0;
  cout << "Edges of minimum spanning tree : ";
  for(auto& e : mst) {
    cout << "[" << e.start_node.first << "-" << e.end_node.first << "](" << e.weight << ") ";
    cost += e.weight;
  }
  cout << endl << "Cost of minimum spanning tree : " << cost << endl;
}

void Graph::kruskals_mst() {
  for (int i = 0; i < num_nodes; i++) {
    parent[i].second = i;
    rank[i] = 0;
  }

  // Sort edges in ascending order
  sort(edges.begin(), edges.end(), compare_weight);

  // Implementation using union-set structure
  // Note that rather than creating a separate class for the disjoint set union data structure
  // I prefer to implement it within the Kruskal's algorithm.
  for (auto &edge: edges) {
    pair<string, int> root1 = get_parent(edge.start_node);
    pair<string, int> root2 = get_parent(edge.end_node);

    if (root1 != root2) {
      mst.push_back(edge);
      if (rank[root1.second] < rank[root2.second]) {
        parent[root1.second] = root2;
        rank[root1.second]++;
      } else {
        parent[root2.second] = root1;
        rank[root2.second]++;
      }
    }
  }
}

int main() {
  // Open the file
  ifstream q1_file("city_plan_1.txt");    // input.txt has integers, one per line
  if (!q1_file) {
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
  while (!q1_file.eof()) {
    getline(q1_file, v1, ',');
    getline(q1_file, v2, ',');
    q1_file >> weight;
    getline(q1_file, linebreak, '\n'); //this is for reading the \n character into dummy variable

    // if vertices are not in the map, add them
    if (g_v.find(v1) == g_v.end()) {
      g_v.insert(pair<string, int>(v1, n_vertices++));
    }
    if (g_v.find(v2) == g_v.end()) {
      g_v.insert(pair<string, int>(v2, n_vertices++));
    }

    Edge e = Edge(pair<string, int>(v1, g_v[v1]), pair<string, int>(v2, g_v[v2]), weight);
    edges.push_back(e);
  }

  Graph monstantinople = Graph(g_v.size());
  monstantinople.set_edges(edges);
  monstantinople.kruskals_mst();
  monstantinople.print_mst();

  q1_file.close();
  return 0;
}
