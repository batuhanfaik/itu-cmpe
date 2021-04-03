#ifndef HW1_TREE_H
#define HW1_TREE_H
#include "node.h"
#include <string>

using namespace std;

class Tree {
  int node_amount;
  int nodes_in_memory;
  Node* root;
  string letters;
  string operand1, operand2, result;
  vector<int> operand1_map, operand2_map, result_map;
  string solution, solution_to_file;

  string get_distinct_letters();
  vector<int> map_operand(string);
  void populate(int, Node*);
  bool satisfies_constraints(Node&);
  bool check_solution(Node&);
 public:
  Tree(const string&, const string&, const string&);

  int bfs();

  int dfs();

  int get_nodes_in_memory() const;

  string get_solution() const;

  string get_solution_to_file() const;
};


#endif //HW1_TREE_H
