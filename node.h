#ifndef HW1_NODE_H
#define HW1_NODE_H
#include <vector>
#include <iostream>

using namespace std;

class Node {
  int distinct_letter_amount;
  vector<vector<int>> data;
  Node *parent;
  vector<Node*> children;
public:
  Node(int);

  Node(int, Node*, int, int);

  void set_children(vector<Node*>);

  vector<vector<int>> get_data();

  void print();
};


#endif //HW1_NODE_H
