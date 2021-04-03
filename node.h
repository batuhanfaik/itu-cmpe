#ifndef HW1_NODE_H
#define HW1_NODE_H
#include <vector>
#include <iostream>

using namespace std;

class Node {
  int distinct_letter_amount;
  int index;
  bool is_leaf;
  vector<vector<uint8_t>> data;
  Node *parent;
  vector<Node*> children;
public:
  Node(int);

  Node(int, Node*, int, int);

  vector<Node*> get_children() const;

  void set_children(vector<Node*>);

  int get_index() const;

  bool leaf() const;

  vector<vector<uint8_t>> get_data();

  void print();
};


#endif //HW1_NODE_H
