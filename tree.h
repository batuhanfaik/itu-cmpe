#ifndef HW1_TREE_H
#define HW1_TREE_H
#include "node.h"
#include <string>

using namespace std;

class Tree {
  Node* root;
  string letters;
  void populate(int, Node*);
public:
  Tree(string);

  void print();
};


#endif //HW1_TREE_H
