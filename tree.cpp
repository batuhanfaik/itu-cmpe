#include "tree.h"

Tree::Tree(string letters) {
  this->letters = letters;
  int dla = letters.length();
  this->root = new Node(dla);

  // Fill the tree
  populate(0, root);
}

// Populate the tree with all possible permutations
void Tree::populate(int level, Node *parent) {
  if (level < letters.length()) {
    vector<Node*> children;
    // Create all ten children and push them
    for (int i = 0; i < 10; i++) {
      Node *child = new Node(letters.length(), parent, level, i);
      child->print();
      children.push_back(child);
    }
    parent->set_children(children);
    for (int i = 0; i < 10; i++) {
      populate(level + 1, children[i]);
    }
  }
}
