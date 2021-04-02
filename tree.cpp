#include "tree.h"
#include <bits/stdc++.h>

Tree::Tree(const string& o1, const string& o2, const string& r) {
  this->operand1 = o1;
  this->operand2 = o2;
  this->result = r;
  this->letters = get_distinct_letters();

  // Map the operands
  this->operand1_map = map_operand(operand1);
  this->operand2_map = map_operand(operand2);
  this->result_map = map_operand(result);

  // Get distinct letter amount
  int dla = letters.length();
  this->root = new Node(dla);

  // Fill the tree
  populate(0, root);
}

// Get distinct letters of the operands
string Tree::get_distinct_letters() {
  // Concatenate operands and the result
  string concat = operand1 + operand2 + result;
  transform(concat.begin(), concat.end(), concat.begin(), ::toupper); // Uppercase
  // Find distinct letters
  string distinct_letters;
  for (char i : concat) {
    if (distinct_letters.find(i) == string::npos) {
      distinct_letters += i;
    }
  }
  return distinct_letters;
}

// Map operands to the letters
vector<int> Tree::map_operand(string op) {
  vector<int> op_map = vector<int>(op.length());
  for (int i = 0; i < operand1.length(); i++) {
    for (int j = 0; j < letters.length(); j++) {
      if (op[i] == letters[j])
        op_map[i] = j;
    }
  }
  return op_map;
}

// Populate the tree with all possible permutations
void Tree::populate(int level, Node *parent) {
  if (level < letters.length()) {
    vector<Node*> children;
    // Create all ten children and push them
    for (int i = 0; i < 10; i++) {
      Node *child = new Node(letters.length(), parent, level, i);
      check_solution(*child);
      children.push_back(child);
    }
    parent->set_children(children);
    for (int i = 0; i < 10; i++) {
      populate(level + 1, children[i]);
    }
  }
}

// Check if the node satisfies constraints of a possible solution
bool Tree::satisfies_constraints(Node& node) {
  // Make a copy of the data matrix
  vector<vector<int>> data = node.get_data();

  // constraint 1 - first letters of operands are not zero
  if (data[operand1_map[0]][0] || data[operand2_map[0]][0] || data[result_map[0]][0])
    return false;

  // constraint 2 - different letters are mapped to different numbers
  for (int i = 0; i < 10; i++) {
    int sum = 0;
    for (int j = 0; j < letters.length(); j++) {
      sum += data[j][i];
      if (sum > 1) {
        return false;
      }
    }
  }

  // Constraints satisfied, return true
  return true;
}
bool Tree::check_solution(Node &node) {
  // Make a copy of the data matrix
  vector<vector<int>> data = node.get_data();
  // Make sure a number is assigned to every letter (node is leaf)
  int sum = 0;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < letters.length(); j++) {
      sum += data[j][i];
    }
  }
  if (sum != letters.length()) {
    return false;
  }

  // Get the values of the letters of the node
  vector<int> letter_values = vector<int>(letters.length());
  for (int i = 0; i < letters.length(); i++) {
    int j = 0;
    while (j < 10) {
      if (data[i][j] == 1) {
        letter_values[i] = j;
        j = 9;  // I could use break here but what the heck
      }
      j++;
    }
  }

  // Create the carry vector
  int carry_length, shorter_operand, shorter_operand_length;
  // Find the length of the carry array
  if (operand1.length() > operand2.length()) {
    carry_length = operand1.length();
    shorter_operand = 2;
    shorter_operand_length = operand2.length();
  } else {
    carry_length = operand2.length();
    shorter_operand = 1;
    shorter_operand_length = operand1.length();
  }
  vector<int> carry = vector<int>(carry_length, 0);

  // Calculate carries
  for (int i = 0; i < shorter_operand_length; i++) {
    if (i == 0)
      carry[i] = (letter_values[operand1_map[i]] + letter_values[operand2_map[i]]) / 10;
    else
      carry[i] = (letter_values[operand1_map[i]] + letter_values[operand2_map[i]] + carry[i - 1]) / 10;
  }
  // Handle the case where operands are not the same length
  for (int i = shorter_operand_length; i < carry_length; i++) {
    if (shorter_operand == 2) {
      carry[i] = (letter_values[operand1_map[i]] + carry[i - 1]) / 10;
    } else {
      carry[i] = (letter_values[operand2_map[i]] + carry[i - 1]) / 10;
    }
  }
  // Finally check the summation
  if (letter_values.back() != carry.back()) // Last carry is not equal to the result's highest order
    return false;
  for (int i = 0; i < shorter_operand_length; i++) {
    if (letter_values[operand1_map[i]] + letter_values[operand2_map[i]] != letter_values[result_map[i]] + carry[i]*10)
      return false;
  }
  for (int i = shorter_operand_length; i < carry_length; i++) {
    if (shorter_operand == 2) {
      if (letter_values[operand1_map[i]] != letter_values[result_map[i]] + carry[i]*10)
        return false;
    } else {
      if (letter_values[operand2_map[i]] != letter_values[result_map[i]] + carry[i]*10)
        return false;
    }
  }
  // Solution found!
  return true;
}

