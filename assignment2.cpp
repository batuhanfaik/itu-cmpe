//
// Created by;
// Batuhan Faik Derinbay 
// 150180705
// on 4/21/19.
//

#include <iostream>
#include <string>

#define MAX_OPERATOR_SIZE 1028

using namespace std;

//CLASS DECLARATIONS
class Operator{
    int center_x;
    int center_y;
    int op_size;
    int op_num;
public:
    Operator(int x, int y, int size); //Default constructor

    void reset(int new_x, int new_y, int new_size); //Reset position

    void set_x(int new_x); //Setter of position x
    int get_x(); //Getter of position x

    void set_y(int new_y); //Setter of position y
    int get_y(); //Getter of position y

    void set_size(int new_size); //Setter of the operator size
    int get_size(); //Getter of the operator size

    void set_num(int); //Setter of the op_num
    int get_num(); //Getter of the op_num
};

class ArithmeticOperator: public Operator {
    char sign;
public:
    ArithmeticOperator(); //Default constructor

    ArithmeticOperator(int x, int y, int size, char sign); //Constructor

    char get_sign(); //Getter method of the sign

    void print_operator(); //Prints out operator's center location, size and sign character
};

class OperatorGrid{
    int grid_rows;
    int grid_cols;
    char **grid;
    ArithmeticOperator* **aop_ptr_grid;

    int num_operators;
    ArithmeticOperator *operators;
public:
    OperatorGrid(int rows, int cols); //Default constructor
    ~OperatorGrid();

    bool place_operator(ArithmeticOperator *); //Places an operator
    bool move_operator(int x, int y, char direction, int move_by); //Moves the operator
    void print_operators(); //Prints the operator
};

//CLASS DEFINITIONS
Operator::Operator(int x, int y, int size) {
    center_x = x;
    center_y = y;
    op_size = size;
    op_num = -1;
}
void Operator::reset(int new_x, int new_y, int new_size) {
    //Reassign all values
    center_x = new_x;
    center_y = new_y;
    op_size = new_size;
}

void Operator::set_x(int new_x) { //Setter of x
    center_x = new_x;
}
int Operator::get_x() { //Getter of x
    return center_x;
}

void Operator::set_y(int new_y) { //Setter of y
    center_y = new_y;
}
int Operator::get_y() { //Getter of y
    return center_y;
}

void Operator::set_size(int new_size) { //Setter of size
    op_size = new_size;
}
int Operator::get_size() { //Getter of size
    return op_size;
}

void Operator::set_num(int num) {
    op_num = num;
}
int Operator::get_num(){
    return op_num;
}

ArithmeticOperator::ArithmeticOperator():Operator(0,0,0),sign('\0') {}

ArithmeticOperator::ArithmeticOperator(int x, int y, int size, char sign):Operator(x,y,size) {
    if (sign == '+' || sign == '-' || sign == 'x' || sign == '/'){
        this->sign = sign;
    }
    else {
        cout << "SIGN parameter is invalid!" << endl;
    }
}

char ArithmeticOperator::get_sign() { //Getter of the sign
    return sign;
}

void ArithmeticOperator::print_operator() {
    cout << "ARITHMETIC_OPERATOR[" << sign << "], CENTER_LOCATION[" << get_x() << "," << get_y()
    << "], SIZE[" << get_size() << "]" << endl;
}

//Default constructor of the OperatorGrid
OperatorGrid::OperatorGrid(int rows, int cols) {
    grid_rows = rows;
    grid_cols = cols;

    //Create a grid 2d array
    grid = new char*[grid_rows];
    for (int i = 0; i < grid_rows; ++i) {
        grid[i] = new char[grid_cols]();
    }

    //ArithmeticOperator Pointer Grid initialization
    aop_ptr_grid = new ArithmeticOperator**[grid_rows];
    for (int i = 0; i < grid_rows; ++i) {
        aop_ptr_grid[i] = new ArithmeticOperator*[grid_cols]();
    }

    operators = new ArithmeticOperator[MAX_OPERATOR_SIZE];     //REQUIRES DYNAMIC MEMORY ALLOCATION
    num_operators = 0;
}

//Destructor of the OperatorGrid
OperatorGrid::~OperatorGrid() {
    //Delete the grid 2d array
    for (int i = 0; i < grid_rows; ++i) {
        delete[] grid[i];
    }
    delete[] grid;
    cout << "DESTRUCTOR: GIVE BACK[" << grid_rows << "," << grid_cols << "] chars." << endl;
    //Delete the operators array
    delete[] operators;
    cout << "DESTRUCTOR: GIVE BACK[" << num_operators << "] Operators." << endl;
}

//Place an operator type ArithmeticOperator in the 2d grid
bool OperatorGrid::place_operator(ArithmeticOperator *current_operator) {
    //IMPROVEMENT: Use switch-case
    if (current_operator->get_sign() == '+'){       //Case sign = '+'
        //Initialize errors and indexing variables
        int is_empty = 1;
        int checked_all = 0;
        int border_error = 0;

        //BORDER ERROR Checking in four directions
        if (current_operator->get_x() - current_operator->get_size() - 1 < 0){        //Direction: North
            border_error = 1;
        } else if (current_operator->get_size() + current_operator->get_y() > grid_cols){       //Direction: East
            border_error = 1;
        } else if (current_operator->get_size() + current_operator->get_x() > grid_rows) {       //Direction: South
            border_error = 1;
        } else if (current_operator->get_y() - current_operator->get_size() - 1 < 0) {       //Direction: West
            border_error = 1;
        }

        //CONFLICT ERROR Checking in four directions
        while (is_empty == 1 && checked_all == 0){
            for (int i = 0; i < current_operator->get_size(); ++i) {
                //Check if the location is inbounds
                if(current_operator->get_x()-i-1 < 0 || current_operator->get_y()+i-1 > grid_cols ||
                current_operator->get_x()+i-1 > grid_rows || current_operator->get_y()-i-1 < 0) {
                    is_empty = 1; //No conflict error rather a border error
                } else if (grid[current_operator->get_x()-i-1][current_operator->get_y()-1] != '\0'){      //Direction: North
                    is_empty = 0;
                } else if (grid[current_operator->get_x()-1][current_operator->get_y()+i-1] != '\0'){       //Direction: East
                    is_empty = 0;
                } else if (grid[current_operator->get_x()+i-1][current_operator->get_y()-1] != '\0'){       //Direction: South
                    is_empty = 0;
                } else if (grid[current_operator->get_x()-1][current_operator->get_y()-i-1] != '\0'){       //Direction: West
                    is_empty = 0;
                }
                if (is_empty == 0){      //Stop the loop if there is a conflict (IMPROVEMENT: Use a while loop)
                    break;
                }
            }
            if (is_empty == 1){ //If all the locations were empty until this point, then we can quit the loop
                checked_all = 1;
            }
        }
        if (border_error == 1){ //If there is a border error, print out the statement
            cout << "BORDER ERROR: Operator " << current_operator->get_sign() << " with size " << current_operator->get_size() << " can not be placed on ("
            << current_operator->get_x() << "," << current_operator->get_y() << ")." << endl;
        }
        // If there is a conflict error, print out the statement
        if (is_empty == 0){ //Meaning that at least one location that we want to place the operator on top is occupied
            cout << "CONFLICT ERROR: Operator " << current_operator->get_sign() << " with size " << current_operator->get_size() << " can not be placed on ("
            << current_operator->get_x() << "," << current_operator->get_y() << ")." << endl;
        }
        //If there are not any errors, place the operator
        if (border_error == 0 && is_empty == 1){
            for (int i = 0; i < current_operator->get_size() + 1; ++i) {
                grid[current_operator->get_x() - i - 1][current_operator->get_y() - 1] = '+';     //Direction: North
                aop_ptr_grid[current_operator->get_x() - i - 1][current_operator->get_y() - 1] = current_operator;
                grid[current_operator->get_x() - 1][current_operator->get_y() + i - 1] = '+';     //Direction: East
                aop_ptr_grid[current_operator->get_x() - 1][current_operator->get_y() + i - 1] = current_operator;
                grid[current_operator->get_x() + i - 1][current_operator->get_y() - 1] = '+';     //Direction: South
                aop_ptr_grid[current_operator->get_x() + i - 1][current_operator->get_y() - 1] = current_operator;
                grid[current_operator->get_x() - 1][current_operator->get_y() - i - 1] = '+';     //Direction: West
                aop_ptr_grid[current_operator->get_x() - 1][current_operator->get_y() - i - 1] = current_operator;
            }
            operators[num_operators] = *current_operator;    //Append the successful placement in to the operators array
            current_operator->set_num(num_operators); //Change the number of the operator
            cout << "SUCCESS: Operator " << current_operator->get_sign() << " with size " << current_operator->get_size() << " is placed on ("
            << current_operator->get_x() << "," << current_operator->get_y() << ")." << endl;
            num_operators++;
            return true; //Successfully placed
        } else {
            return false; //Placement wasn't successful
        }
    } else if (current_operator->get_sign() == '-'){        //Case sign = '-'
        //Initialize errors and indexing variables
        int is_empty = 1;
        int checked_all = 0;
        int border_error = 0;

        //BORDER ERROR Checking in two directions
        if (current_operator->get_size() + current_operator->get_y() > grid_cols){       //Direction: East
            border_error = 1;
        } else if (current_operator->get_y() - current_operator->get_size() - 1 < 0) {       //Direction: West
            border_error = 1;
        }

        //CONFLICT ERROR Checking in two directions
        while (is_empty == 1 && checked_all == 0){
            for (int i = 0; i < current_operator->get_size(); ++i) {
                //Check if the location is inbounds
                if(current_operator->get_y()+i-1 > grid_cols || current_operator->get_y()-i-1 < 0){
                    is_empty = 1; //No conflict error rather a border error
                } else if (grid[current_operator->get_x()-1][current_operator->get_y()+i-1] != '\0'){       //Direction: East
                    is_empty = 0;
                } else if (grid[current_operator->get_x()-1][current_operator->get_y()-i-1] != '\0'){       //Direction: West
                    is_empty = 0;
                }
                if (is_empty == 0){      //Stop the loop if there is a conflict (IMPROVEMENT: Use a while loop)
                    break;
                }
            }
            if (is_empty == 1){ //If all the locations were empty until this point, then we can quit the loop
                checked_all = 1;
            }
        }
        if (border_error == 1){ //If there is a border error, print out the statement
            cout << "BORDER ERROR: Operator " << current_operator->get_sign() << " with size " << current_operator->get_size() << " can not be placed on ("
            << current_operator->get_x() << "," << current_operator->get_y() << ")." << endl;
        }
        // If there is a conflict error, print out the statement
        if (is_empty == 0){ //Meaning that at least one location that we want to place the operator on top is occupied
            cout << "CONFLICT ERROR: Operator " << current_operator->get_sign() << " with size " << current_operator->get_size() << " can not be placed on ("
            << current_operator->get_x() << "," << current_operator->get_y() << ")." << endl;
        }
        //If there are not any errors, place the operator
        if (border_error == 0 && is_empty == 1){
            for (int i = 0; i < current_operator->get_size() + 1; ++i) {
                grid[current_operator->get_x() - 1][current_operator->get_y() + i - 1] = '-';     //Direction: East
                aop_ptr_grid[current_operator->get_x() - 1][current_operator->get_y() + i - 1] = current_operator;
                grid[current_operator->get_x() - 1][current_operator->get_y() - i - 1] = '-';     //Direction: West
                aop_ptr_grid[current_operator->get_x() - 1][current_operator->get_y() - i - 1] = current_operator;
            }
            operators[num_operators] = *current_operator;    //Append the successful placement in to the operators array
            current_operator->set_num(num_operators); //Change the number of the operator
            cout << "SUCCESS: Operator " << current_operator->get_sign() << " with size " << current_operator->get_size() << " is placed on ("
            << current_operator->get_x() << "," << current_operator->get_y() << ")." << endl;
            num_operators++;
            return true; //Successfully placed
        } else {
            return false; //Placement wasn't successful
        }
    } else if (current_operator->get_sign() == 'x') {        //Case sign = 'x'
        //Initialize errors and indexing variables
        int is_empty = 1;
        int checked_all = 0;
        int border_error = 0;

        //BORDER ERROR Checking in four directions (IMPROVEMENT: Can be done by checking two opposing diagonals)
        if ((current_operator->get_x() - current_operator->get_size() - 1 < 0) &&
        (current_operator->get_y() + current_operator->get_size() > grid_cols)){        //Direction: Northeast
            border_error = 1;
        } else if ((current_operator->get_size() + current_operator->get_x() > grid_rows) &&
        (current_operator->get_y() + current_operator->get_size() > grid_cols)) {       //Direction: Southeast
            border_error = 1;
        } else if ((current_operator->get_size() + current_operator->get_x() > grid_rows) &&
        (current_operator->get_y() - current_operator->get_size() - 1 < 0)) {       //Direction: Southwest
            border_error = 1;
        } else if ((current_operator->get_x() - current_operator->get_size() - 1 < 0) &&
        (current_operator->get_y() - current_operator->get_size() - 1 < 0)) {       //Direction: Northwest
            border_error = 1;
        }

        //CONFLICT ERROR Checking in four directions
        while (is_empty == 1 && checked_all == 0){
            for (int i = 0; i < current_operator->get_size(); ++i) {
                //Check if the location is inbounds
                if(current_operator->get_x()-i-1 < 0 || current_operator->get_y()+i-1 > grid_cols ||
                current_operator->get_x()+i-1 > grid_rows || current_operator->get_y()-i-1 < 0) {
                    is_empty = 1; //No conflict error rather a border error
                } else if (grid[current_operator->get_x()-i-1][current_operator->get_y()+i-1] != '\0'){      //Direction: Northeast
                    is_empty = 0;
                } else if (grid[current_operator->get_x()+i-1][current_operator->get_y()+i-1] != '\0'){       //Direction: Southeast
                    is_empty = 0;
                } else if (grid[current_operator->get_x()+i-1][current_operator->get_y()-i-1] != '\0'){       //Direction: Southwest
                    is_empty = 0;
                } else if (grid[current_operator->get_x()-i-1][current_operator->get_y()-i-1] != '\0'){       //Direction: Northwest
                    is_empty = 0;
                }
                if (is_empty == 0){      //Stop the loop if there is a conflict (IMPROVEMENT: Use a while loop)
                    break;
                }
            }
            if (is_empty == 1){ //If all the locations were empty until this point, then we can quit the loop
                checked_all = 1;
            }
        }
        if (border_error == 1){ //If there is a border error, print out the statement
            cout << "BORDER ERROR: Operator " << current_operator->get_sign() << " with size " << current_operator->get_size() << " can not be placed on ("
            << current_operator->get_x() << "," << current_operator->get_y() << ")." << endl;
        }
        // If there is a conflict error, print out the statement
        if (is_empty == 0){ //Meaning that at least one location that we want to place the operator on top is occupied
            cout << "CONFLICT ERROR: Operator " << current_operator->get_sign() << " with size " << current_operator->get_size() << " can not be placed on ("
            << current_operator->get_x() << "," << current_operator->get_y() << ")." << endl;
        }
        //If there are not any errors, place the operator
        if (border_error == 0 && is_empty == 1){
            for (int i = 0; i < current_operator->get_size() + 1; ++i) {
                grid[current_operator->get_x() - i - 1][current_operator->get_y() + i - 1] = 'x';     //Direction: Northeast
                aop_ptr_grid[current_operator->get_x() - i - 1][current_operator->get_y() + i - 1] = current_operator;
                grid[current_operator->get_x() + i - 1][current_operator->get_y() + i - 1] = 'x';     //Direction: Southeast
                aop_ptr_grid[current_operator->get_x() + i - 1][current_operator->get_y() + i - 1] = current_operator;
                grid[current_operator->get_x() + i - 1][current_operator->get_y() - i - 1] = 'x';     //Direction: Southwest
                aop_ptr_grid[current_operator->get_x() + i - 1][current_operator->get_y() - i - 1] = current_operator;
                grid[current_operator->get_x() - i - 1][current_operator->get_y() - i - 1] = 'x';     //Direction: Northwest
                aop_ptr_grid[current_operator->get_x() - i - 1][current_operator->get_y() - i - 1] = current_operator;
            }
            operators[num_operators] = *current_operator;    //Append the successful placement in to the operators array
            current_operator->set_num(num_operators); //Change the number of the operator
            cout << "SUCCESS: Operator " << current_operator->get_sign() << " with size " << current_operator->get_size() << " is placed on ("
            << current_operator->get_x() << "," << current_operator->get_y() << ")." << endl;
            num_operators++;
            return true; //Successfully placed
        } else {
            return false; //Placement wasn't successful
        }
    } else if (current_operator->get_sign() == '/'){        //Case sign = '/'
        //Initialize errors and indexing variables
        int is_empty = 1;
        int checked_all = 0;
        int border_error = 0;

        //BORDER ERROR Checking in two directions
        if ((current_operator->get_x() - current_operator->get_size() - 1 < 0) &&
        (current_operator->get_y() + current_operator->get_size() > grid_cols)){        //Direction: Northeast
            border_error = 1;
        } else if ((current_operator->get_size() + current_operator->get_x() > grid_rows) &&
        (current_operator->get_y() - current_operator->get_size() - 1 < 0)) {       //Direction: Southwest
            border_error = 1;
        }
        //CONFLICT ERROR Checking in four directions
        while (is_empty == 1 && checked_all == 0){
            for (int i = 0; i < current_operator->get_size(); ++i) {
                //Check if the location is inbounds
                if(current_operator->get_x()-i-1 < 0 || current_operator->get_y()+i-1 > grid_cols ||
                current_operator->get_x()+i-1 > grid_rows || current_operator->get_y()-i-1 < 0){
                    is_empty = 1; //No conflict error rather a border error
                } else if (grid[current_operator->get_x()-i-1][current_operator->get_y()+i-1] != '\0'){      //Direction: Northeast
                    is_empty = 0;
                } else if (grid[current_operator->get_x()+i-1][current_operator->get_y()-i-1] != '\0'){       //Direction: Southwest
                    is_empty = 0;
                }
                if (is_empty == 0){      //Stop the loop if there is a conflict (IMPROVEMENT: Use a while loop)
                    break;
                }
            }
            if (is_empty == 1){ //If all the locations were empty until this point, then we can quit the loop
                checked_all = 1;
            }
        }
        if (border_error == 1){ //If there is a border error, print out the statement
            cout << "BORDER ERROR: Operator " << current_operator->get_sign() << " with size " << current_operator->get_size() << " can not be placed on ("
            << current_operator->get_x() << "," << current_operator->get_y() << ")." << endl;
        }
        // If there is a conflict error, print out the statement
        if (is_empty == 0){ //Meaning that at least one location that we want to place the operator on top is occupied
            cout << "CONFLICT ERROR: Operator " << current_operator->get_sign() << " with size " << current_operator->get_size() << " can not be placed on ("
            << current_operator->get_x() << "," << current_operator->get_y() << ")." << endl;
        }
        //If there are not any errors, place the operator
        if (border_error == 0 && is_empty == 1){
            for (int i = 0; i < current_operator->get_size() + 1; ++i) {
                grid[current_operator->get_x() - i - 1][current_operator->get_y() + i - 1] = '/';     //Direction: Northeast
                aop_ptr_grid[current_operator->get_x() - i - 1][current_operator->get_y() + i - 1] = current_operator;
                grid[current_operator->get_x() + i - 1][current_operator->get_y() - i - 1] = '/';     //Direction: Southwest
                aop_ptr_grid[current_operator->get_x() + i - 1][current_operator->get_y() - i - 1] = current_operator;
            }
            operators[num_operators] = *current_operator;    //Append the successful placement in to the operators array
            current_operator->set_num(num_operators); //Change the number of the operator
            cout << "SUCCESS: Operator " << current_operator->get_sign() << " with size " << current_operator->get_size() << " is placed on ("
            << current_operator->get_x() << "," << current_operator->get_y() << ")." << endl;
            num_operators++;
            return true; //Successfully placed
        } else {
            return false; //Placement wasn't successful
        }
    } else {
        return false;
    }
}

//Move the placed operator
bool OperatorGrid::move_operator(int x, int y, char direction, int move_by) {
    if (x > grid_rows || y > grid_cols){ //Check if the pointed cell is in the grid range
        return false;
    }
    ArithmeticOperator *current_operator = aop_ptr_grid[x-1][y-1];
    if (current_operator == nullptr){       //If the pointed cell is empty
        return false;
    } else {
        int mid_x = current_operator->get_x();
        int mid_y = current_operator->get_y();
        int current_op_size = current_operator->get_size();
        char current_op_sign = current_operator->get_sign();

        //Clear out the cells containing the current operator
        if (current_op_sign == '+') { //sign is +
            for (int i = 0; i < current_op_size + 1; i++) {
                grid[mid_x - i - 1][mid_y - 1] = '\0';     //Direction: North
                aop_ptr_grid[mid_x - i - 1][mid_y - 1] = nullptr;
                grid[mid_x - 1][mid_y + i - 1] = '\0';     //Direction: East
                aop_ptr_grid[mid_x - 1][mid_y + i - 1] = nullptr;
                grid[mid_x + i - 1][mid_y - 1] = '\0';     //Direction: South
                aop_ptr_grid[mid_x + i - 1][mid_y - 1] = nullptr;
                grid[mid_x - 1][mid_y - i - 1] = '\0';     //Direction: West
                aop_ptr_grid[mid_x - 1][mid_y - i - 1] = nullptr;
            }
        } else if (current_op_sign == '-') { //sign is -
            for (int i = 0; i < current_op_size + 1; i++) {
                grid[mid_x - 1][mid_y + i - 1] = '\0';     //Direction: East
                aop_ptr_grid[mid_x - 1][mid_y + i - 1] = nullptr;
                grid[mid_x - 1][mid_y - i - 1] = '\0';     //Direction: West
                aop_ptr_grid[mid_x - 1][mid_y - i - 1] = nullptr;
            }
        } else if (current_op_sign == 'x') { //sign is x
             for (int i = 0; i < current_op_size + 1; i++) {
                grid[mid_x - i - 1][mid_y + i - 1] = '\0';     //Direction: Northeast
                aop_ptr_grid[mid_x - i - 1][mid_y + i - 1] = nullptr;
                grid[mid_x + i - 1][mid_y + i - 1] = '\0';     //Direction: Southeast
                aop_ptr_grid[mid_x + i - 1][mid_y + i - 1] = nullptr;
                grid[mid_x + i - 1][mid_y - i - 1] = '\0';     //Direction: Southwest
                aop_ptr_grid[mid_x + i - 1][mid_y - i - 1] = nullptr;
                grid[mid_x - i - 1][mid_y - i - 1] = '\0';     //Direction: Northwest
                aop_ptr_grid[mid_x - i - 1][mid_y - i - 1] = nullptr;
            }
        } else if (current_op_sign == '/') { //sign is /
            for (int i = 0; i < current_op_size + 1; i++) {
                grid[mid_x - i - 1][mid_y + i - 1] = '\0';     //Direction: Northeast
                aop_ptr_grid[mid_x - i - 1][mid_y + i - 1] = nullptr;
                grid[mid_x + i - 1][mid_y - i - 1] = '\0';     //Direction: Southwest
                aop_ptr_grid[mid_x + i - 1][mid_y - i - 1] = nullptr;
            }
        }
        //Check if movable towards the input direction
        //If so move, if not replace the already cleared operator
        int new_mid_x = mid_x;
        int new_mid_y = mid_y;
        if (direction == 'U'){      //Move up
            new_mid_x = mid_x - move_by;
        } else if (direction == 'R'){       //Move right
            new_mid_y = mid_y + move_by;
        } else if (direction == 'D'){       //Move down
            new_mid_x = mid_x + move_by;
        } else if (direction == 'L'){       //Move left
            new_mid_y = mid_y - move_by;
        } else {
            return false;
        }
        //Test for movement and move if available
        if (current_op_sign == '+'){       //Case sign = '+'
            //Initialize errors and indexing variables
            int is_empty = 1;
            int checked_all = 0;
            int border_error = 0;

            //BORDER ERROR Checking in four directions
            if (new_mid_x - current_op_size - 1 < 0){        //Direction: North
                border_error = 1;
            } else if (current_op_size + new_mid_y > grid_cols){       //Direction: East
                border_error = 1;
            } else if (current_op_size + new_mid_x > grid_rows) {       //Direction: South
                border_error = 1;
            } else if (new_mid_y - current_op_size - 1 < 0) {       //Direction: West
                border_error = 1;
            }

            //CONFLICT ERROR Checking in four directions
            while (is_empty == 1 && checked_all == 0){
                for (int i = 0; i < current_op_size + 1; i++) {
                    if (new_mid_x-i-1 < 0 || new_mid_y+i-1 > grid_cols || //Check if inbound
                    new_mid_x+i-1 > grid_rows || new_mid_y-i-1 < 0){
                        is_empty = 1; //Not a conflict error but a bound error
                    } else if (grid[new_mid_x-i-1][new_mid_y-1] != '\0'){      //Direction: North
                        is_empty = 0;
                    } else if (grid[new_mid_x-1][new_mid_y+i-1] != '\0'){       //Direction: East
                        is_empty = 0;
                    } else if (grid[new_mid_x+i-1][new_mid_y-1] != '\0'){       //Direction: South
                        is_empty = 0;
                    } else if (grid[new_mid_x-1][new_mid_y-i-1] != '\0'){       //Direction: West
                        is_empty = 0;
                    }
                    if (is_empty == 0){      //Stop the loop if there is a conflict (IMPROVEMENT: Use a while loop)
                        break;
                    }
                }
                if (is_empty == 1){ //If all the locations were empty until this point, then we can quit the loop
                    checked_all = 1;
                }
            }
            if (border_error == 1){ //If there is a border error, print out the statement
                cout << "BORDER ERROR: " << current_op_sign << " can not be moved from ("
                << mid_x << "," << mid_y << ") to (" << new_mid_x << "," << new_mid_y << ")." << endl;
            }
            // If there is a conflict error, print out the statement
            if (is_empty == 0){ //Meaning that at least one location that we want to place the operator on top is occupied
                cout << "CONFLICT ERROR: " << current_op_sign << " can not be moved from ("
                << mid_x << "," << mid_y << ") to (" << new_mid_x << "," << new_mid_y << ")." << endl;
            }
            //If there are not any errors, place the operator
            if (border_error == 0 && is_empty == 1){
                for (int i = 0; i < current_op_size + 1; i++) {
                    grid[new_mid_x - i - 1][new_mid_y - 1] = '+';     //Direction: North
                    aop_ptr_grid[new_mid_x - i - 1][new_mid_y - 1] = current_operator;
                    grid[new_mid_x - 1][new_mid_y + i - 1] = '+';     //Direction: East
                    aop_ptr_grid[new_mid_x - 1][new_mid_y + i - 1] = current_operator;
                    grid[new_mid_x + i - 1][new_mid_y - 1] = '+';     //Direction: South
                    aop_ptr_grid[new_mid_x + i - 1][new_mid_y - 1] = current_operator;
                    grid[new_mid_x - 1][new_mid_y - i - 1] = '+';     //Direction: West
                    aop_ptr_grid[new_mid_x - 1][new_mid_y - i - 1] = current_operator;
                }
                cout << "SUCCESS: " << current_op_sign << " moved from ("
                << mid_x << "," << mid_y << ") to (" << new_mid_x << "," << new_mid_y << ")." << endl;
                current_operator->set_x(new_mid_x); //Set the new values
                current_operator->set_y(new_mid_y);
                operators[current_operator->get_num()] = *current_operator; //Replace the current operators address
                return true; //Movement was successful
            } else {  //Replace the old operator
                for (int i = 0; i < current_op_size + 1; i++) {
                    grid[mid_x - i - 1][mid_y - 1] = '+';     //Direction: North
                    aop_ptr_grid[mid_x - i - 1][mid_y - 1] = current_operator;
                    grid[mid_x - 1][mid_y + i - 1] = '+';     //Direction: East
                    aop_ptr_grid[mid_x - 1][mid_y + i - 1] = current_operator;
                    grid[mid_x + i - 1][mid_y - 1] = '+';     //Direction: South
                    aop_ptr_grid[mid_x + i - 1][mid_y - 1] = current_operator;
                    grid[mid_x - 1][mid_y - i - 1] = '+';     //Direction: West
                    aop_ptr_grid[mid_x - 1][mid_y - i - 1] = current_operator;
                }
                return false; //Movement was unsuccessful
            }
        } else if (current_op_sign == '-'){        //Case sign = '-'
            //Initialize errors and indexing variables
            int is_empty = 1;
            int checked_all = 0;
            int border_error = 0;

            //BORDER ERROR Checking in two directions
            if (current_op_size + new_mid_y > grid_cols){       //Direction: East
                border_error = 1;
            } else if (new_mid_y - current_op_size - 1 < 0) {       //Direction: West
                border_error = 1;
            }

            //CONFLICT ERROR Checking in two directions
            while (is_empty == 1 && checked_all == 0){
                for (int i = 0; i < current_op_size + 1; i++) {
                    if (new_mid_y+i-1 > grid_cols || new_mid_y-i-1 < 0){ //Check if inbound
                        is_empty = 1; //Not a conflict error but a bound error
                    } else if (grid[new_mid_x-1][new_mid_y+i-1] != '\0'){       //Direction: East
                        is_empty = 0;
                    } else if (grid[new_mid_x-1][new_mid_y-i-1] != '\0'){       //Direction: West
                        is_empty = 0;
                    }
                    if (is_empty == 0){      //Stop the loop if there is a conflict (IMPROVEMENT: Use a while loop)
                        break;
                    }
                }
                if (is_empty == 1){ //If all the locations were empty until this point, then we can quit the loop
                    checked_all = 1;
                }
            }
            if (border_error == 1){ //If there is a border error, print out the statement
                cout << "BORDER ERROR: " << current_op_sign << " can not be moved from ("
                << mid_x << "," << mid_y << ") to (" << new_mid_x << "," << new_mid_y << ")." << endl;
            }
            // If there is a conflict error, print out the statement
            if (is_empty == 0){ //Meaning that at least one location that we want to place the operator on top is occupied
                cout << "CONFLICT ERROR: " << current_op_sign << " can not be moved from ("
                << mid_x << "," << mid_y << ") to (" << new_mid_x << "," << new_mid_y << ")." << endl;
            }
            //If there are not any errors, place the operator
            if (border_error == 0 && is_empty == 1){
                for (int i = 0; i < current_op_size + 1; i++) {
                    grid[new_mid_x - 1][new_mid_y + i - 1] = '-';     //Direction: East
                    aop_ptr_grid[new_mid_x - 1][new_mid_y + i - 1] = current_operator;
                    grid[new_mid_x - 1][new_mid_y - i - 1] = '-';     //Direction: West
                    aop_ptr_grid[new_mid_x - 1][new_mid_y - i - 1] = current_operator;
                }
                cout << "SUCCESS: " << current_op_sign << " moved from ("
                << mid_x << "," << mid_y << ") to (" << new_mid_x << "," << new_mid_y << ")." << endl;
                current_operator->set_x(new_mid_x); //Set the new values
                current_operator->set_y(new_mid_y);
                operators[current_operator->get_num()] = *current_operator; //Replace the current operators address
                return true; //Movement was successful
            } else { //Replace the old operator
                for (int i = 0; i < current_op_size + 1; i++) {
                    grid[mid_x - 1][mid_y + i - 1] = '-';     //Direction: East
                    aop_ptr_grid[mid_x - 1][mid_y + i - 1] = current_operator;
                    grid[mid_x - 1][mid_y - i - 1] = '-';     //Direction: West
                    aop_ptr_grid[mid_x - 1][mid_y - i - 1] = current_operator;
                }
                return false; //Movement was unseccessful
            }
        } else if (current_op_sign == 'x') {        //Case sign = 'x'
            //Initialize errors and indexing variables
            int is_empty = 1;
            int checked_all = 0;
            int border_error = 0;

            //BORDER ERROR Checking in four directions (IMPROVEMENT: Can be done by checking two opposing diagonals)
            if ((new_mid_x - current_op_size - 1 < 0) ||
            (new_mid_y + current_op_size > grid_cols)){        //Direction: Northeast
                border_error = 1;
            } else if ((current_op_size + new_mid_x > grid_rows) ||
            (new_mid_y + current_op_size > grid_cols)) {       //Direction: Southeast
                border_error = 1;
            } else if ((current_op_size + new_mid_x > grid_rows) ||
            (new_mid_y - current_op_size - 1 < 0)) {       //Direction: Southwest
                border_error = 1;
            } else if ((new_mid_x - current_op_size - 1 < 0) ||
            (new_mid_y - current_op_size - 1 < 0)) {       //Direction: Northwest
                border_error = 1;
            }

            //CONFLICT ERROR Checking in four directions
            while (is_empty == 1 && checked_all == 0){
                for (int i = 0; i < current_op_size + 1; i++) {
                    if (new_mid_x-i-1 < 0 || new_mid_y+i-1 > grid_cols || //Checking the bounds
                    new_mid_x+i-1 > grid_rows || new_mid_y-i-1 < 0){
                        is_empty = 1; //Not a conflict error but a bound error
                    } else if (grid[new_mid_x-i-1][new_mid_y+i-1] != '\0'){      //Direction: Northeast
                        is_empty = 0;
                    } else if (grid[new_mid_x+i-1][new_mid_y+i-1] != '\0'){       //Direction: Southeast
                        is_empty = 0;
                    } else if (grid[new_mid_x+i-1][new_mid_y-i-1] != '\0'){       //Direction: Southwest
                        is_empty = 0;
                    } else if (grid[new_mid_x-i-1][new_mid_y-i-1] != '\0'){       //Direction: Northwest
                        is_empty = 0;
                    }
                    if (is_empty == 0){      //Stop the loop if there is a conflict (IMPROVEMENT: Use a while loop)
                        break;
                    }
                }
                if (is_empty == 1){ //If all the locations were empty until this point, then we can quit the loop
                    checked_all = 1;
                }
            }
            if (border_error == 1){ //If there is a border error, print out the statement
                cout << "BORDER ERROR: " << current_op_sign << " can not be moved from ("
                << mid_x << "," << mid_y << ") to (" << new_mid_x << "," << new_mid_y << ")." << endl;
            }
            // If there is a conflict error, print out the statement
            if (is_empty == 0){ //Meaning that at least one location that we want to place the operator on top is occupied
                cout << "CONFLICT ERROR: " << current_op_sign << " can not be moved from ("
                << mid_x << "," << mid_y << ") to (" << new_mid_x << "," << new_mid_y << ")." << endl;
            }
            //If there are not any errors, place the operator
            if (border_error == 0 && is_empty == 1){
                for (int i = 0; i < current_op_size + 1; i++) {
                    grid[new_mid_x - i - 1][new_mid_y + i - 1] = 'x';     //Direction: Northeast
                    aop_ptr_grid[new_mid_x - i - 1][new_mid_y + i - 1] = current_operator;
                    grid[new_mid_x + i - 1][new_mid_y + i - 1] = 'x';     //Direction: Southeast
                    aop_ptr_grid[new_mid_x + i - 1][new_mid_y + i - 1] = current_operator;
                    grid[new_mid_x + i - 1][new_mid_y - i - 1] = 'x';     //Direction: Southwest
                    aop_ptr_grid[new_mid_x + i - 1][new_mid_y - i - 1] = current_operator;
                    grid[new_mid_x - i - 1][new_mid_y - i - 1] = 'x';     //Direction: Northwest
                    aop_ptr_grid[new_mid_x - i - 1][new_mid_y - i - 1] = current_operator;
                }
                cout << "SUCCESS: " << current_op_sign << " moved from ("
                << mid_x << "," << mid_y << ") to (" << new_mid_x << "," << new_mid_y << ")." << endl;
                current_operator->set_x(new_mid_x); //Set the new values
                current_operator->set_y(new_mid_y);
                operators[current_operator->get_num()] = *current_operator; //Replace the current operators address
                return true; //Movement was successful
            } else { //Replace the old operator
                for (int i = 0; i < current_op_size + 1; i++) {
                    grid[mid_x - i - 1][mid_y + i - 1] = 'x';     //Direction: Northeast
                    aop_ptr_grid[mid_x - i - 1][mid_y + i - 1] = current_operator;
                    grid[mid_x + i - 1][mid_y + i - 1] = 'x';     //Direction: Southeast
                    aop_ptr_grid[mid_x + i - 1][mid_y + i - 1] = current_operator;
                    grid[mid_x + i - 1][mid_y - i - 1] = 'x';     //Direction: Southwest
                    aop_ptr_grid[mid_x + i - 1][mid_y - i - 1] = current_operator;
                    grid[mid_x - i - 1][mid_y - i - 1] = 'x';     //Direction: Northwest
                    aop_ptr_grid[mid_x - i - 1][mid_y - i - 1] = current_operator;
                }
                return false; //Movement was unsuccessful
            }
        } else if (current_op_sign == '/'){        //Case sign = '/'
            //Initialize errors and indexing variables
            int is_empty = 1;
            int checked_all = 0;
            int border_error = 0;

            //BORDER ERROR Checking in two directions
            if ((new_mid_x - current_op_size - 1 < 0) ||
            (new_mid_y + current_op_size > grid_cols)){        //Direction: Northeast
                border_error = 1;
            } else if ((current_op_size + new_mid_x > grid_rows) ||
            (new_mid_y - current_op_size - 1 < 0)) {       //Direction: Southwest
                border_error = 1;
            }
            //CONFLICT ERROR Checking in four directions
            while (is_empty == 1 && checked_all == 0){
                for (int i = 0; i < current_op_size + 1; i++) {
                    if (new_mid_x-i-1 < 0 || new_mid_y+i-1 > grid_cols || //Check if inbound
                    new_mid_x+i-1 > grid_rows || new_mid_y-i-1 < 0) {
                        is_empty = 1; //Not a conflict error but a bound error
                    } else if (grid[new_mid_x-i-1][new_mid_y+i-1] != '\0'){      //Direction: Northeast
                        is_empty = 0;
                    } else if (grid[new_mid_x+i-1][new_mid_y-i-1] != '\0'){       //Direction: Southwest
                        is_empty = 0;
                    }
                    if (is_empty == 0){      //Stop the loop if there is a conflict (IMPROVEMENT: Use a while loop)
                        break;
                    }
                }
                if (is_empty == 1){ //If all the locations were empty until this point, then we can quit the loop
                    checked_all = 1;
                }
            }
            if (border_error == 1){ //If there is a border error, print out the statement
                cout << "BORDER ERROR: " << current_op_sign << " can not be moved from ("
                << mid_x << "," << mid_y << ") to (" << new_mid_x << "," << new_mid_y << ")." << endl;
            }
            // If there is a conflict error, print out the statement
            if (is_empty == 0){ //Meaning that at least one location that we want to place the operator on top is occupied
                cout << "CONFLICT ERROR: " << current_op_sign << " can not be moved from ("
                << mid_x << "," << mid_y << ") to (" << new_mid_x << "," << new_mid_y << ")." << endl;
            }
            //If there are not any errors, place the operator
            if (border_error == 0 && is_empty == 1){
                for (int i = 0; i < current_op_size + 1; i++) {
                    grid[new_mid_x - i - 1][new_mid_y + i - 1] = '/';     //Direction: Northeast
                    aop_ptr_grid[new_mid_x - i - 1][new_mid_y + i - 1] = current_operator;
                    grid[new_mid_x + i - 1][new_mid_y - i - 1] = '/';     //Direction: Southwest
                    aop_ptr_grid[new_mid_x + i - 1][new_mid_y - i - 1] = current_operator;
                }
                cout << "SUCCESS: " << current_op_sign << " moved from ("
                << mid_x << "," << mid_y << ") to (" << new_mid_x << "," << new_mid_y << ")." << endl;
                current_operator->set_x(new_mid_x); //Set the new values
                current_operator->set_y(new_mid_y);
                operators[current_operator->get_num()] = *current_operator; //Replace the current operators address
                return true; //Movement was successful
            } else { //Replace the old operator
                for (int i = 0; i < current_op_size + 1; i++) {
                    grid[mid_x - i - 1][mid_y + i - 1] = '/';     //Direction: Northeast
                    aop_ptr_grid[mid_x - i - 1][mid_y + i - 1] = current_operator;
                    grid[mid_x + i - 1][mid_y - i - 1] = '/';     //Direction: Southwest
                    aop_ptr_grid[mid_x + i - 1][mid_y - i - 1] = current_operator;
                }
                return false; //Movement was unsuccessful
            }
        } else {
            return false;
        }
    }
}

//Print method of the operators
void OperatorGrid::print_operators() {
    for (int i = 0; i < num_operators; ++i) {
        operators[i].print_operator();
    }
}
