// OOMD 2020-2021 Assignment 4

#include <iostream>
#include <string>
#include <vector>
using namespace std;

// Reperesents the grade earned by a student for a particular course
class GradeForCourse {
  string course;          // Code of the course e.g. "BLG468E"
  float grade;            // Grade earned by the student
 public:
  GradeForCourse(string c, float g) : course(c), grade(g) { ; } // constructor
  const string &getCourse() { return course; }
  float getGrade() { return grade; }
};

// Class System_A represents the remote system A
class System_A {
 public:
  void establish_connection() {
    cout << "Connection to System A is established" << endl;
  }
  void close_connection() {
    cout << "Connection to System A is closed" << endl;
  }
  // This method represents the operation that is performed by A to receive necessary information
  // It receives ID and name
  void post(const string &ID, const string &name) {
    cout << "System A receives:" << endl;
    cout << "ID: " << ID << endl;
    cout << "Name: " << name << endl;
  }
};

// Class System_B represents the remote system B
class System_B {
 public:
  void connect() {
    cout << "Connect to System B" << endl;
  }
  void disconnect() {
    cout << "Disconnect from System B" << endl;
  }
  // This method represents the operation that is performed by B to receive necessary information
  // It receives ID and vector of grades
  void send_data(const string &ID, const vector<GradeForCourse *> &grades) {
    cout << "System B receives:" << endl;
    cout << "ID: " << ID << endl;
    cout << "Courses and grades" << endl;
    for (unsigned int j = 0; j < grades.size(); j++)    // display vector contents
      cout << grades[j]->getCourse() << ' ' << grades[j]->getGrade() << endl;
  }
};

// Student class
class Student {
  string ID, name;
  vector<GradeForCourse *> grades;           // create the vector of grades
 public:
  Student();                              // read ID and name from the keyboard
  Student(const string &, const string &);// take ID and name from the main method
  void setGradeforCourse();               // read code of course and grade for that course
  void send();                            // Connect to a remote system and send data
  ~Student();                             // Destructor
  // Getters for the Student attributes
  const string &getID() { return ID; };
  const string &getName() { return name; };
  const vector<GradeForCourse *> &getGrades() { return grades; };
};

// During construction of the Student read ID and name from the keyboard
Student::Student() {
  cout << "Give the student an ID: ";
  cin >> this->ID;
  cout << "Give the student a name: ";
  cin >> this->name;
}

Student::Student(const string& ID, const string& name) {
  this->ID = ID;
  this->name = name;
}

// Destructor deletes GradeForCourse objects in the vector of the grades
Student::~Student() {
  for (unsigned int j = 0; j < grades.size(); j++)
    delete grades[j];
}

// This method creates the grade for a particular course and writes the data into the vector of the grades in the Student object
void Student::setGradeforCourse() {
  cout << "Give the code of the course for " << name << ": ";
  string code;
  cin >> code;            // Write the code without a space character e.g. "BLG468E"
  float grade;
  do {
    cout << "Give the grade for the course: ";
    cin >> grade;
  } while (grade < 0 || grade > 4);
  GradeForCourse *courseGrade = new GradeForCourse(code, grade);  // Create the grade for the course
  grades.push_back(courseGrade);                      //Put the grade into the vector of the grades
}

// Remote System Adapter Interface
class IRemoteSystemAdapter {
 public:
  virtual void sendToRemoteSystem(Student &) = 0;
  virtual ~IRemoteSystemAdapter()= default;
};

// System A Adapter
class SystemAAdapter : public IRemoteSystemAdapter {
  System_A* system;
 public:
  SystemAAdapter();
  void sendToRemoteSystem(Student &s) override;
  ~SystemAAdapter() override;
};

SystemAAdapter::SystemAAdapter() {
  system = new System_A;
}

SystemAAdapter::~SystemAAdapter() {
  delete system;
}

void SystemAAdapter::sendToRemoteSystem(Student &s) {
  system->establish_connection();
  system->post(s.getID(), s.getName());
  system->close_connection();
}

// System B Adapter
class SystemBAdapter : public IRemoteSystemAdapter {
  System_B* system;
 public:
  SystemBAdapter();
  void sendToRemoteSystem(Student &s) override;
  ~SystemBAdapter() override;
};

SystemBAdapter::SystemBAdapter() {
  system = new System_B;
}

SystemBAdapter::~SystemBAdapter() {
  delete system;
}

void SystemBAdapter::sendToRemoteSystem(Student &s) {
  system->connect();
  system->send_data(s.getID(), s.getGrades());
  system->disconnect();
}

// The SystemsFactory
class SystemsFactory {
  static SystemsFactory *instance;
  string studentIDControl;
  SystemAAdapter *sysAAdapter;
  SystemBAdapter *sysBAdapter;
  // Methods
  SystemsFactory();
  static SystemAAdapter *getSysAAdapter();
  static SystemBAdapter *getSysBAdapter();
 public:
  static SystemsFactory *getInstance();
  IRemoteSystemAdapter *getRemoteSystemAdapter(Student&);
  void setStudentIDControl(const string&);
  ~SystemsFactory();
};

SystemsFactory *SystemsFactory::instance = nullptr;

SystemsFactory *SystemsFactory::getInstance() {
  if (!instance) {
    instance = new SystemsFactory();
    instance->studentIDControl = "150170000";
  }
  return instance;
}

SystemsFactory::~SystemsFactory() {
  delete sysAAdapter;
  delete sysBAdapter;
  delete instance;
}

SystemAAdapter *SystemsFactory::getSysAAdapter() {
  return new SystemAAdapter;
}

SystemBAdapter *SystemsFactory::getSysBAdapter() {
  return new SystemBAdapter;
}

SystemsFactory::SystemsFactory() {
  sysAAdapter = getSysAAdapter();
  sysBAdapter = getSysBAdapter();
}

IRemoteSystemAdapter *SystemsFactory::getRemoteSystemAdapter(Student& s) {
  // The remote system decision logic is implemented in the factory
  // If student is registered to computer engineering before 2017 get system A, else get system B
  if (s.getID() < studentIDControl) {
    return sysAAdapter;
  } else {
    return sysBAdapter;
  }
}

void SystemsFactory::setStudentIDControl(const string&s) {
  studentIDControl = s;
}

// This method sends information about students to different systems
void Student::send() {
  SystemsFactory *sysFact = SystemsFactory::getInstance();
  IRemoteSystemAdapter *remoteSystemAdapter = sysFact->getRemoteSystemAdapter(*this);
  remoteSystemAdapter->sendToRemoteSystem(*this);
}

// Test Create students, grades, send data to different systems (A or B), change the systems in run-time
int main() {
  Student student1 = Student("150160001", "Ahmet");
  Student student2 = Student("150180705", "Batuhan");

  student1.setGradeforCourse();
  student2.setGradeforCourse();
  student1.send();
  student2.send();

  /* SystemsFactory is assigned in the main method just to show the changes in the remote systems
  *  This is not required and can be considered wrong
  *  The only reason I included it in the main method is to show that changes in the run-time
  *     can be done by changing the decision logic too */
  SystemsFactory *sysFact = SystemsFactory::getInstance();
  cout << "\nChanging the decision logic!\n\n";
  // Change the student number control to 150190000 so student Batuhan uses remote system A, instead of B
  sysFact->setStudentIDControl("150190000");

  student1.send();
  student2.send();
  return 0;
}