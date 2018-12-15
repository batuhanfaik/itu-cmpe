from hashlib import sha256

#This function was copied from "https://bitbucket.org/damienjadeduff/hashing_example/raw/master/hash_password.py"
def create_hash(password):
    pw_bytestring = password.encode()
    return sha256(pw_bytestring).hexdigest()

keepCommenting = "y"
password_hash = "b15d6647b6f40b3568a66c6c787ed9cb1f68438bd99197cbd21fc8e82aff0b28"
comments = []

while (keepCommenting == "y"):
    comment = input("Enter your comment: ")
    if password_hash == create_hash(input("Enter your password: ")):
        comments.append(comment)
    else:
        print("Sorry, you are not authorized to perform this action.")
    for comment in comments: print(comment)
    keepCommenting = input("Do you want to add another comment? (y/n): ")
