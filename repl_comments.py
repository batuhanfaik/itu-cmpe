keepCommenting = "y"
comments = []
while (keepCommenting == "y"):
    comments.append(input("Enter your comment: "))
    for comment in comments: print(comment)
    keepCommenting = input("Do you want to add another comment? (y/n): ")
