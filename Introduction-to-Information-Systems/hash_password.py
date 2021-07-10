'''The below is a simple example of how you would use "hashing" to
compare passwords without having to save the password - you would-
save the "hash" of the password instead.
Written for the class BLG101E assignment 2.'''

'''We need to import a function for generating hashes from byte
strings.'''
from hashlib import sha256

''' To make use of the above sha256 function we need to do a couple
of things. So we write our own function which takes a password and
returns a string consisting of a hash of the password.'''
def create_hash(password):
    pw_bytestring = password.encode()
    return sha256(pw_bytestring).hexdigest()

'''In the following example we get a password from the user, generate a
hash from it, then get another password and generate a hash, and
check if they are the same. But note that we do not compare the
passwords themselves - once we have the hashes we no longer need the
passwords.'''

pw1 = input('Please enter your password:')
hsh1 = create_hash(pw1)
print('The hash of that is',hsh1)

pw2 = input('Please enter another password that we can check against that:')
hsh2 = create_hash(pw2)
print('The hash of that is',hsh2)

if hsh1 == hsh2:
    print('Those were the same passwords')
else:
    print('Those were different passwords')
