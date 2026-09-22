
#  argon2 password hasher used to protect stored passwords
from argon2 import PasswordHasher
# imports the common errors that can occur during password verification
from argon2.exceptions import (
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)
# creates one reusable argon2 password hasher
password_hasher = PasswordHasher()
# converts a plain text password into a secure hash
def hash_password(password):
    # convert plain text password to argon2 hash
    clean_password = str(password or "")
    # empty passwords should not be hashed
    if not clean_password:
        raise ValueError("A password is required.")
    return password_hasher.hash(clean_password)


# checks whether a supplied password matches the stored hash
def verify_password(password, stored_password_hash):
    # check if the password matches the stored hash
    clean_password = str(password or "")
    clean_hash = str(stored_password_hash or "")
    # missing password data cannot be verified
    if not clean_password or not clean_hash:
        return False
    try:
        return password_hasher.verify(clean_hash, clean_password)
    # invalid hashes or incorrect passwords are treated as failed verification
    except (
        VerifyMismatchError,
        InvalidHashError,
        VerificationError,
    ):
        return False