# This file is used to setup the project. It is executed when the project is imported.
# This file should be used to download all large files (e.g., model weights) and store them to disk.
# In this file, you can also check if the environment works as expected.
# If something goes wrong, you can exit the script with a non-zero exit code.
# This will help you detect issues early on.
#
# Below, you can find some sample code:

def download_large_files():
    return True

def check_environment():
    return True


if __name__ == "__main__":
    print("Perform your setup here.")
    
    if not check_environment():
        print("Environment check failed.")
        exit(1)
        
    if not download_large_files():
        print("Downloading large files failed.")
        exit(1)
        
    exit(0)