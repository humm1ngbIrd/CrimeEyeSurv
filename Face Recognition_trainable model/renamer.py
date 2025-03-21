import os

directory = 'dataset\jaryd'
print(os.scandir(directory))
for i,entry in enumerate(os.scandir(directory)):
        # rename file by removing all characters before underscore and inserting 'bhairav' before underscore
        os.rename(entry.path, os.path.join(directory, f"bhairav_{i}.jpg" ))
