import glob
import piexif

nfiles = 0
for filename in glob.iglob('/data/ssd1/jju/ImageNet/**/*.JPEG', recursive=True):
    nfiles = nfiles + 1
    print("About to process file %d, which is %s." % (nfiles,filename))
    piexif.remove(filename)