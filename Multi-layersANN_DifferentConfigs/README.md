CS 615 Deep Learning Himanshu Gupta Assignment 2 (Matlab) README:

Please refer to the Analysis.pdf for graphs and detailed analysis. For
simplicity, I have serialized yalefaces images into database.mat for
faster read and analysis.

1)  "hg387.pdf" is the Report containing all the required findings.
2)  "ANN2.m" contains the source code for Shallow ANN where: default
    directory for getting images is "./yalefaces/" if anyone wants to
    changes it, portion of the code commented to replace the default
    directory name Also, for increasing performance, I am attaching
    database.mat file which is pre-populated matrix of images so first
    program looks for this file, if not exists then run the algorithm by
    reading images from the directory. On top of these, plotted graph
    would be saved in "ANN2.fig" file. Sample is attached with the zip
    file.
3)  "ANN3.m" contains the source code for Multi-layer ANN where: default
    directory for getting images is "./yalefaces/" if anyone wants to
    changes it, portion of the code commented to replace the default
    directory name Also, for increasing performance, I am attaching
    database.mat file which is pre-populated matrix of image so first
    program looks for this file, if not exists then run the algorithm by
    reading images from the directory. On top of these, plotted graph
    would be saved in "ANN3.fig" file. Sample is attached with the zip
    file.
4)  The programs are throughly tested and performed well in various
    cases.

