==============================
COMPUTER SCIENCE PROJECT
==============================


==============================
AUTHOR
==============================

Endri Kastrati, ekas6@student.monash.edu, endriau@gmail.com


==============================
TITLE
==============================

A Deep learning approach to thyroid disease classification



=============================
DEPENDENCIES
=============================

Project was build using GSL-2.4 but any version will suffice.
You must have installed the GNU scientific library (globally)


=============================
HOW TO COMPILE
=============================

Navigate inside the csproject directory and execute "make"


============================
HOW TO TRAIN THE CLASSIFIER
============================

After having successfully compiled the project execute the following command:

/neuralnet --train --pattern-classification --normalization=yes --in-file=datasets/thyroid-train.data --dump-dir=thyroidologist --signals=5 --nlayers=2 --neurons-per-layer=[20,3] --activation=lgst --epsilon=1e-12 --eta=0.5 --momentum=0.009 --epochs=700



==============================
HOW TO RUN THE WEB APPLICATION
==============================

After having successfully compiled the project and obtained a thyroid disease classifier navigate into the webapp directory and execute the following:  node index.js
Now navigate to localhost:3000, and start using the web interface.



====================================
HOW TO RUN THE CURVE APPROXIMATIONS
====================================

Make sure that you have compiled the source code via "make" and the neuralnet program has been generated.
Navigate into the datasets directory and execute the following commands:

octave --no-gui
>> quadratic
>> circlef
>> sinf

