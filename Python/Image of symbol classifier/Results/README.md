# User manual for executing project code

For running this code, all files must be in the same directory. 

Files needed:
- mathwriting-2024symbols_imgr3.zip can be downloaded from https://drive.google.com/file/d/1i19kYTqdaqE6XnK4_G1oMHncK_T_p2in/view?usp=sharing
- mathwriting-2024symbols.zip can be downloaded from https://drive.google.com/file/d/1qNP0rjf5rgZm2ekCdefD_xc5pnv2LRZc/view?usp=drive_link 

Libraries needed:
- os
- zipfile
- xml
- functools
- operator
- pillow
- torch
- torchvision
- scikit-learn
- matplotlib

Executing the code:<br>
After you have imported all the files and installed libraries, these are the function that can be called from your terminal:  
python train_model.py --datatype='mathwriting' --lr=0.01 --optim='sgd' --epochs=1 --saveweights='name.pth'  
python eval_model.py --datatype='mathwriting' --weights='name.pth'  
python predict.py --datatype='mathwriting' --weights='name.pth'

Parameters:  
datatype should be either 'mathwriting' or 'mnist'.  
lr should be an integer.  
optim should be either 'sgd' or 'adam'.  
epochs should be an integer.  
saveweights should be a string ending in '.pth'.  
weights should be a string ending in '.pth'.
