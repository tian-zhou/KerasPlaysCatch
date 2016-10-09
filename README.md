
####Problem formulation
1. The **action** space $A$ = {move_left, stay, move_right}, $A\in \mathbb{R}, |A| = 3$.
2. The **state** space $S$ =  [fruit_row, fruit_column, basket_center], $S\in \mathbb{R}^3, |S| = N^3$ where $N$ is the grid size for the square canvas, since the range of each element of $S$ is $[1,N]$
3. The **reward** is defined as 
	- $+1$ for catching the fruit (when fruit lands on floor)  
	- $-1$ for missing the fruit (when fruit lands on floor)
	- $0$ for fruit still falling (in the air) 
4. 




==The following are README from original repo about how to run the code==

---
Code for [Keras plays catch](http://edersantana.github.io/articles/keras_rl/) blog post

#### Train
```bash
python qlearn.py
```

#### Generate figures
```bash
python test.py
```

#### Make gif
```bash
ffmpeg -i %03d.png output.gif -vf fps=1
```

#### Requirements
* Prior supervised learning and Keras knowledge
* Python science stack (numpy, scipy, matplotlib) - Install Anaconda!
* Theano or Tensorflow
* Keras
* ffmpeg (optional)
