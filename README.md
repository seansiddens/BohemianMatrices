A script for visualizing the distributions of eigenvalues. 

Inspired by [http://www.bohemianmatrices.com/]()

## Dependencies
- numpy
- pillow
- tqdm
- matplotlib
- perlin_noise

## Example Images


![littlewood6_deg23_small](https://user-images.githubusercontent.com/11508260/120935883-b9008300-c6b9-11eb-8cf5-f5dea203e26c.png)

Distribution of roots of all degree 23 Littlewood polynomials. Roots are found by constructing the corresponding companion matrix for each polynomial and computing it's eigenvalues. Only complex eigenvalues are plotted in the final image. 


![tridiagonal_small](https://user-images.githubusercontent.com/11508260/120935893-bef66400-c6b9-11eb-9b36-93ba5617214e.png)

Plot of eigenvalues of 25 million 10x10 tridiagonal matrices. Entries are sampled from a discrete pre-defined set of values. 

![beta_small](https://user-images.githubusercontent.com/11508260/120935900-c6b60880-c6b9-11eb-88af-71790ee50dfa.png)

Eigenvalues of 25 million 4x4 matrices. Entries of the matrices are in the form 2x - 1, where x is sampled from Beta(0.01, 0.01). 

![rotation3](https://user-images.githubusercontent.com/11508260/120935909-d03f7080-c6b9-11eb-8368-53b1b0ff5ba7.gif)

Eigenvalues of 4x4 companion matrices with entries in the final column also in the form 2x-1. However, these entries are also rotated along the unit circle every frame. 
