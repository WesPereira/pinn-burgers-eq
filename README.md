# Physics-informed Neural Networks: Burgers Equation
<span><img src="https://img.shields.io/github/contributors/WesPereira/pinn-burgers-eq"> <img src="https://img.shields.io/github/last-commit/WesPereira/pinn-burgers-eq"></span>

Physics-informed Neural Networks are a type of neural networks that are trained to solve supervised learning tasks while respecting any given laws of physics described by general nonlinear partial differential equations.

in this repository we introduced PINNs applied to [Burgers Equations](https://en.wikipedia.org/wiki/Burgers%27_equation). Therefore, in this repository we will apply PINNs to solve the equation through a Multi Layer Perceptron (MLP).

## PINNs Schematic

Shown below is a schematic drawing of the network used for **1D Burgers Equation**.

![image 1](imgs/image1.png)


## How to run it

To run the training you need to have the `x` and `t` simulation file in `.pkl` (pickle) format. Then, install the requirements:

```bash
pip3 install -r requiments.txt
```

To plot results in Latex format, we need to download necessary configurations:

```bash
sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
```

And finally, run training:

```bash
python3 train.py --save_path=/example/weights.pt
```

## Results for 1D Burgers Equation

After training, the results in shown below.

![image 2](imgs/image2.png)

From the figure above, we can see that the network, even if simple, managed to represent well the solution of the proposed differential equation.

## Contributors
The list of contributors is presented below.
<table>
  <tr>
    <td align="center"><a href="https://github.com/WesPereira"><img src="https://avatars.githubusercontent.com/u/49962478?v=4" width="100px;" alt=""/><br /><sub><b>WesPereira</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/esgomi"><img src="https://avatars.githubusercontent.com/u/6525442?v=4" width="100px;" alt=""/><br /><sub><b>esgomi</b></sub></a><br /></td>
  </tr>
</table>