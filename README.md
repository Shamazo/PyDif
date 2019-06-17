
### Introduction


PyDif is Python library for automatic differentiation, a computational technique for evaluating derivatives that has several distinct advantages compared to alternative computer-based methods for differentiation.


Numerical methods, such as the finite differences method, are simple to implement but are plagued by floating point round-off errors and approximation errors, affecting accuracy and stability.

On the other hand, symbolic differentiation can compute derivatives up to machine precision, but can quickly become very computationally expensive.

As an alternative, automatic differentiation is capable of machine precision accuracy without the computational cost of symbolic differentiation.

This has significant implications for an extremely wide variety of applications across science and engineering.

In our case, the software library allows the computation of the gradient, first order partial derivatives (Jacobian) and some second order partial derivatives (diagonal of Hessian) for any mathematical function a user has implemented in Python, supporting a scalar function or a vector function of scalar inputs and scalar or vector outputs.


Users can leverage our software for their own applications, such as optimization, root-finding, and much more. The software library implements optimizers and a root-finding algorithm that use automatic differentiation.

This library is largely still a work in progess and comes with no guarantees of any kind.

The project is licensed under the MIT License.  

For a prettier latex presentation, check out the ipynb version of this readme [here](docs/pydif_docs_final.ipynb)

[![Build Status](https://travis-ci.org/pydif/cs207-FinalProject.svg?branch=master)](https://travis-ci.org/pydif/cs207-FinalProject)

[![Coverage Status](https://coveralls.io/repos/github/pydif/cs207-FinalProject/badge.svg?branch=master)](https://coveralls.io/github/pydif/cs207-FinalProject?branch=master)

### Installation

PyDif is currently in development. It is available on PypI. Pip is the prefferred way to install PyDif. 

```
pip install pydif
```
See the "How to Use Package" section for guidance for the correct imports. 



Alternatively you can also clone or otherwise download our project repo from Github, but you may run into python path issues:
https://github.com/Shamazo/PyDif

If you do not have numpy installed, run the following:

```
pip install -r requirements.txt
```

Once numpy is installed, you can navigate to the cs207-FinalProject folder and start writing your software in that directory.

Please refer to the "How to Use Package" section for guidance on how to correctly import the relvant modules and how to use them in your software.




### Background

#### Chain Rule

In calculus,  the derivative of the composition of two or more functions is defined as:

$$\dot{F}(x) = \dot{g}(x)\dot{f}(g(x))$$

In automatic differentiation, this rule is frequently utilized. All functions, even seemingly simple ones, use this rule. For example, $F(x) = sin(x)$ can be rewritten in the form $f(g(x))$ where $g(x)$ is simply $x$ and $f(x)$ is $sin(x)$. Thus, we have:


$$\dot{F}(x) = \dot{g}(x)\dot{f}(g(x))$$
$$\dot{F}(x)  = 1 *cos(x)$$

where 1 is $\dot{g}(x) = 1$ and $\dot{f}(x) = cos(x)$.

For a more complicated example, let us take $F(x) = sin(x^{2})$

$$ f(x) = sin(g(x))$$
$$ g(x) = x^{2}$$
$$ \dot{f}(x) = cos(g(x))$$
$$ \dot{g}(x) = 2x$$
$$\dot{F}(x) = \dot{g}(x)\dot{f}(g(x))$$
$$\dot{F}(x) = 2xcos(x^{2})$$

Using this rule, one can calculate the derivatives of increasinly complex functions, all while only needing to calculate the derivatives of a series of elementary functions.

#### Computational Graph

The computation graph describes the order of calculations done to compute the derivative. For example consider the funcion $f\left(x, y, z\right) = \dfrac{1}{xyz} + \sin\left(\dfrac{1}{x} + \dfrac{1}{y} + \dfrac{1}{z}\right)$ where we want to evaluate the partial derivatives at (1, 2, 3). The table representation of the graph would look like this. Each row of the table represents one elementary step in the calculation. The function in each row is an elementary function on a combination of earlier rows, which lets us step by step build up the derivative by repeatedly applying the chain rule and at the same time we can also evaluate the function. The table can also be presented in a graph format, but this quickly becomes unwiedly for complicated functions. 


| Trace | Elementary Function | Current Value | Elementary Function Derivative | $\nabla_{x}$ Value  | $\nabla_{y}$ Value  | $\nabla_{z}$ Value  |
| :-------: | :-----------------: | :-----------: | :----------------------------: | :-----------------: | :-----------------: | :-----------------: |
| $x_{1} = x$ | $x_{1}$ | 1 | $\dot{x}_{1}$ | $1$ | $0$ | $0$ | 
| $x_{2} = y$ | $x_{2}$ | 2 | $\dot{x}_{2}$ | $0$ | $1$ | $0$ | 
| $x_{3} = z$ | $x_{3}$ | 3 | $\dot{x}_{3}$ | $0$ | $0$ | $1$ | 
| $x_{4}$ | $1/x_{1}$ | 1 | $-\dot{x}_{1}/x_{1}^{2}$ | $-1$ | $0$ | $0$ | 
| $x_{5}$ | $1/x_{2}$ | 1/2 | $-\dot{x}_{2}/x_{2}^{2}$ | $0$ | $-1/4$ | $0$ | 
| $x_{6}$ | $1/x_{3}$ | 1/3 | $-\dot{x}_{3}/x_{3}^{2}$ | $0$ | $0$ | $-1/9$ | 
| $x_{7}$ | $x_{4} * x_{5}$ | 1/2 | $\dot{x_4}x_5 + x_4 \dot{x_5}$ | $-1/2$ | $-1/4$ | $0$ | 
| $x_{8}$ | $x_{7} * x_{6}$ | 1/6 | $\dot{x_7}x_6 + x_7 \dot{x_6}$ | $-1/6$ | $-1/12$ | $-1/18$ | 
| $x_{9}$ | $x_{4} + x_{5} + x_{6}$ | 11/6 | $\dot{x_4} + \dot{x_5} + \dot{x_6}$ | $-1$ | $-1/4$ | $-1/9$ | 
| $x_{10}$ | $Sin(x_9)$ | 0.9657 | $\dot{x_9}Cos(x_9)$ | $0.2595$ | $-0.06488$ | $-0.02883$ | 
| $x_{11}$ | $x_{10} + x_{8} $ | 1.1324 | $\dot{x_{10}} + \dot{x_8}$ | $0.0928$ | $-0.0184$ | $-0.0267$ | 



#### Dual Numbers

Much like how imaginary numbers of the form $x+yi$ are an extension of the real plane that have some very useful properties, dual numbers are another interesting type of numbers of the from $x+y\epsilon$.

Similar to how we defined $i$ to have the property that $i^2 = -1$, the $\epsilon$ in dual numbers is defined such that $\epsilon^2 = 0$ (note: $\epsilon$ here is not 0!). A neat property of dual numbers can be illustrated with a very simple example.

Consider $(x+yi)^2$:
$$ (x+yi)^2 = x^2 + 2xyi + y^2i^2 = x^2 + 2xyi + y^2 $$
In particular, notice how the magnitude of the imaginary component ($y$) "feeds into" the real value of the result (the +y^2 contribution).

Now instead, consider $(x+y\epsilon)^2$:

$$ (x+y\epsilon)^2 = x^2  + 2xy\epsilon + y^2\epsilon^2 = x^2 + 2xy\epsilon (\text{recall:} \epsilon^2 = 0!) $$
Notice how with dual numbers, the magnitude of the dual component ($y$) _does not_ feed into the real value of the result.

Amongst other things, this quirk of the dual number makes it particularly suited for automatic differentiation. Consider the following:

Evaluate $y = x^2$ when $x\leftarrow x + \epsilon x^{\prime}$.

$y = (x + \epsilon x^{\prime})^2$

$y = (x^2 + 2 x x^{\prime} \epsilon + x^{\prime^2} \epsilon ^2)$

$y = (x^2 + 2 x x^{\prime} \epsilon)$

Notice here that by evaluating $y = x^2$ at $x\leftarrow x + \epsilon x^{\prime}$, we have calculated its derivative ($2x$)! Neat! 

In our implementation of automatic differentiation, dual numbers is used to keep track of a function's value and it's derivative simulatenously.


#### Elementary Functions

In order to evaluate derivatives of a variety of different functions iteratively through the chain rule, we ultimately need to rely on a minimum set of so-called "elementary functions." These elementary functions consist of arithmetic operations (+, -, x, /) as well as atomic-sized, differentiable functions of a single variable that can be used as building blocks for more complex functions.



Here is a list of elementary functions:

* Addition, subtraction, multiplication, division
* Absolute Value
* Powers
* Roots
* Exponential
* Log
* Trigonometric Functions
* Inverse Trigonometric Functions
* Hyperbolic trig functions
* inverse hyperbloc trig functions
* sigmoid/ReLU


Explicitly defining the evaluation of these functions and their derivatives makes it possible to use the chain rule to evaluate the derivative of any differentiable function that is a combination of multiple elementary functions.

#### Seed Vectors


The use of seed vectors becomes apparent when considering more complicated examples. Consider a vector function $\textbf{f}(\textbf{x})$ that takes a vector input $\textbf{x} = (x_1, x_2)$:

$$\textbf{f(x)} = \begin{bmatrix} 2x_1+2x_2 \\ x_1x_2\end{bmatrix} $$

Let $\textbf{x}_{o} = (x_{o_1}, x_{o_2})$ be the outputs from the function. And define a directional derivative $D_{p}$ and seed vector $\textbf{p}$ such as follows:

$$D_{p}x_{o_i} = \sum_{j=1}^{2}{\dfrac{\partial x_{o_i}}{\partial x_{j}}p_{j}}$$

Computing the directional derivative for $\textbf{f}(\textbf{x})$:

$D_{p}x_{o_1} = {\dfrac{\partial x_{o_1}}{\partial x_{1}}p_{1}} + {\dfrac{\partial x_{o_1}}{\partial x_{2}}p_{2}}  = 2p_{1} + 2p_2$

$D_{p}x_{o_2} = {\dfrac{\partial x_{o_2}}{\partial x_{1}}p_{1}} + {\dfrac{\partial x_{o_2}}{\partial x_{2}}p_{2}} = x_2p_1 + x_1p_2$

This can be written more compactly as the Jacobian: $\textbf{J}=$
$
  \begin{bmatrix}
        2p_{1}   & 2p_2                        \\
        x_2p_1 & x_1p_2
      \end{bmatrix}
 $
 
Notice that by choosing different values for the seed vector $\textbf{p}$ allows us to select different components of the Jacobian! We can recover ${\dfrac{\partial x_{o}}{\partial x_{1}}}$ by choosing $\textbf{p} = (p_1, p_2) = (1,0)$, and recover ${\dfrac{\partial x_{o}}{\partial x_{2}}}$ by choosing $\textbf{p} = (0,1)$. We can also recover the full Jacobian by choosing $\textbf{p} = (1/\sqrt{2},1/\sqrt{2})$.

Typically, we would only be interested in the action of the Jacobian on some vector. The use of this seed vector allows us to compute the components of the Jacobian that we are interested in! We can however always still recover the full Jacobian by choosing an appropriate seed vector.

### Software Organization
The directory structure of pydif

  ```
     pydif\
           pydif\
                 __init__.py
                 pydif.py
                 test_pydif.py
                  dual/
                       __init__.py
                      dual.py
                      test_dual.py
                  elementary/
                       __init__.py
                      elementary.py
                      test_elementary.py
                   optimize/
                       __init__.py
                       optimize.pu
                       test_optimize.py
                   
           README.md
           setup.py
           LICENSE.txt
           requirements.txt
  ```
  
Each of our modules has a corresponding test file, and testing is performed through pytest. We use TravisCI for continuous integration and Coveralls for assessing coverage. pydif.py contains the core of our package and is where we actually implement automatic differentiation. dual.py contains our implementation of dual numbers, and elementary.py contains the elementary functions. 

## How to use package

An example usage of the package is provided as newton-example.py in the project repository. The following examlpe assumes that the package has been installed via github. The prefferred method to install pydif is with pip i.e pip install pydif. When the package is installed with pip you do not need to alter the python path which is what the first three lines of the example below are doing. 

The mantra to import the autodiff object and the relevant elementary functions:

```
import sys
import os
sys.path.append(os.path.join(os.getcwd(),'pydif'))
from pydif.pydif import autodiff #import autodiff objects
from pydif.elementary import elementary as el #imports the elementary functions
```
We first define the function that we wish to evaluate:

```
def f(x):
        return x - el.exp(-2.0 * el.sin(4.0*x) * el.sin(4.0*x))
```
Then instead of having to manually define the jacobian, we can instead define a autodiff object that automatically evaluates the derivative.
```
dfdx = autodiff(f)
```

The derivative of the function at any point can then be retrieved as simply as follows!

```
gradient = dfdx.get_der(point, jacobian=True)[0]
```

The evaluation of gradient from the autodiff object at a `point` using `jacobian=True` returns a list of the partial derivatives. Since this function only has one input parameter `x`, we simply take the first element from this list each time.

## Elementary functions
In order to implement automatic differentiation functions must be wrapped with library specific code. This means, for example, that you cannot use sin or np.sin in your code, rather you need to use the pydif implemention of sin. Pydif currently supports the following elementary functions which correspond to numpy functions of the same name.



*   cos(x)
*   sin(x)
*   tan(x)
* arccos(x)
* arcsin(x)
* arctan(x)
* cosh(x)
* sinh(x)
* tanh(x)
* arccosh(x)
* arcsinh(x)
* arctanh(x)
* exp2(x)
* log(x)
* log2(x)
* log10(x)
* sqrt(x)

In addition to the above functions, pydif also supports the logistic sigmoid and relu functions, which are frequently used as activation functions in neural nets. The logistic sigmoid is implemented as $\frac{1}{1 + e^{-x}}$ and the relu function is 0 for x <0 and x for x > 0. We define the derivative of the relu function to be 0 at x = 0. 

* sigmoid(x)
* relu(x)

These functions are used like so:



```
from pydiff.elementary import elementary as el 

def example_func(x):
      return 5 * el.cos(x) + el.log10(x)
```





## Dual Numbers

Pydif uses Dual numbers to quickly calculate derivatives of complicated functions. In order to instantiate an object of the class Dual,  one must pass in three arguments: the value of a function at a point, an array or array-like object of partial first derivatives at a point and an array or array-like object of partial second derivatives at a point. Pydif duals currently support the following elementary operations:

*  addition
*  subtraction
* multiplication
* division
* exponentiation
* negation
* equality
* inequality

These functions are used like so:

```
import pydif.dual as Dual 

a = Dual(1, 2, 3)

print(a)
>>> [1, 2, 3]

b = Dual(1, [3, 4, 5], [4, 5, 6])

print(b)
>>> [1, [3, 4, 5], [4, 5, 6]]
```





## AutoDiff & @diff, @diffdiff

To actually evaluate the value and derivates of functions or vector functions, autodiff objects have to be defined. These autodiff objects are initialized with a function or the vector function of interest. Values and derivates are then evaluated at specific points by calling functions of the autodiff object.

Alternatively, the function of interest can be decorated to return its derivatives.

autodiff objects expose the following functions to the users for function evaluation:

* get_val(pos, direction=None)

This evaluates the function of interest at the specified position.

* get_der(pos, wrt_variables=False, direction=None, order=1):

This evaluates the first or second derivate of the function at the specified position. The derivative can be evaluated in several modes:
Assume the function being evaluated is $u(a)$, $u(x,y)$.

1. Vanilla: $\frac{d}{d a}u(a)$ (just takes the derivative of the function)
2. Jacobian: $\frac{\partial}{\partial x}u(x,y)$, $\frac{\partial}{\partial y}u(x,y)$ (returns the partial derivates of the function)
3. Diagonal Elements of Hessian: $\frac{\partial}{\partial x^2}u(x,y)$, $\frac{\partial}{\partial y^2}u(x, y)$
4. DIrectional Derivates: $d \cdot u(x, y)$ or $d \cdot \frac{\partial}{\partial x^2}u(x,y)$ (returns the dot production of a direction vector and the value or partial derivaties of the function)

Currently supports scalar and vector functions of scalar input and scalar or vector output. Several key current limitations are noted below.

These functions are used like so:

```

from pydif.pydif import autodiff


def func(x,y,z):
    return 1/(x + y + z)
    
    
ad = autodiff(func)


pos = [0,1,13] #evaluate at the position (x,y,z) = (0,1,13)
direction = [0,0,1] #evaluate along the direction (0,0,1)


ad.get_val(pos) #returns the value of the function evaluated at the position
ad.get_val(pos, direction = direction) #returns the dot product of the value of the function at the position and the direction vector


ad.get_der(pos) #returns the derivative of the function evaluated at the position

ad.get_der(pos, wrt_variables=True, order=1) #returns the first order partial derivates of the function evaluated at the position

ad.get_der(pos, wrt_variables=True, order=2) #returns the second order partial derivates of the function (only hessian diagonals) evaluated at the position

ad.get_der(pos, wrt_variables=True, direction = direction) #returns the dot product of the first order partial derivates of the function evaluated at the position and the direction vector
```

Alternatively, pydif also provides decoraters to evaluate the derivative of functions like so:
```
from pydif.pydif import diff, diffdiff

def func(x,y,z):
    return 1/(x + y + z)
    
#always evaluates the first order partial derivatives of the function
@diff
def func(x,y,z):
    return (1/(x + y + z))
    
#always evaluates the second order partial derivatives of the function (hessian diagonal)
@diffdiff
def func(x,y,z):
    return (1/(x + y + z))
    
```

Pydif also supports scalar and vector functions and outputs like so:
```
from pydif.pydif import diff, diffdiff, autodiff_vector

#scalar function of vector output
def f1(x,y):
    return x, y+1
    
#scalar function of vector output
def f2(x,y):
    return x+1, y
    
#scalar function of vector output
def f3(x):
    return x, 0
    
#vector function
f4 = [f1, f2, f3]

#autodiff object of vector function of vector outputs
ad = autodiff_vector(f4)
ad.get_val(pos=(3,4))
```

In particular, please note the following limitations:
* _only functions with scalar input are supported, a vector input should be appropriately unpacked._ e.g. instead of a function defined with a vector input like f(x) where x is a vector, instead consider defining the function as f(x_1, x_2) where x_1 and x_2 are the components of the x vector explicitly unpacked when specified as input arguments to the function.

* _vector functions can have a different number of input arguments but must be defined with the variables in the same order._ e.g. assume a vector function of the form [f1, f2]. where f1 has the signature f1(x, y, z) and f2(x, y). Although the functions can take in a different number of input arguments, the functions must be defined with input arguments in the same order (so f2(x, y) is fine, but not f2(y, x)).

* _all components of the vector function can only be evaluated at the same position and in the same direction_* e.g. assume a vector function of (f1(x,y), f2(x)) cannot be evaluated at ((1,2),(3)), it can only instead be evaluated at ((1,2),(1)) or ((3,2),(3)). 



## Optimize

One widely applicable use of these derivatives is the optimization of various functions. Pydif allows a user to not only find a functions optima, but can also output a graph of the history of the optimization method's path. If the user simply wishes to find the optima, or to create their own visualization, they can instantiate the optimize class and call the method wrapping whichever technique they would like to use. The history is only returned if return_iters is set to True in this method call. If a standard graphic output is desired, the user can instead call the plot_optimization function. This method is a wrapper for the other methods, allowing the user to specify the name of the optimization technique they wish to use and get a premade graph of the optimization technique's path to the optima. Note that this method uses the default parameter for each optimization technique: a step size of .1, 100 iterations max, a precision of .001 and does not return the path of the technique. BFGS does not have a step size since the algorithm dynaically determines the optimal step size at each iteration. 

When return_hist is set to true the optimizers will return the value of the parameters of the function being optimized at each iteration as the second value and the found optimal paramters as the first value. 

Pydfi.optimize currently supports the following techniques:

* Newton's method as newton
* Steepest descent as steepest_descent
* Gradient descent as gradient_descent
* Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm as BFGS

The module is used as shown below:

```
from pydif.elementary import elementary as el
from pydif.optimize import optimize

def func(x,y,z):
      return x**2 + y**2 + z**2
    

opt = optimize(func)

opt.gradient_descent([1,2], step_size=0.1, max_iters=100, precision=0.001, return_hist=False)


opt.newton([1,2], step_size=0.1, max_iters=100, precision=0.001, return_hist=False)

opt.BFGS([1,2], step_size=0.1, precision=10**-8, return_hist=False)


```

Below are link for plots for 2 of these methods, gradient descent and BFGS. Each plot shows the path the method took to reach the optima of a Rosenbrock function, defined below, starting at (0,1). Gradient descent does not perform well on Rosenbrock functions due to the local optima.

$* f(x) = 100(y-x^2)^2 + (1-x)^2$

![BFGS](https://drive.google.com/open?id=1DhX9XiOBNCTUJvB84l6bAgnCprzQIyTL)

![Gradient Descent](https://drive.google.com/open?id=1o0q75WkdS2Zj81Jzh1kKkmE1L6Q2SS7R)

 

# Implementation Details
## General notes
This library only depends on NumPy. It is tested with version 1.15.4.

## Duals

The Dual class is used by pydif to quickly take the derivative by storing the value of a function at a point, as well as the partial derivatives and second deriatives at that point. 

The first argument is the value of the function and the second is a numpy array of partial first derivatives. The final argument is an array of partial second derivatives. Numpy is used to allow for componentwise operations across partial derivatives.

The class itself consists of two attributes: val, der and der2, identical to the above arguments. Dual.val is a float that contains the value of the function at a point. Dual.der on the hand contains a numpy array of partial derivatives at a point and Dual.der2 is a numpy array of the second partial derivatives, but only with respect to the same variable twice. i.e the array has $\frac{df}{dx^2}$ but not $\frac{df}{dxdy}$

The elementary operators for addition, subtraction, multiplication, division, power and negation were all overloaded, as well as the reverse operators. For the derivatives, operators were overlaoded to adhere to properties of differentiation, such as the product rule for mulitplication. 

Each operator first tries to operate assuming the other input is a dual number and falls back to scalar operations if necessary. 




## Elementary functions

Elementary functions are small wrappers around their numpy equivilants to support operations on Dual numbers. Each function is a try except clause which first trys the input as a dual number and falls back to treating it as a normal python numeric type. 


## AutoDiff & @diff, @diffdiff

The autodiff class and @diff decorators in pydif.py implement the necessary logic to actually use these dual numbers to evaluate the value and derivatives of functions at specified position in the specified direction.

Autodiff objects can be initialized by providing a function of single variables of the form f(x,y,z,...) or a vector of functions. The autodiff objects can then be evaluated at specified positions to find the value of the function, and its derivaties. Autodiff objects implement the necessary logic to initialize the evaluate the necessary dual numbers and perform basic checks on the dimentionality of the object inputs. When autodiff objects are initialized, the function and the number of parameters that the function takes as input are stored as attributes of the autodiff object.

Autodiff objects have two main functions that will be called get_val() and get_der() to evaluate the value, first and second order derivates of the function respectively. 

When either of these functions are called, autodiff objects first check the requested position to evaluate the function at to ensure that the shape matches the function call signature, a value error is thrown otherwise. This logic is handeled internally by autodiff objects by passing the requested position to evaluate the function at to an internal function (_check_dim()) which does the required shape comparison. Additionally, if the autodiff objects are to be evaluated in a requested direction, the same dimentionality check is performed on the specified direction vector (again, using _check_dim()). Additionally, the user specified direction function is normalized by the autodiff objects with an interal function (_enforce_unitvector()).

Then, the autodiff object calls the _eval() internal function with the necessary parameters to actually evaluate the function at this position. Notice that regardless of whether get_val() or get_der() are called, both the value and derivatives are calculated each time. Within the _eval() internal function, autodiff objects initialize the appropriate number of dual numbers in the appropriate fashion (e.g. with single valued derivative attribute if not requesting the derivatives be evaluated with respect to the variables, or two numpy arrays for the first derivative and second derivative if requiring the first or second order derivatives (to keep track of all the partial derivatives)). Autodiff objects then call the function using these initialized dual numbers as input and collect the output from these functions, which are also dual numbers! The details of dual number operations have already been described above when talking about the Dual class. Either the real attributes of the dual numbers (Dual.val) or the derivative attributes of the dual numbers (Dual.der, Dual.der2) are returned depending on which of the evaluation routines was called and with what parameters. If a direction to evaluate the function in was specified, a dot product of the output and the direction is performed before returning the result.

For evaluation of the derivaties with respect to the variables, the result is thus returned as a list of partial derivatives with the order the same as the function call signature. For example, consider a function of the form f(x,y), then the call to get_der() with wrt_variable=True, order=1, direction=[0,1] will return a numpy array of two partial derivatives evaluated in a direction ($0,\frac{\partial}{\partial y}$, notice the same order as function call definition) or a call to get_der() with wrt_variable=True,order=2 will return a numpy array of two partial derivaties ($\frac{\partial}{\partial x^2}, \frac{\partial}{\partial y^2}$).

For evaluation of vector functions, pydif exposes autodiff_vector() objects. Autodiff_vector objects can be initialized by providing a vector of functions. The component functions of the vector function must satisify the same requirements for scalar functions when using autodiff objects. The autodiff_vector object handles the necessary logic to construct a vector of autodiff objects of the same shape at initialization (held as an attribute self.func_vector). Autodiff_vector objects provide the same functions that will be called to evaluate the value (get_val()) and first and second order dervatives of the function (get_der()). When either of these functions are called, the autodiff_vector objects create an empty res numpy array of the same shape as the vector input. This result array is filled out by iterating over the array component by component, calling the same function on the component autodiff objects with appropriate formatted position and direction and storing the result in the appropriate position in the array. The position and direction are trimmed as required by calling an inner function (_clean_dim()). This allows for the input specification of some components of the vector function to be a subset of the largest input specification as long as the order is preserved (e.g. allows for a vector function of the form $[f1(x,y), f2(x)]$ but not $[f1(x,y),f2(y)]$). This result frame is the returned. Thus, the autodiff_vector can be thought of as a convenience tool that provides a way to handle evaluating the value, derivatives of multiple autodiff objects at the same position and in the same direction.

Similarly, the @diff decorators wrap a given function to replace function inputs with equivalent dual numbers and return the appropriate attributes of the result. Both the @diff and @diffdiff wrappers call the _eval_func() inner function to handle the initialization of the appropriate dual number objects and the collection of the appropriate attributes of the resultant dual number objects.

# Future
In future releases of this library we hope to implement the backwards mode of automatic differentiation. Consider a function $f: R^n \rightarrow R^m$

Backwards mode is more efficient at computing gradients of functions with $n > > m $. Forward mode scales O(n) with the number of input variables if we want to find all the partial derivatives, since we need to run the computation with n different seed vectors.  Backwards mode starts from the end of our compuational graph and works backgrounds and runs in O(m) time. Many applications have functions of many to a few variables, most notably neural networks which can be expressed as functions with thousands of input variables but only a few output variables. This would require a significant rewrite of our code as we would need to keep track of the values and gradients on the forward pass. 


Currently we support optimizing functions over all of the input variables. However, we would like to allow users to pass in a mask array to select with respect to which variables they want to optimize. This is a highly useful feature, since many machine learning methods are effectively functions with two sets of inputs, the weights and the features, but which we only want to optimize with respect to the weights at the training points. 

Currently our import statements are rather ugly. We would like to have numpy style imports e.g import numpy as np rather then have to write all the sub modules. This would require writing more elaborate \__init__.py files to hide the submodules from the user. 

In the future we would like to support the full hessian function, currently we can only calculate second derivatives with respect to the same variable. Implementing the full hessian was significantly harder than we anticipated and appears to require multiple passes on the computation graph. 

We intend to support a basic cache in future versions. Currently, we re-evaluate the value, first and second derivatives of the function each time a value or derivative is requested. Instead, a basic cache for each autodiff instance could keep a record of the result of evaluating the autodiff object (and by extension the function) at the last few positions.
