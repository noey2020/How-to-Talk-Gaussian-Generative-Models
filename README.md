# How-to-Talk-Gaussian-Generative-Models

December 26, 2020

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!

Hire me! 😊

- So we have just seen the multivariate Gaussian,
a popular and powerful probability distribution
for data of arbitrary dimension.
Today, we'll look at using these distributions
to build classifiers.
Now, we've gone over the generative approach
to classification, in which we fit
a probability distribution to each class individually.
What we'll look at today is
the kinds of decision boundaries
that result from doing this.
So let's start by going back to the winery data.
If you remember, this was a dataset with three classes,
representing three different wineries and 13 features.
We started by just picking
one of the features, alcohol level,
and we've fit a one-dimensional Gaussian to this feature
for each of the three classes.
The resulting classifier had an error rate of
29% on the test set, not very good at all.
Then we added a second feature, flavonoids,
and we've fit a bivariate Gaussian to each class,
and that's what you see over here,
the three ellipsoids that resulted.
And the classification boundary, the decision boundary
turned out to be this.
So, the points over here get classified as red,
the points over here get classified as green,
the points over here get classified as black,
and adding the second feature
made the test error drop
from 29% to just 8%.
But now that we have
the multivariate Gaussian under control,
we can use all 13 of the features.
So we can fit a multivariate Gaussian
to the 13 features for winery one
and an other one for winery two,
and another one for winery three.
And then apply Bayes Rule for Classification.
If we do that, we see that the test error
actually goes down to zero.
Does that mean that it's a perfect classifier?
No, it doesn't really mean that,
but one thing the test said, it's not that large,
it consists just of 48 points.
But it does mean that it's a much better classifier
than what we were able to obtain
using just one or two features,
and it's a sign that it can be very beneficial
to include multiple relevant features.
So now, let's look at the decision boundaries
that you get in general
from Gaussian generative modeling.
So the multivariate Gaussian has this density
that we've gone over.
And as we saw, the first part of the density
is really just a normalizing factor.
The important part is the stuff inside the exponent,
which is a quadratic function.
Now, that quadratic is in the exponent.
Is there a way to bring it down?
There is, if you want to bring it down,
what you need to do is to take the logarithm of the density.
Let's see what happens if we do that.
Let's take log of p of x.
So the first part just becomes some constant.
And the part in the exponent then becomes
negative 1/2 x minus mu
transpose sigma inverse x minus mu.
And so, log of px is a bona fide quadratic function.
As we'll see, what this means is that
if you use Gaussian distributions
in the generative approach to classification,
the decision boundaries you get will,
in general, be quadratic.
So let's see a little bit more closely
why that's the case.
Let's look at a binary situation for simplicity.
So there are just two classes,
and the first thing we do in the generative approach
is to estimate the class probabilities, pi one and pi two.
So maybe 60% of the data is from class one,
so pi one is .6, and pi two is .4.
Then we fit a Gaussian to each class, pi one and pi two,
and now, we're letting them be
arbitrary multivariate Gaussians.
So these are the parameters of the model.
Now, when a new point x comes along,
in order to classify x,
we compute pi one times p one of x,
and we compute pi two times p two of x,
and we look at which of these two is larger.
So, that's our classification rule.
Now, what we can do is we can plug in
the formula for the Gaussian density
into p one and into p two,
and then we can take the log of both sides,
which brings the quadratic functions down,
and if we do some algebraic simplifications,
it turns out that this is what we get.
This is the decision rule.
So let's see what this is.
So first, we have x transpose Mx,
and as we've seen, this generically a quadratic function.
So M is a matrix and of that,
we can say precisely what it is.
It's just the difference of
the inverse covariance matrices.
The second term is twice x transpose x,
so that's the same as w dot x,
and so that's a linear function.
And w at the end can just be read off
from the means and covariances.
And then, there's a threshold theta,
which is just a number like .2 or negative .6.
And that again can be read off from the parameters,
including pi one and pi two.
So what happens is, when you use this rule,
some points get classified as class one,
some points as class two,
and the decision boundary is the separating region
between these two zones.
The decision boundary corresponds exactly
to this equation,
x transpose MX
plus two w transpose x equals theta.
And this is a quadratic boundary.
Now, it turns out that, sometimes,
the matrix M is zero.
When will this happen?
Well, it would happen, looking at the equation form,
it would happen if
the two covariance matrices are identical.
In that case, M drops out,
the quadratic term drops out,
and we just get a linear decision boundary.
Now, a linear boundary is a special case,
a degenerate case of a quadratic boundary,
and let's start by looking at that linear case.
So, the linear boundary arises
when the two classes have exactly
the same covariance matrix,
sigma one equals sigma two.
Let's see an example of that.
Let's say that each class is a spherical Gaussian,
and they had exactly the same covariance matrix,
say, the identity matrix.
And two make the situation even more symmetric,
let's say that the two classes had
exactly the same class probability.
So pi one equals pi two equals .5.
What is the decision boundary in that case?
Well, it's just the perpendicular bisector
between the centers of the two classes,
between the two Gaussian means.
So the center of class one is mu of one,
the center of class two is mu of two,
and the perpendicular bisector between them
is the decision boundary,
and is exactly what one would expect,
given the extreme symmetry of this case.
But now, let's if we were to vary it a little bit.
Let say that we change this one thing.
Currently, we're saying that
the class probabilities are equal,
they're both .5.
What if class one is a little bit more likely,
say, pi one equals .6 and p two equals .4,
what happens then?
So here's what happens.
The decision boundary just moves slightly to the right,
and that's in pi this is.
So if you look at the points
smack between the two centers, this point over here,
this point would now get classified as class one
just because class one occurs more frequently
than class two.
And so, the decision boundary just get shifted
slightly to the right
to take care of this discrepancy
in prior class probabilities.
Now, what if the two classes
have the same covariance matrix,
but they're not spherical?
That's what we see in this picture here.
Because the covariance matrices are the same,
the boundary's again linear,
but now, it's not the perpendicular bisector any longer,
it's slightly skewed to accommodate the different shape.
So in all of these cases,
the classification rule is a simple linear rule
of the form w dot x greater or equal to theta,
and the vector w and the number theta can just be read off
from the parameters of the model,
from the means, the covariances,
and the weights, pi one and pi two.
Now, it turns out that the way in which
these models are often used is to set w in this way,
but to then allow theta, just that number, that threshold,
to vary a little, to tweak it a little
to maximize performance on the training set
or on some validation set.
So, in other words, you choose exactly this boundary,
The one shown over here,
but you allow it to shift parallel to itself
slightly to the left or right if that helps performance.
Okay, so this is the case where the covariances are equal.
What is they aren't equal?
If the covariances aren't equal,
then the boundary is a general quadratic.
And let's see some examples of that.
So, in this first example over here,
the two classes are both spherical Gaussians,
so that's nice and simple,
but they don't have the same covariance matrix.
So the one on the left has got a larger variance,
so maybe it's covariance matrix
is four times the identity,
whereas the one on the right
might just be the identity.
So the variance is higher on the left.
What kind of decision boundary do we get in this case?
Let's take a look.
So here's what results.
The decision boundary is a sphere.
Points inside the purple sphere
get classified as being class two,
whereas points outside the sphere are class one.
So this gets classified as two,
this gets classified as one,
and this point over here also gets classified as one.
Seems a little strange, right?
Well, perhaps it will be a little clearer
if we look at a one-dimensional analog of this.
So let's look at roughly the same picture in 1-d.
So, here we have two Gaussians and one dimension,
and one of them has got a very large variance,
the one for class one, whereas the other for class two
has a very small variance.
What is the decision boundary?
The data is one-dimensional.
Let's say the weights are the same,
pi one equals pi two equals .5.
In that case, this would be the boundary.
You look at where the densities overlap.
This is the boundary.
Everything over here is class one,
everything over here also gets classified as one,
and the stuff in the middle over here
gets classified as two.
Here's another example.
So one of the classes is a spherical Gaussian,
and the other one is a diagonal Gaussian.
So you have an ellipsoidal shape
because it's diagonal.
It's ellipsoidal but parallel to the axes.
And, in this case, the decision boundary
turns out to be a parabola.
And there are lots of other ways
the decision boundary could look,
but they're all quadratic forms of one kind or another.
So we've talked mostly about the binary case,
and if the case where there are just two labels,
and this is because it's particularly simple
to then talk about the boundary between these two regions,
class one and class two.
But when there are multiple labels,
the overall picture is roughly the same.
So let's say that there are k classes.
So what we do is that, for each class,
we fit a weight, Pi of j, that's just a number,
and we fit a Gaussian density, P of j.
So what's our classification rule in this case?
When a new point x comes along,
we compute pi j times P j of x,
and we pick the class j for each this is largest.
We can equivalently pick the class j
for which the log of this is the largest.
Doesn't change the outcome,
and it can be a little bit easier to think of
because, as we saw a little earlier,
log of the Gaussian density is a quadratic function.
So what happens essentially is that
each of the k classes has got its own quadratic function.
And when a new point comes along,
each class evaluates its quadratic function.
They each get a number,
and the one with the largest number is the winner.
That's the class that gets predicted.
So that's how classification works
when there are k classes with general Gaussians.
Now, in the situation where
all the covariance matrices of the k classes are the same,
then once again, the boundaries become linear.
So we've talked a lot about generative modeling,
about building a classifier by fitting
a Gaussian model to each class.
This is a very simple approach to classification,
and it works extremely well in many settings.
But the Gaussian isn't the only distribution
that can be used, and next time,
we'll look at some other choices.

I included some posts for reference.

https://github.com/noey2020/How-to-Talk-Multivariate-Gaussian

https://github.com/noey2020/How-to-Talk-Linear-Algebra-Review-3

https://github.com/noey2020/How-to-Talk-Linear-Algebra-Review-2

https://github.com/noey2020/How-to-Talk-Linear-Algebra-Review-1

https://github.com/noey2020/How-to-Talk-2D-Generative-Modeling

https://github.com/noey2020/How-to-Talk-Probability-Review-3

https://github.com/noey2020/How-to-Talk-Probability-Review-2

https://github.com/noey2020/How-to-Talk-Generative-Modeling-in-One-Dimension

https://github.com/noey2020/How-to-Talk-Probability-Review-1

https://github.com/noey2020/How-to-Talk-Generative-Approach-to-Classification

https://github.com/noey2020/How-to-Talk-of-Fitting-a-Distribution-to-Data-

https://github.com/noey2020/How-to-Talk-of-Host-of-Prediction-Problems

https://github.com/noey2020/How-to-Talk-of-Useful-Distance-Functions

https://github.com/noey2020/How-to-Talk-of-Improving-Nearest-Neighbor

https://github.com/noey2020/How-to-Talk-of-Prediction-Problems

https://github.com/noey2020/How-to-Talk-Matlab-Tricks-and-Tweaks

https://github.com/noey2020/How-to-Talk-Trading-and-Investing

https://github.com/noey2020/How-to-Work-in-Matlab-Development-Environment

https://github.com/noey2020/How-to-Talk-Vaccines

https://github.com/noey2020/How-to-Talk-Regression-in-Matlab

https://github.com/noey2020/How-to-Get-Started-in-Matlab

https://github.com/noey2020/How-to-Convert-Data-from-Web-Service-Using-Matlab

https://github.com/noey2020/Quote-for-the-Day

https://github.com/noey2020/How-to-Talk-Good-Investment-Strategy

https://github.com/noey2020/How-to-Talk-of-Good-Plan

https://github.com/noey2020/Thought-for-the-Day

https://github.com/noey2020/How-to-Talk-Stock-Watch-of-the-Day

https://github.com/noey2020/How-to-Talk-Data-Science

https://github.com/noey2020/How-to-Talk-Fundamental-Analysis

https://github.com/noey2020/How-to-Read-Company-Profiles

https://github.com/noey2020/How-to-Import-Data-from-Spreadsheets-and-Text-Files-Matlab-Without-Coding

https://github.com/noey2020/How-to-Talk-Model-of-Stock-Market-Prices-

https://github.com/noey2020/How-to-Talk-Digital-Wallets

https://github.com/noey2020/How-to-Talk-Investing

https://github.com/noey2020/How-to-Double-Your-Money-in-5years

https://github.com/noey2020/How-to-Talk-Matlab

I appreciate comments. Shoot me an email at noel_s_cruz@yahoo.com!
