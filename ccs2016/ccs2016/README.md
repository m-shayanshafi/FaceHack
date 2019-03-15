## Dependencies

* MatConvNet (<http://www.vlfeat.org/matconvnet/>), a convlutional
neural networks library for Matlab.

* dlib for Python (<https://pypi.python.org/pypi/dlib>) is used for
landmark localization and image alignment. Here's a good tutorial for
how to install dlib: <https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/>.

## How to run

To run the provided demo:
1. Run `init.m` to initialize the MatConvNet toolbox and add relevant
paths.
2. Run demos of digital and physical attacks by running `demo.m`. The
demos use the neural network that was provided by the VGG group
(See: <http://www.robots.ox.ac.uk/~vgg/software/vgg_face/>).

The images in the `data/` folder have been aligned already, so one can 
run the demo before installing dlib. To align new images, the code in
`preprocess_data.m` can be used.

**Note:** There are paths in `init.m`, `demo.m`, and
`preprocess_data.m` that need to be updated (e.g., MatConvNet's 
installation path in `init.m`). The comment `% update me` was added to 
denote these paths.
