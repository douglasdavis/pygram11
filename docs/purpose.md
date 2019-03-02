# Purpose

There are more than enough histogramming packages for python out in
the wild, and some have _a lot_ of functionality beyond dropping data
into bins. This package is _just_ for dropping data into bins, but as
fast as possible (on a CPU), while also keeping a simple code
base. Another property of histograms lacking from other options is the
ability to retrieve the sum of weights in each bin. Finally, I thought
it would be fun to learn how to write software with pybind11.
