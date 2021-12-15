# am226-project-stable-memory
(Reproduce) Stable Memory with Unstable Synapses

* GitHub repo of the original paper: [https://github.com/lsusman/stable-memory](https://github.com/lsusman/stable-memory)
* data.mat: pre-initialized network, including x, W, rcseed (rate-control random number seed), and deseed (decorrelation random number seed)
* eigenshuffle.m: a script provided by John D’Errico, freely available online for sorting
eigenvalues
* eigenshuffle-license: open-source license of eigenshuffle.m
* rate_control_learning.m simulates
the homeostatic rate-control case and Hebbian learning by STDP with external stimulation. It
produces a plot corresponding to Fig 2A in the supplementary materials of the paper and shows
Hebbian learning by STDP embeds persistent imaginary-coded memory. rate_control_learning(modified).m 
is based on it but includes the code that can produce Fig 2B.
* decorrelation_learning.m is
similar but simulates the decorrelation homeostasis learning case. It produces a plot corresponding to Fig 3a
in the paper. decorrelation_learning(modified).m is based on it but includes the code that can produce Fig 3b. 
* erosion_real.m purely focuses on the erosion of real-coded memories over time, so
there is no external input b or learning process. It, together with erosion_imag.m, produces Fig 2c-e in the paper.
* erosion_imag.m is similar but focuses on imaginary-coded memories.
