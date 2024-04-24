# Readme file for MATLAB code for BEE Extended Essay
Modified based on [1] to adapt to my own needs.
Added APDAGD algorithm from [2].

# References
[1]"Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration"
-- Jason Altschuler, Jonathan Weed, Philippe Rigollet
(https://github.com/JasonAltschuler/OptimalTransportNIPS17)

[2]"Computational Optimal Transport: Complexity by Accelerated Gradient Descent 
Is Better Than by Sinkhorn’s Algorithm"
-- Pavel Dvurechensky, Alexander Gasnikov, Alexey Kroshnin
(https://github.com/chervud/AGD-vs-Sinkhorn)

# Algorithms 
-- compute_ot_lp.m    Compute optimal transport directly using a MATLAB linear program solver.\
-- round_transpoly.m  Implementation of algorithm for rounding to transport polytope. \
-- frobinnerproduct.m Compute sum of entry-wise products.

-- sinkhorn.m         Implementation of classical Sinkhorn-Knopp algorithm for matrix scaling.\
-- sinkhorn_tm.m      Same as above, with fixed time input.

-- greenkhorn.m       Implementation of Greenkhorn algorithm for matrix scalng.\
-- greenkhorn_tm.m    Same as above, with fixed time input.

-- APDAGD.m           Implementation of APDAGD algorithm.\
-- APDAGD_tm.m        Same as above, with fixed time input.

-- gcpb_ot.m          Implementation of SAG algorithm for discrete OT.\
-- gcpb_ot_tm.m       Same as above, with fixed time input.\
-- gcpb_compute_ot.m  Compute OT value for SAG algorithm.

# Plotting Scripts
-- rowcol_vs.m        Compares Sinkhorn-Knopp, SAG, Greenkhorn, and APDAGD under row/col updates regime.\
-- timegrid_vs.m      Compares Sinkhorn-Knopp, Greenkhorn, and APDAGD under fixed clock time.


# Input Generations
(all in 'input_generation/' sub-directory)\
Contains code creating OT instances out of images (either synthetically generated or MNIST).
