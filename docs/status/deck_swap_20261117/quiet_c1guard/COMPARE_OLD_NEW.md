```
sessions: old 108, new 108

== quiet_min: per-animal epoch means and change vs own pre (old -> new)
           pre           acute         subacute         chronic         acute-pre         subacute-pre         chronic-pre        
           old     new     old     new      old     new     old     new       old     new          old     new         old     new
animal                                                                                                                            
PS92     4.978   4.995  11.610  11.614    3.491   3.541   3.957   4.045     6.632   6.619       -1.487  -1.454      -1.021  -0.950
PS93     8.971   9.014  13.481  13.501    6.325   6.363   6.094   6.200     4.510   4.487       -2.646  -2.650      -2.878  -2.814
PS94    10.349  10.375  22.339  22.485    8.577   8.710   8.101   8.119    11.991  12.111       -1.771  -1.665      -2.248  -2.256
PS95    44.996  45.194  53.650  53.829   24.643  24.716  22.339  22.508     8.654   8.635      -20.353 -20.478     -22.657 -22.686

== orofacial|rate: per-animal epoch means and change vs own pre (old -> new)
          pre          acute         subacute        chronic        acute-pre         subacute-pre        chronic-pre       
          old    new     old     new      old    new     old    new       old     new          old    new         old    new
animal                                                                                                                      
PS92    4.757  2.511  16.226  11.894   13.594  9.201  13.256  9.218    11.469   9.383        8.837  6.690       8.499  6.707
PS93    5.358  3.473  23.749  18.959    9.935  6.144   5.145  2.992    18.392  15.485        4.578  2.671      -0.212 -0.481
PS94    5.618  3.357  14.644  10.848    9.085  6.513   3.744  1.991     9.027   7.491        3.467  3.156      -1.873 -1.366
PS95    1.951  1.016  10.308   7.152    4.825  2.631   3.076  1.675     8.356   6.136        2.874  1.615       1.124  0.659

== orofacial|rate_both: per-animal epoch means and change vs own pre (old -> new)
          pre          acute         subacute         chronic         acute-pre         subacute-pre         chronic-pre        
          old    new     old     new      old     new     old     new       old     new          old     new         old     new
animal                                                                                                                          
PS92    8.503  4.505  35.289  26.551   24.998  16.717  24.881  16.803    26.786  22.045       16.495  12.212      16.379  12.298
PS93    9.919  6.610  47.362  37.100   20.581  13.019  12.710   7.588    37.443  30.491       10.662   6.410       2.791   0.979
PS94    8.223  4.837  29.950  22.371   18.255  13.368   7.451   3.993    21.727  17.534       10.033   8.530      -0.772  -0.844
PS95    4.889  2.590  26.132  17.890   10.043   5.592   6.938   3.926    21.243  15.300        5.154   3.002       2.049   1.336

== limb|rate: per-animal epoch means and change vs own pre (old -> new)
          pre         acute        subacute        chronic        acute-pre        subacute-pre        chronic-pre       
          old    new    old    new      old    new     old    new       old    new          old    new         old    new
animal                                                                                                                   
PS92    0.213  0.213  0.987  1.015    1.336  1.332   6.871  6.719     0.774  0.802        1.123  1.119       6.658  6.506
PS93    0.054  0.054  1.404  1.402    0.080  0.080   1.039  1.036     1.350  1.348        0.026  0.026       0.985  0.982
PS94    0.663  0.662  1.127  1.121    2.041  2.068   1.597  1.593     0.464  0.459        1.378  1.406       0.934  0.931
PS95    0.054  0.053  0.037  0.037    0.192  0.197   0.127  0.127    -0.016 -0.016        0.138  0.144       0.074  0.074

== LMM epoch terms vs August pre (rate = log10(rate + 0.2)), old -> new
  measure          column                 term  coef_old  coef_new  q_old  q_new
quiet_min         session    C(epoch)[T.acute]    5.6829    4.7772 0.0215 0.3429
quiet_min         session C(epoch)[T.subacute]   -7.4946   -6.4716 0.0020 0.1969
quiet_min         session  C(epoch)[T.chronic]   -7.1616   -7.1945 0.0001 0.1343
     rate            limb    C(epoch)[T.acute]    0.2806    0.2820 0.1703 0.1659
     rate            limb C(epoch)[T.subacute]    0.3705    0.3728 0.0607 0.0578
     rate            limb  C(epoch)[T.chronic]    0.6078    0.6065 0.0044 0.0054
     rate       orofacial    C(epoch)[T.acute]    0.5846    0.6914 0.0000 0.0000
     rate       orofacial C(epoch)[T.subacute]    0.3592    0.4079 0.0004 0.0026
     rate       orofacial  C(epoch)[T.chronic]    0.1263    0.1384 0.1713 0.2824
     rate  orofacial_left    C(epoch)[T.acute]    0.6086    0.7363 0.0000 0.0000
     rate  orofacial_left C(epoch)[T.subacute]    0.3569    0.4182 0.0000 0.0002
     rate  orofacial_left  C(epoch)[T.chronic]    0.1538    0.1944 0.0287 0.0378
     rate orofacial_right    C(epoch)[T.acute]    0.5572    0.6607 0.0000 0.0000
     rate orofacial_right C(epoch)[T.subacute]    0.3243    0.3718 0.0008 0.0015
     rate orofacial_right  C(epoch)[T.chronic]    0.0813    0.1049 0.3070 0.2861
rate_both            limb    C(epoch)[T.acute]    0.2868    0.2777 0.1509 0.1427
rate_both            limb C(epoch)[T.subacute]    0.3597    0.3508 0.0606 0.0538
rate_both            limb  C(epoch)[T.chronic]    0.6239    0.6120 0.0012 0.0008
rate_both       orofacial    C(epoch)[T.acute]    0.6623    0.7838 0.0000 0.0000
rate_both       orofacial C(epoch)[T.subacute]    0.4105    0.4682 0.0000 0.0000
rate_both       orofacial  C(epoch)[T.chronic]    0.2007    0.2141 0.0052 0.0219
rate_both  orofacial_left    C(epoch)[T.acute]    0.6856    0.8360 0.0000 0.0000
rate_both  orofacial_left C(epoch)[T.subacute]    0.4038    0.4646 0.0000 0.0000
rate_both  orofacial_left  C(epoch)[T.chronic]    0.2233    0.2595 0.0017 0.0004
rate_both orofacial_right    C(epoch)[T.acute]    0.6264    0.7467 0.0000 0.0000
rate_both orofacial_right C(epoch)[T.subacute]    0.3926    0.4291 0.0000 0.0000
rate_both orofacial_right  C(epoch)[T.chronic]    0.1595    0.1753 0.0316 0.0092
 rate_neg            limb    C(epoch)[T.acute]    0.0638    0.0542 0.6027 0.6300
 rate_neg            limb C(epoch)[T.subacute]    0.1191    0.1062 0.3476 0.4104
 rate_neg            limb  C(epoch)[T.chronic]    0.3042    0.2902 0.0090 0.0061
 rate_neg       orofacial    C(epoch)[T.acute]    0.7693    0.8799 0.0000 0.0000
 rate_neg       orofacial C(epoch)[T.subacute]    0.4664    0.5226 0.0000 0.0000
 rate_neg       orofacial  C(epoch)[T.chronic]    0.2914    0.2935 0.0000 0.0010
 rate_neg  orofacial_left    C(epoch)[T.acute]    0.7857    0.8969 0.0000 0.0000
 rate_neg  orofacial_left C(epoch)[T.subacute]    0.4364    0.4849 0.0000 0.0000
 rate_neg  orofacial_left  C(epoch)[T.chronic]    0.3003    0.2884 0.0000 0.0000
 rate_neg orofacial_right    C(epoch)[T.acute]    0.7188    0.8341 0.0000 0.0000
 rate_neg orofacial_right C(epoch)[T.subacute]    0.4481    0.4931 0.0000 0.0000
 rate_neg orofacial_right  C(epoch)[T.chronic]    0.2490    0.2432 0.0000 0.0004
v2_quiet_lp_20261117: chosen k (median of the leave-one-animal-out choices, positive events): {'orofacial': 3.75, 'limb': 8.75}
v2_quiet_lp_c1guard_20261117: chosen k (median of the leave-one-animal-out choices, positive events): {'orofacial': 4.5, 'limb': 8.75}
```
