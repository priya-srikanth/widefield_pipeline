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
          pre          acute         subacute         chronic         acute-pre         subacute-pre        chronic-pre       
          old    new     old     new      old     new     old     new       old     new          old    new         old    new
animal                                                                                                                        
PS92    4.757  4.733  16.226  16.201   13.594  13.838  13.256  13.222    11.469  11.468        8.837  9.105       8.499  8.489
PS93    5.358  5.340  23.749  23.729    9.935   9.995   5.145   5.145    18.392  18.389        4.578  4.655      -0.212 -0.195
PS94    5.618  5.628  14.644  14.667    9.085   9.114   3.744   3.830     9.027   9.038        3.467  3.486      -1.873 -1.799
PS95    1.951  1.956  10.308  10.459    4.825   4.864   3.076   3.099     8.356   8.503        2.874  2.908       1.124  1.143

== orofacial|rate_both: per-animal epoch means and change vs own pre (old -> new)
          pre          acute         subacute         chronic         acute-pre         subacute-pre         chronic-pre        
          old    new     old     new      old     new     old     new       old     new          old     new         old     new
animal                                                                                                                          
PS92    8.503  8.516  35.289  35.245   24.998  25.604  24.881  24.840    26.786  26.729       16.495  17.088      16.379  16.323
PS93    9.919  9.877  47.362  47.348   20.581  20.631  12.710  12.678    37.443  37.471       10.662  10.755       2.791   2.801
PS94    8.223  8.273  29.950  30.045   18.255  18.283   7.451   7.545    21.727  21.771       10.033  10.010      -0.772  -0.729
PS95    4.889  4.906  26.132  26.361   10.043  10.085   6.938   6.980    21.243  21.455        5.154   5.179       2.049   2.074

== limb|rate: per-animal epoch means and change vs own pre (old -> new)
          pre         acute         subacute         chronic         acute-pre        subacute-pre        chronic-pre        
          old    new    old     new      old     new     old     new       old    new          old    new         old     new
animal                                                                                                                       
PS92    0.213  5.242  0.987  12.225    1.336  11.300   6.871  18.735     0.774  6.983        1.123  6.058       6.658  13.493
PS93    0.054  4.580  1.404  12.640    0.080   3.965   1.039   7.894     1.350  8.060        0.026 -0.615       0.985   3.314
PS94    0.663  8.983  1.127   8.087    2.041  11.659   1.597   9.897     0.464 -0.896        1.378  2.677       0.934   0.914
PS95    0.054  3.854  0.037   4.124    0.192   3.014   0.127   2.322    -0.016  0.271        0.138 -0.839       0.074  -1.532

== LMM epoch terms vs August pre (rate = log10(rate + 0.2)), old -> new
  measure          column                 term  coef_old  coef_new  q_old  q_new
quiet_min         session    C(epoch)[T.acute]    5.6829    4.7772 0.0215 0.3429
quiet_min         session C(epoch)[T.subacute]   -7.4946   -6.4716 0.0020 0.1969
quiet_min         session  C(epoch)[T.chronic]   -7.1616   -7.1945 0.0001 0.1343
     rate            limb    C(epoch)[T.acute]    0.2806    0.1651 0.1703 0.2877
     rate            limb C(epoch)[T.subacute]    0.3705    0.1539 0.0607 0.3369
     rate            limb  C(epoch)[T.chronic]    0.6078    0.1848 0.0044 0.2614
     rate       orofacial    C(epoch)[T.acute]    0.5846    0.5860 0.0000 0.0000
     rate       orofacial C(epoch)[T.subacute]    0.3592    0.3616 0.0004 0.0003
     rate       orofacial  C(epoch)[T.chronic]    0.1263    0.1298 0.1713 0.2614
     rate  orofacial_left    C(epoch)[T.acute]    0.6086    0.6083 0.0000 0.0000
     rate  orofacial_left C(epoch)[T.subacute]    0.3569    0.3576 0.0000 0.0000
     rate  orofacial_left  C(epoch)[T.chronic]    0.1538    0.1526 0.0287 0.0767
     rate orofacial_right    C(epoch)[T.acute]    0.5572    0.5566 0.0000 0.0000
     rate orofacial_right C(epoch)[T.subacute]    0.3243    0.3239 0.0008 0.0007
     rate orofacial_right  C(epoch)[T.chronic]    0.0813    0.0786 0.3070 0.3139
rate_both            limb    C(epoch)[T.acute]    0.2868    0.2682 0.1509 0.0394
rate_both            limb C(epoch)[T.subacute]    0.3597    0.1737 0.0606 0.1892
rate_both            limb  C(epoch)[T.chronic]    0.6239    0.2306 0.0012 0.0632
rate_both       orofacial    C(epoch)[T.acute]    0.6623    0.6626 0.0000 0.0000
rate_both       orofacial C(epoch)[T.subacute]    0.4105    0.4107 0.0000 0.0000
rate_both       orofacial  C(epoch)[T.chronic]    0.2007    0.2011 0.0052 0.0112
rate_both  orofacial_left    C(epoch)[T.acute]    0.6856    0.6910 0.0000 0.0000
rate_both  orofacial_left C(epoch)[T.subacute]    0.4038    0.3933 0.0000 0.0000
rate_both  orofacial_left  C(epoch)[T.chronic]    0.2233    0.2248 0.0017 0.0002
rate_both orofacial_right    C(epoch)[T.acute]    0.6264    0.6282 0.0000 0.0000
rate_both orofacial_right C(epoch)[T.subacute]    0.3926    0.3942 0.0000 0.0000
rate_both orofacial_right  C(epoch)[T.chronic]    0.1595    0.1606 0.0316 0.0451
 rate_neg            limb    C(epoch)[T.acute]    0.0638    0.4644 0.6027 0.0000
 rate_neg            limb C(epoch)[T.subacute]    0.1191    0.2191 0.3476 0.0116
 rate_neg            limb  C(epoch)[T.chronic]    0.3042    0.3654 0.0090 0.0000
 rate_neg       orofacial    C(epoch)[T.acute]    0.7693    0.7697 0.0000 0.0000
 rate_neg       orofacial C(epoch)[T.subacute]    0.4664    0.4550 0.0000 0.0000
 rate_neg       orofacial  C(epoch)[T.chronic]    0.2914    0.2894 0.0000 0.0000
 rate_neg  orofacial_left    C(epoch)[T.acute]    0.7857    0.7837 0.0000 0.0000
 rate_neg  orofacial_left C(epoch)[T.subacute]    0.4364    0.4373 0.0000 0.0000
 rate_neg  orofacial_left  C(epoch)[T.chronic]    0.3003    0.3036 0.0000 0.0000
 rate_neg orofacial_right    C(epoch)[T.acute]    0.7188    0.7229 0.0000 0.0000
 rate_neg orofacial_right C(epoch)[T.subacute]    0.4481    0.4517 0.0000 0.0000
 rate_neg orofacial_right  C(epoch)[T.chronic]    0.2490    0.2538 0.0000 0.0000
v2_quiet_lp_20261117: chosen k (median of the leave-one-animal-out choices, positive events): {'orofacial': 3.75, 'limb': 8.75}
v2_quiet_lp_c1guard_k375_20261117: chosen k (median of the leave-one-animal-out choices, positive events): {'orofacial': 3.75, 'limb': 3.75} [user override]
```
