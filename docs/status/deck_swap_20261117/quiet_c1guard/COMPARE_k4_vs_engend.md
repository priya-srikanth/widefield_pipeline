```
sessions: old 108, new 108

== quiet_min: per-animal epoch means and change vs own pre (old -> new)
           pre           acute         subacute         chronic         acute-pre         subacute-pre         chronic-pre        
           old     new     old     new      old     new     old     new       old     new          old     new         old     new
animal                                                                                                                            
PS92     4.995   4.657  11.614  11.614    3.541   3.265   4.045   3.224     6.619   6.958       -1.454  -1.391      -0.950  -1.433
PS93     9.014   7.440  13.501  13.481    6.363   6.342   6.200   6.032     4.487   6.041       -2.650  -1.098      -2.814  -1.408
PS94    10.375  10.173  22.485  22.440    8.710   8.710   8.119   7.906    12.111  12.267       -1.665  -1.464      -2.256  -2.268
PS95    45.194  44.916  53.829  53.757   24.716  24.664  22.508  22.367     8.635   8.841      -20.478 -20.251     -22.686 -22.549

== orofacial|rate: per-animal epoch means and change vs own pre (old -> new)
          pre          acute         subacute         chronic         acute-pre         subacute-pre        chronic-pre       
          old    new     old     new      old     new     old     new       old     new          old    new         old    new
animal                                                                                                                        
PS92    4.079  3.824  14.544  15.495   11.786  12.320  11.941  12.532    10.465  11.671        7.707  8.496       7.862  8.707
PS93    4.626  4.595  22.074  25.876    8.663  11.641   4.391   5.917    17.448  21.281        4.037  7.046      -0.235  1.322
PS94    4.881  4.787  13.149  13.164    7.948   7.948   3.037   2.951     8.269   8.377        3.068  3.161      -1.844 -1.836
PS95    1.562  1.553   9.344   9.338    4.030   4.060   2.416   2.391     7.782   7.786        2.467  2.508       0.854  0.838

== orofacial|rate_both: per-animal epoch means and change vs own pre (old -> new)
          pre          acute         subacute         chronic         acute-pre         subacute-pre         chronic-pre        
          old    new     old     new      old     new     old     new       old     new          old     new         old     new
animal                                                                                                                          
PS92    7.225  6.777  32.081  34.176   21.868  23.904  22.031  22.746    24.856  27.398       14.643  17.126      14.806  15.969
PS93    8.659  7.972  43.820  51.478   18.029  23.845  10.696  14.759    35.161  43.506        9.370  15.873       2.037   6.787
PS94    7.168  6.977  27.048  27.064   16.111  16.111   6.029   5.868    19.880  20.087        8.943   9.134      -1.139  -1.109
PS95    3.952  3.906  23.277  23.290    8.491   8.531   5.726   5.707    19.325  19.384        4.539   4.625       1.774   1.801

== limb|rate: per-animal epoch means and change vs own pre (old -> new)
          pre          acute         subacute         chronic         acute-pre        subacute-pre        chronic-pre        
          old    new     old     new      old     new     old     new       old    new          old    new         old     new
animal                                                                                                                        
PS92    4.525  4.658  10.351  10.969    9.783  10.257  17.365  18.428     5.826  6.311        5.258  5.599      12.840  13.770
PS93    3.843  4.204  11.211  12.314    3.253   3.771   7.049   7.785     7.367  8.109       -0.590 -0.433       3.206   3.581
PS94    8.116  8.041   7.307   7.290   10.157  10.157   9.024   8.972    -0.809 -0.751        2.042  2.116       0.908   0.931
PS95    3.233  3.231   3.437   3.441    2.638   2.640   1.993   1.983     0.204  0.211       -0.595 -0.590      -1.240  -1.248

== LMM epoch terms vs August pre (rate = log10(rate + 0.2)), old -> new
  measure          column                 term  coef_old  coef_new  q_old  q_new
quiet_min         session    C(epoch)[T.acute]    4.7772    5.3643 0.3429 0.2913
quiet_min         session C(epoch)[T.subacute]   -6.4716   -5.9902 0.1969 0.2371
quiet_min         session  C(epoch)[T.chronic]   -7.1945   -6.9308 0.1343 0.1543
     rate            limb    C(epoch)[T.acute]    0.1677    0.1772 0.2745 0.2676
     rate            limb C(epoch)[T.subacute]    0.1556    0.1687 0.3535 0.3265
     rate            limb  C(epoch)[T.chronic]    0.2045    0.2134 0.2311 0.1725
     rate       orofacial    C(epoch)[T.acute]    0.6058    0.6432 0.0000 0.0000
     rate       orofacial C(epoch)[T.subacute]    0.3671    0.4168 0.0009 0.0008
     rate       orofacial  C(epoch)[T.chronic]    0.1201    0.1689 0.2340 0.1725
     rate  orofacial_left    C(epoch)[T.acute]    0.6521    0.6831 0.0000 0.0000
     rate  orofacial_left C(epoch)[T.subacute]    0.3863    0.4449 0.0000 0.0000
     rate  orofacial_left  C(epoch)[T.chronic]    0.1558    0.2207 0.0885 0.0516
     rate orofacial_right    C(epoch)[T.acute]    0.5983    0.6279 0.0000 0.0000
     rate orofacial_right C(epoch)[T.subacute]    0.3427    0.3909 0.0009 0.0006
     rate orofacial_right  C(epoch)[T.chronic]    0.0884    0.1342 0.2971 0.1725
rate_both            limb    C(epoch)[T.acute]    0.2678    0.2853 0.0430 0.0419
rate_both            limb C(epoch)[T.subacute]    0.1726    0.1901 0.1898 0.1731
rate_both            limb  C(epoch)[T.chronic]    0.2485    0.2648 0.0481 0.0454
rate_both       orofacial    C(epoch)[T.acute]    0.6923    0.7312 0.0000 0.0000
rate_both       orofacial C(epoch)[T.subacute]    0.4237    0.4780 0.0000 0.0000
rate_both       orofacial  C(epoch)[T.chronic]    0.1969    0.2511 0.0189 0.0164
rate_both  orofacial_left    C(epoch)[T.acute]    0.7364    0.7713 0.0000 0.0000
rate_both  orofacial_left C(epoch)[T.subacute]    0.4178    0.4927 0.0000 0.0000
rate_both  orofacial_left  C(epoch)[T.chronic]    0.2315    0.2875 0.0003 0.0095
rate_both orofacial_right    C(epoch)[T.acute]    0.6704    0.7179 0.0000 0.0000
rate_both orofacial_right C(epoch)[T.subacute]    0.4122    0.4766 0.0000 0.0000
rate_both orofacial_right  C(epoch)[T.chronic]    0.1683    0.2229 0.0453 0.0339
 rate_neg            limb    C(epoch)[T.acute]    0.4637    0.4940 0.0000 0.0000
 rate_neg            limb C(epoch)[T.subacute]    0.2215    0.2449 0.0127 0.0146
 rate_neg            limb  C(epoch)[T.chronic]    0.3856    0.4115 0.0000 0.0000
 rate_neg       orofacial    C(epoch)[T.acute]    0.7944    0.8321 0.0000 0.0000
 rate_neg       orofacial C(epoch)[T.subacute]    0.4830    0.5454 0.0000 0.0000
 rate_neg       orofacial  C(epoch)[T.chronic]    0.2811    0.3425 0.0005 0.0004
 rate_neg  orofacial_left    C(epoch)[T.acute]    0.8140    0.8631 0.0000 0.0000
 rate_neg  orofacial_left C(epoch)[T.subacute]    0.4446    0.5056 0.0000 0.0000
 rate_neg  orofacial_left  C(epoch)[T.chronic]    0.2947    0.3540 0.0000 0.0000
 rate_neg orofacial_right    C(epoch)[T.acute]    0.7730    0.8222 0.0000 0.0000
 rate_neg orofacial_right C(epoch)[T.subacute]    0.4755    0.5695 0.0000 0.0000
 rate_neg orofacial_right  C(epoch)[T.chronic]    0.2660    0.3281 0.0000 0.0019
v2_quiet_lp_c1guard_k4_20261118: chosen k (median of the leave-one-animal-out choices, positive events): {'orofacial': 4.0, 'limb': 4.0} [user override]
v2_quiet_lp_c1guard_engend_20261118: chosen k (median of the leave-one-animal-out choices, positive events): {'orofacial': 4.0, 'limb': 4.0} [user override]
```
