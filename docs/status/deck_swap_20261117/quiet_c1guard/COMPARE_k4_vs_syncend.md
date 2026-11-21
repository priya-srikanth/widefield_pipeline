```
sessions: old 108, new 108

== quiet_min: per-animal epoch means and change vs own pre (old -> new)
           pre           acute         subacute         chronic         acute-pre         subacute-pre         chronic-pre        
           old     new     old     new      old     new     old     new       old     new          old     new         old     new
animal                                                                                                                            
PS92     4.995   4.659  11.614  11.614    3.541   3.314   4.045   3.232     6.619   6.955       -1.454  -1.344      -0.950  -1.427
PS93     9.014   7.444  13.501  13.486    6.363   6.342   6.200   6.032     4.487   6.042       -2.650  -1.102      -2.814  -1.411
PS94    10.375  10.186  22.485  22.440    8.710   8.710   8.119   7.914    12.111  12.255       -1.665  -1.476      -2.256  -2.271
PS95    45.194  44.926  53.829  53.781   24.716  24.670  22.508  22.376     8.635   8.855      -20.478 -20.256     -22.686 -22.550

== orofacial|rate: per-animal epoch means and change vs own pre (old -> new)
          pre          acute         subacute         chronic         acute-pre         subacute-pre        chronic-pre       
          old    new     old     new      old     new     old     new       old     new          old    new         old    new
animal                                                                                                                        
PS92    4.079  3.823  14.544  15.550   11.786  12.245  11.941  12.521    10.465  11.727        7.707  8.422       7.862  8.698
PS93    4.626  4.599  22.074  25.870    8.663  11.641   4.391   5.916    17.448  21.271        4.037  7.043      -0.235  1.318
PS94    4.881  4.782  13.149  13.164    7.948   7.948   3.037   2.976     8.269   8.383        3.068  3.167      -1.844 -1.806
PS95    1.562  1.547   9.344   9.316    4.030   4.060   2.416   2.390     7.782   7.769        2.467  2.513       0.854  0.843

== orofacial|rate_both: per-animal epoch means and change vs own pre (old -> new)
          pre          acute         subacute         chronic         acute-pre         subacute-pre         chronic-pre        
          old    new     old     new      old     new     old     new       old     new          old     new         old     new
animal                                                                                                                          
PS92    7.225  6.775  32.081  34.230   21.868  23.869  22.031  22.728    24.856  27.456       14.643  17.094      14.806  15.953
PS93    8.659  7.975  43.820  51.467   18.029  23.845  10.696  14.757    35.161  43.492        9.370  15.870       2.037   6.782
PS94    7.168  6.968  27.048  27.064   16.111  16.111   6.029   5.903    19.880  20.096        8.943   9.143      -1.139  -1.065
PS95    3.952  3.900  23.277  23.261    8.491   8.530   5.726   5.706    19.325  19.361        4.539   4.629       1.774   1.805

== limb|rate: per-animal epoch means and change vs own pre (old -> new)
          pre          acute         subacute         chronic         acute-pre        subacute-pre        chronic-pre        
          old    new     old     new      old     new     old     new       old    new          old    new         old     new
animal                                                                                                                        
PS92    4.525  4.656  10.351  10.969    9.783  10.227  17.365  18.389     5.826  6.313        5.258  5.572      12.840  13.733
PS93    3.843  4.203  11.211  12.319    3.253   3.771   7.049   7.784     7.367  8.116       -0.590 -0.432       3.206   3.581
PS94    8.116  8.067   7.307   7.290   10.157  10.157   9.024   8.998    -0.809 -0.778        2.042  2.090       0.908   0.931
PS95    3.233  3.230   3.437   3.440    2.638   2.640   1.993   1.985     0.204  0.210       -0.595 -0.590      -1.240  -1.246

== LMM epoch terms vs August pre (rate = log10(rate + 0.2)), old -> new
  measure          column                 term  coef_old  coef_new  q_old  q_new
quiet_min         session    C(epoch)[T.acute]    4.7772    5.3598 0.3429 0.2922
quiet_min         session C(epoch)[T.subacute]   -6.4716   -5.9892 0.1969 0.2376
quiet_min         session  C(epoch)[T.chronic]   -7.1945   -6.9313 0.1343 0.1546
     rate            limb    C(epoch)[T.acute]    0.1677    0.1768 0.2745 0.2674
     rate            limb C(epoch)[T.subacute]    0.1556    0.1681 0.3535 0.3284
     rate            limb  C(epoch)[T.chronic]    0.2045    0.2130 0.2311 0.1712
     rate       orofacial    C(epoch)[T.acute]    0.6058    0.6443 0.0000 0.0000
     rate       orofacial C(epoch)[T.subacute]    0.3671    0.4169 0.0009 0.0007
     rate       orofacial  C(epoch)[T.chronic]    0.1201    0.1696 0.2340 0.1712
     rate  orofacial_left    C(epoch)[T.acute]    0.6521    0.6832 0.0000 0.0000
     rate  orofacial_left C(epoch)[T.subacute]    0.3863    0.4445 0.0000 0.0000
     rate  orofacial_left  C(epoch)[T.chronic]    0.1558    0.2209 0.0885 0.0500
     rate orofacial_right    C(epoch)[T.acute]    0.5983    0.6284 0.0000 0.0000
     rate orofacial_right C(epoch)[T.subacute]    0.3427    0.3909 0.0009 0.0005
     rate orofacial_right  C(epoch)[T.chronic]    0.0884    0.1349 0.2971 0.1712
rate_both            limb    C(epoch)[T.acute]    0.2678    0.2852 0.0430 0.0419
rate_both            limb C(epoch)[T.subacute]    0.1726    0.1890 0.1898 0.1739
rate_both            limb  C(epoch)[T.chronic]    0.2485    0.2642 0.0481 0.0450
rate_both       orofacial    C(epoch)[T.acute]    0.6923    0.7317 0.0000 0.0000
rate_both       orofacial C(epoch)[T.subacute]    0.4237    0.4781 0.0000 0.0000
rate_both       orofacial  C(epoch)[T.chronic]    0.1969    0.2516 0.0189 0.0158
rate_both  orofacial_left    C(epoch)[T.acute]    0.7364    0.7714 0.0000 0.0000
rate_both  orofacial_left C(epoch)[T.subacute]    0.4178    0.4925 0.0000 0.0000
rate_both  orofacial_left  C(epoch)[T.chronic]    0.2315    0.2877 0.0003 0.0093
rate_both orofacial_right    C(epoch)[T.acute]    0.6704    0.7184 0.0000 0.0000
rate_both orofacial_right C(epoch)[T.subacute]    0.4122    0.4766 0.0000 0.0000
rate_both orofacial_right  C(epoch)[T.chronic]    0.1683    0.2233 0.0453 0.0349
 rate_neg            limb    C(epoch)[T.acute]    0.4637    0.4943 0.0000 0.0000
 rate_neg            limb C(epoch)[T.subacute]    0.2215    0.2431 0.0127 0.0151
 rate_neg            limb  C(epoch)[T.chronic]    0.3856    0.4106 0.0000 0.0000
 rate_neg       orofacial    C(epoch)[T.acute]    0.7944    0.8323 0.0000 0.0000
 rate_neg       orofacial C(epoch)[T.subacute]    0.4830    0.5456 0.0000 0.0000
 rate_neg       orofacial  C(epoch)[T.chronic]    0.2811    0.3426 0.0005 0.0003
 rate_neg  orofacial_left    C(epoch)[T.acute]    0.8140    0.8633 0.0000 0.0000
 rate_neg  orofacial_left C(epoch)[T.subacute]    0.4446    0.5055 0.0000 0.0000
 rate_neg  orofacial_left  C(epoch)[T.chronic]    0.2947    0.3541 0.0000 0.0000
 rate_neg orofacial_right    C(epoch)[T.acute]    0.7730    0.8228 0.0000 0.0000
 rate_neg orofacial_right C(epoch)[T.subacute]    0.4755    0.5696 0.0000 0.0000
 rate_neg orofacial_right  C(epoch)[T.chronic]    0.2660    0.3279 0.0000 0.0017
v2_quiet_lp_c1guard_k4_20261118: chosen k (median of the leave-one-animal-out choices, positive events): {'orofacial': 4.0, 'limb': 4.0} [user override]
v2_quiet_lp_c1guard_syncend_20261118: chosen k (median of the leave-one-animal-out choices, positive events): {'orofacial': 4.0, 'limb': 4.0} [user override]
```
