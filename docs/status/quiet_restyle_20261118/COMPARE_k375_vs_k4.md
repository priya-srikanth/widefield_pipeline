```
sessions: old 108, new 108

== quiet_min: per-animal epoch means and change vs own pre (old -> new)
           pre           acute         subacute         chronic         acute-pre         subacute-pre         chronic-pre        
           old     new     old     new      old     new     old     new       old     new          old     new         old     new
animal                                                                                                                            
PS92     4.995   4.995  11.614  11.614    3.541   3.541   4.045   4.045     6.619   6.619       -1.454  -1.454      -0.950  -0.950
PS93     9.014   9.014  13.501  13.501    6.363   6.363   6.200   6.200     4.487   4.487       -2.650  -2.650      -2.814  -2.814
PS94    10.375  10.375  22.485  22.485    8.710   8.710   8.119   8.119    12.111  12.111       -1.665  -1.665      -2.256  -2.256
PS95    45.194  45.194  53.829  53.829   24.716  24.716  22.508  22.508     8.635   8.635      -20.478 -20.478     -22.686 -22.686

== orofacial|rate: per-animal epoch means and change vs own pre (old -> new)
          pre          acute         subacute         chronic         acute-pre         subacute-pre        chronic-pre       
          old    new     old     new      old     new     old     new       old     new          old    new         old    new
animal                                                                                                                        
PS92    4.733  4.079  16.201  14.544   13.838  11.786  13.222  11.941    11.468  10.465        9.105  7.707       8.489  7.862
PS93    5.340  4.626  23.729  22.074    9.995   8.663   5.145   4.391    18.389  17.448        4.655  4.037      -0.195 -0.235
PS94    5.628  4.881  14.667  13.149    9.114   7.948   3.830   3.037     9.038   8.269        3.486  3.068      -1.799 -1.844
PS95    1.956  1.562  10.459   9.344    4.864   4.030   3.099   2.416     8.503   7.782        2.908  2.467       1.143  0.854

== orofacial|rate_both: per-animal epoch means and change vs own pre (old -> new)
          pre          acute         subacute         chronic         acute-pre         subacute-pre         chronic-pre        
          old    new     old     new      old     new     old     new       old     new          old     new         old     new
animal                                                                                                                          
PS92    8.516  7.225  35.245  32.081   25.604  21.868  24.840  22.031    26.729  24.856       17.088  14.643      16.323  14.806
PS93    9.877  8.659  47.348  43.820   20.631  18.029  12.678  10.696    37.471  35.161       10.755   9.370       2.801   2.037
PS94    8.273  7.168  30.045  27.048   18.283  16.111   7.545   6.029    21.771  19.880       10.010   8.943      -0.729  -1.139
PS95    4.906  3.952  26.361  23.277   10.085   8.491   6.980   5.726    21.455  19.325        5.179   4.539       2.074   1.774

== limb|rate: per-animal epoch means and change vs own pre (old -> new)
          pre          acute         subacute         chronic         acute-pre        subacute-pre        chronic-pre        
          old    new     old     new      old     new     old     new       old    new          old    new         old     new
animal                                                                                                                        
PS92    5.242  4.525  12.225  10.351   11.300   9.783  18.735  17.365     6.983  5.826        6.058  5.258      13.493  12.840
PS93    4.580  3.843  12.640  11.211    3.965   3.253   7.894   7.049     8.060  7.367       -0.615 -0.590       3.314   3.206
PS94    8.983  8.116   8.087   7.307   11.659  10.157   9.897   9.024    -0.896 -0.809        2.677  2.042       0.914   0.908
PS95    3.854  3.233   4.124   3.437    3.014   2.638   2.322   1.993     0.271  0.204       -0.839 -0.595      -1.532  -1.240

== LMM epoch terms vs August pre (rate = log10(rate + 0.2)), old -> new
  measure          column                 term  coef_old  coef_new  q_old  q_new
quiet_min         session    C(epoch)[T.acute]    4.7772    4.7772 0.3429 0.3429
quiet_min         session C(epoch)[T.subacute]   -6.4716   -6.4716 0.1969 0.1969
quiet_min         session  C(epoch)[T.chronic]   -7.1945   -7.1945 0.1343 0.1343
     rate            limb    C(epoch)[T.acute]    0.1651    0.1677 0.2877 0.2745
     rate            limb C(epoch)[T.subacute]    0.1539    0.1556 0.3369 0.3535
     rate            limb  C(epoch)[T.chronic]    0.1848    0.2045 0.2614 0.2311
     rate       orofacial    C(epoch)[T.acute]    0.5860    0.6058 0.0000 0.0000
     rate       orofacial C(epoch)[T.subacute]    0.3616    0.3671 0.0003 0.0009
     rate       orofacial  C(epoch)[T.chronic]    0.1298    0.1201 0.2614 0.2340
     rate  orofacial_left    C(epoch)[T.acute]    0.6083    0.6521 0.0000 0.0000
     rate  orofacial_left C(epoch)[T.subacute]    0.3576    0.3863 0.0000 0.0000
     rate  orofacial_left  C(epoch)[T.chronic]    0.1526    0.1558 0.0767 0.0885
     rate orofacial_right    C(epoch)[T.acute]    0.5566    0.5983 0.0000 0.0000
     rate orofacial_right C(epoch)[T.subacute]    0.3239    0.3427 0.0007 0.0009
     rate orofacial_right  C(epoch)[T.chronic]    0.0786    0.0884 0.3139 0.2971
rate_both            limb    C(epoch)[T.acute]    0.2682    0.2678 0.0394 0.0430
rate_both            limb C(epoch)[T.subacute]    0.1737    0.1726 0.1892 0.1898
rate_both            limb  C(epoch)[T.chronic]    0.2306    0.2485 0.0632 0.0481
rate_both       orofacial    C(epoch)[T.acute]    0.6626    0.6923 0.0000 0.0000
rate_both       orofacial C(epoch)[T.subacute]    0.4107    0.4237 0.0000 0.0000
rate_both       orofacial  C(epoch)[T.chronic]    0.2011    0.1969 0.0112 0.0189
rate_both  orofacial_left    C(epoch)[T.acute]    0.6910    0.7364 0.0000 0.0000
rate_both  orofacial_left C(epoch)[T.subacute]    0.3933    0.4178 0.0000 0.0000
rate_both  orofacial_left  C(epoch)[T.chronic]    0.2248    0.2315 0.0002 0.0003
rate_both orofacial_right    C(epoch)[T.acute]    0.6282    0.6704 0.0000 0.0000
rate_both orofacial_right C(epoch)[T.subacute]    0.3942    0.4122 0.0000 0.0000
rate_both orofacial_right  C(epoch)[T.chronic]    0.1606    0.1683 0.0451 0.0453
 rate_neg            limb    C(epoch)[T.acute]    0.4644    0.4637 0.0000 0.0000
 rate_neg            limb C(epoch)[T.subacute]    0.2191    0.2215 0.0116 0.0127
 rate_neg            limb  C(epoch)[T.chronic]    0.3654    0.3856 0.0000 0.0000
 rate_neg       orofacial    C(epoch)[T.acute]    0.7697    0.7944 0.0000 0.0000
 rate_neg       orofacial C(epoch)[T.subacute]    0.4550    0.4830 0.0000 0.0000
 rate_neg       orofacial  C(epoch)[T.chronic]    0.2894    0.2811 0.0000 0.0005
 rate_neg  orofacial_left    C(epoch)[T.acute]    0.7837    0.8140 0.0000 0.0000
 rate_neg  orofacial_left C(epoch)[T.subacute]    0.4373    0.4446 0.0000 0.0000
 rate_neg  orofacial_left  C(epoch)[T.chronic]    0.3036    0.2947 0.0000 0.0000
 rate_neg orofacial_right    C(epoch)[T.acute]    0.7229    0.7730 0.0000 0.0000
 rate_neg orofacial_right C(epoch)[T.subacute]    0.4517    0.4755 0.0000 0.0000
 rate_neg orofacial_right  C(epoch)[T.chronic]    0.2538    0.2660 0.0000 0.0000
v2_quiet_lp_c1guard_k375_20261117: chosen k (median of the leave-one-animal-out choices, positive events): {'orofacial': 3.75, 'limb': 3.75} [user override]
v2_quiet_lp_c1guard_k4_20261118: chosen k (median of the leave-one-animal-out choices, positive events): {'orofacial': 4.0, 'limb': 4.0} [user override]
```
