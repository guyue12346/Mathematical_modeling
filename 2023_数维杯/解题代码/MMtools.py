import math

def computeC_r1(R_1,R_2,u_a,R_3,u_cd):
    return R_1+R_2*u_a+R_3*u_cd

def computeC_p1(R_4,R_5,u_e,R_6,u_c,R_7,u_cd):
    return R_4+R_5*u_e+R_6*u_c+R_7*u_cd

def computeC_y1(R_8,R_9,u_cd):
    return R_8+R_9*u_cd

def computeC_p3(R_10,R_11,u_eh):
    return R_10+R_11*u_eh

def computeC_ry4(R_12,R_13,u_av):
    return R_12+R_13*u_av

R_1=0.000413
R_2=0.00052
R_3=-0.000295
R_4=0.0059
R_5=0.0007193
R_6=0.00055
R_7=-0.000001
R_8=0.0001
R_9=0.00008
R_10=-0.0001
R_11=-0.00019
R_12=-0.00027
R_13=-0.000046


u_c=0
u_cd=-2.1552
u_e=-3.4817
u_a=-2.0743
u_t=0
u_eh=-0.0000009772
u_av=0.00000041869

C_r1=computeC_r1(R_1,R_2,u_a,R_3,u_cd)
C_p1=computeC_p1(R_4,R_5,u_e,R_6,u_c,R_7,u_cd)
C_y1=computeC_y1(R_8,R_9,u_cd)
C_p3=computeC_p3(R_10,R_11,u_eh)
C_ry4=computeC_ry4(R_12,R_13,u_av)

M_r2 = 20.96*u_t*u_t*u_t - 211.26*u_t*u_t + 1041.6*u_t - 678.57