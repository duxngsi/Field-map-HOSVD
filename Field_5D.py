
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from HOSVD import DataSVDCompress
import time
matplotlib.rcParams['font.size'] = '14'
myfsize=14

Field_5D=np.load('Field_5D.npy')    #original field is stored in 5D tensor, saved as binary file "Field_5D.npy"

usefield=np.asarray(Field_5D)

t0=time.time()
decomposed_data=DataSVDCompress(usefield,(3,8,4,4,3))
decomposed_data.compress_ndm()
print(time.time()-t0)
decomposed_data.recover()
a=decomposed_data.original_data[44,:,:,21,2]
b=decomposed_data.recovered_data[44,:,:,21,2]
scalez=a.max()
a=a/scalez
b=b/scalez    #共用一个放大倍数
c=decomposed_data.original_data[44,:,:,21,1]
d=decomposed_data.recovered_data[44,:,:,21,1]
scaley=c.max()
c=c/scaley
d=d/scaley
plt.rc('font', family='serif')

#==============================================
srb=0.01
from mpl_toolkits.axes_grid1 import ImageGrid

figt = plt.figure()
#============Ey,Ez,normalize
grid1 = ImageGrid(figt, 121,          # as in plt.subplot(111)
                 nrows_ncols=(1,2),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="10%",
                 cbar_pad=0.15,
                 )
grid1[0].set_ylabel('z-step', fontsize=myfsize)
grid1[0].set_xlabel('x-step', fontsize=myfsize)
grid1[0].imshow(a,vmax=1.0, vmin=-1.0,origin='lower')
grid1[0].set_title('$E_z$')
im=grid1[1].imshow(c,vmax=1.0, vmin=-1.0,origin='lower')
grid1[1].set_title('$E_y$')


#============Ey differences,Ez differences,normalize
grid1[1].cax.colorbar(im)
grid1[1].cax.toggle_label(True)
grid2 = ImageGrid(figt, 122,          # as in plt.subplot(111)
                 nrows_ncols=(1,2),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="10%",
                 cbar_pad=0.15,
                 )
grid2[0].set_ylabel('z-step', fontsize=myfsize)
grid2[0].set_xlabel('x-step', fontsize=myfsize)
grid2[0].imshow((a-b)/a.max(),vmax=srb,vmin=-srb,cmap='RdBu',origin='lower')
grid2[0].set_title('$\Delta E_z$')
im=grid2[1].imshow((c-d)/c.max(),vmax=srb,vmin=-srb,cmap='RdBu',origin='lower')
grid2[1].set_title('$\Delta E_y$')
cbar=grid2[1].cax.colorbar(im,ticks=[-srb,0,srb], format='%.0e')
grid2[1].cax.toggle_label(True)
cbar.ax.set_yticklabels(['< -'+str(srb*100)+'%', '0', '> '+str(srb*100)+'%'])  # vertically oriented colorba
#========show over
#=======prepare data for smoothness check
ez1=decomposed_data.original_data[12,:,10,10,2]
ez2=decomposed_data.recovered_data[12,:,10,10,2]
ey1=decomposed_data.original_data[12,:,10,10,1]
ey2=decomposed_data.recovered_data[12,:,10,10,1]

dez1=np.zeros(201)
dez2=np.zeros(201)
dey1=np.zeros(201)
dey2=np.zeros(201)
for i in range(200):
    dez1[i]=ez1[i+1]-ez1[i]
    dez2[i]=ez2[i+1]-ez2[i]
    dey1[i]=ey1[i+1]-ey1[i]
    dey2[i]=ey2[i+1]-ey2[i]

#==========================================smoothness of Ez/dz

fig2=plt.figure()
u1=fig2.add_subplot(211)
Loriginal,=plt.plot(dez1/3e5,label='original-data')
Lrecovered,=plt.plot(dez2/3e5,label='RC-data')
plt.legend(handles=[Loriginal,Lrecovered],loc=1, fontsize=myfsize)
plt.tick_params(axis='both',labelsize=myfsize)
#u1.set_xlabel('z')
u1.set_ylabel(r'$δE_z/δz \ [arb.unit]$',fontsize=myfsize)
plt.setp(u1.get_xticklabels(), visible=False)

u2=fig2.add_subplot(212,sharex=u1)
Loriginal2,=plt.plot(dey1/6e4,label='original-data')
Lrecovered2,=plt.plot(dey2/6e4,label='RC-data')
plt.legend(handles=[Loriginal2,Lrecovered2],loc=4, fontsize=myfsize)
plt.tick_params(axis='both',labelsize=myfsize)
u2.set_xlabel('z-steps',fontsize=myfsize)
u2.set_ylabel(r'$δE_y/δz \ [arb.unit]$', fontsize=myfsize)

np.save('data0',decomposed_data.original_data)
data1=[]
data1.append(decomposed_data.v)
data1.append(decomposed_data.compressed_data)
np.save('data1',data1)
plt.savefig("dfasfsf.eps",bbox_inches='tight')

#==================================================compare singular values
fig_s=plt.figure()
s1=fig_s.add_subplot(311)
plt.plot(decomposed_data.s[0][0:15], '.',label=r'singular values of $U^{(1)}$')
plt.legend(fontsize=myfsize)
plt.yscale('log')
s1.set_xlabel('',fontsize=myfsize)
s1.set_ylabel('',fontsize=myfsize)
plt.setp(s1.get_xticklabels(), fontsize=myfsize)
plt.setp(s1.get_yticklabels(), fontsize=myfsize)
plt.title('singular values',fontsize=myfsize)

s2=fig_s.add_subplot(312)
plt.plot(decomposed_data.s[1][0:30], '.',label=r'singular values of $U^{(2)}$')
plt.legend(fontsize=myfsize)
plt.yscale('log')
plt.setp(s2.get_xticklabels(), fontsize=myfsize)
plt.setp(s2.get_yticklabels(), fontsize=myfsize)


s3=fig_s.add_subplot(313)
plt.plot(decomposed_data.s[2][0:10], '.',label=r'singular values of $U^{(3)}$')
plt.legend(fontsize=myfsize)
plt.yscale('log')
plt.setp(s3.get_xticklabels(), fontsize=myfsize)
plt.setp(s3.get_yticklabels(), fontsize=myfsize)

plt.show()

#===========================
