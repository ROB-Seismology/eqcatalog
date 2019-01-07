#FC	= f77 
FC	= gfortran
#FFLAGS	=  -g  -Nl50
FFLAGS	= -std=gnu

cluster2000x:   cluster2000x.o
	$(FC) $(FFLAGS)  -o cluster2000x cluster2000x.o

zero:   zero.o
	$(FC) $(FFLAGS) -o zero zero.o


test_tdif2:   test_tdif2.o
	$(FC) $(FFLAGS) -o test_tdif2 test_tdif2.o

