CC = mpicc
CPP = mpic++
LDC = mpicc
LD_FLAGS = -lm -fopenmp -lstdc++ -lopencv_core -lopencv_highgui -lopencv_imgproc
FLAGS = -fopenmp
CPPFLAGS= -I/usr/include/opencv $(FLAGS)
PROG = PA4.x
RM= /bin/rm


#all rule

OBJS=PA4.o
TOOLS=imageTools.o

all: $(PROG)

$(PROG): $(OBJS) $(TOOLS)
	$(LDC)  $^ $(LD_FLAGS) -o $@

%.o: %.c
	$(CC) $(FLAGS) -c $^ -o $@
%.o: %.cpp
	$(CPP) $(CPPFLAGS) -c $^ -o $@

#clean rule
clean:
	$(RM) -rf *.o *.x *.cx *.mod
