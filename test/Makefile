targets = micro_benchmark trivial_benchmark trivial_test libtrivial.so

all: $(targets)

%.so: %.c
	gcc -shared -fPIC -g -O0 -I. \
		  $^ -o $@ \
		  -D_GNU_SOURCE \
		  -Wall


%: %.c libtrivial.so
	gcc -g -O0 -I.  -Wl,-rpath -Wl,. \
		  $^ -o $@ \
		  -D_GNU_SOURCE \
		  -Wall -lpthread -L. -ltrivial

clean:
	rm -f $(targets)
