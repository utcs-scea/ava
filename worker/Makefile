targets = manager_tcp manager

all: $(targets)


%: %.c ../common/socket.c
	gcc -g -Iinclude \
		  $^ -o $@ \
		  `pkg-config --cflags --libs glib-2.0` \
		  -pthread -lrt -ldl -D_GNU_SOURCE \
		  -Wall

clean:
	rm -f $(targets)
