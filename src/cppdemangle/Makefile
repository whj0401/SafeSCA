all:
	gcc -c -fPIC ./cp-demangle.c -I. -I./include -o ./cp-demangle.o
	gcc -shared ./cp-demangle.o -o ./cp-demangle.so
	rm ./cp-demangle.o

clean:
	rm ./cp-demangle.so
	rm ./cp-demangle.o

