PROJ=dawn_triangle

CPP=em++

CPPFLAGS=-sUSE_WEBGPU=1 -std=c++17

$(PROJ).js: $(PROJ).cpp
	$(CPP) $(CPPFLAGS) -o $@ $^

.PHONY: clean

clean:
	rm $(PROJ).js $(PROJ).wasm
